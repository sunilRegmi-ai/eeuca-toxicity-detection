"""
Toxicity Detection Training Script for GameTox Dataset
Multilingual Model Support & External Configuration
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
import yaml
import json
from datetime import datetime

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class ConfigLoader:
    """Loads configuration from YAML file"""
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        # Data paths
        self.TRAIN_TEXT = self.cfg['data']['train_text']
        self.TRAIN_LABEL = self.cfg['data']['train_label']
        self.VAL_TEXT = self.cfg['data']['val_text']
        self.VAL_LABEL = self.cfg['data']['val_label']
        self.TEST_TEXT = self.cfg['data']['test_text']
        self.AUGMENTED_TRAIN_TEXT = self.cfg['data']['augmented_train_text']
        self.AUGMENTED_TRAIN_LABEL = self.cfg['data']['augmented_train_label']
        self.USE_AUGMENTED = self.cfg['data']['use_augmented']
        
        # Model configurations
        self.MODELS = self.cfg['models']
        
        # Training parameters
        self.MAX_LENGTH = self.cfg['training']['max_length']
        self.BATCH_SIZE = self.cfg['training']['batch_size']
        self.EPOCHS = self.cfg['training']['epochs']
        self.LEARNING_RATE = self.cfg['training']['learning_rate']
        self.WEIGHT_DECAY = self.cfg['training']['weight_decay']
        self.EARLY_STOPPING_PATIENCE = self.cfg['training']['early_stopping_patience']
        self.NUM_CLASSES = self.cfg['training']['num_classes']
        
        # Resampling parameters
        self.RESAMPLING_ENABLED = self.cfg['data'].get('resampling', {}).get('enabled', False)
        self.UNDERSAMPLE_TARGET = self.cfg['data'].get('resampling', {}).get('undersample_target', 15000)
        self.OVERSAMPLE_MINIMUM = self.cfg['data'].get('resampling', {}).get('oversample_minimum', 1000)
        
        # Runtime settings
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GPUS = torch.cuda.device_count()
        
        # Label mapping
        self.LABEL_MAP = {int(k): v for k, v in self.cfg['labels'].items()}

# Global config instance
Config = ConfigLoader()

class ToxicityDataset(Dataset):
    """Dataset class for toxicity detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else -1
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def preprocess_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Remove common artifacts
    text = text.replace("#ERROR!", "").replace("#NAME?", "")
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text


def load_data():
    """Load and merge training, validation, and test data"""
    print("Loading data...")
    
    # Load training data based on flag
    if Config.USE_AUGMENTED:
        print("Using AUGMENTED training dataset...")
        train_text_path = Config.AUGMENTED_TRAIN_TEXT
        train_label_path = Config.AUGMENTED_TRAIN_LABEL
    else:
        print("Using ORIGINAL training dataset...")
        train_text_path = Config.TRAIN_TEXT
        train_label_path = Config.TRAIN_LABEL

    train_text = pd.read_csv(train_text_path)
    train_label = pd.read_csv(train_label_path)
    train_data = pd.merge(train_text, train_label, on='index')
    
    # Load validation data
    val_text = pd.read_csv(Config.VAL_TEXT)
    val_label = pd.read_csv(Config.VAL_LABEL)
    val_data = pd.merge(val_text, val_label, on='index')
    
    # Load test data
    test_data = pd.read_csv(Config.TEST_TEXT)
    
    # Preprocess text
    print("Preprocessing text...")
    train_data['message'] = train_data['message'].apply(preprocess_text)
    val_data['message'] = val_data['message'].apply(preprocess_text)
    test_data['message'] = test_data['message'].apply(preprocess_text)
    
    # Convert labels to integers
    train_data['label'] = train_data['label'].astype(int)
    val_data['label'] = val_data['label'].astype(int)
    
    # --- RESAMPLING LOGIC ---
    if Config.RESAMPLING_ENABLED:
        print(f"\nApplying Resampling (Target Class 0: {Config.UNDERSAMPLE_TARGET}, Min Class Size: {Config.OVERSAMPLE_MINIMUM})...")
        
        balanced_dfs = []
        for label in range(Config.NUM_CLASSES):
            class_df = train_data[train_data['label'] == label]
            if len(class_df) == 0:
                continue
                
            if label == 0:
                # Undersample majority class
                n_samples = min(len(class_df), Config.UNDERSAMPLE_TARGET)
                resampled_df = class_df.sample(n=n_samples, random_state=SEED)
                print(f"  Class {label} ({Config.LABEL_MAP[label]}): Undersampled from {len(class_df)} to {n_samples}")
            else:
                # Oversample minority classes
                if len(class_df) < Config.OVERSAMPLE_MINIMUM:
                    resampled_df = class_df.sample(n=Config.OVERSAMPLE_MINIMUM, replace=True, random_state=SEED)
                    print(f"  Class {label} ({Config.LABEL_MAP[label]}): Oversampled from {len(class_df)} to {Config.OVERSAMPLE_MINIMUM}")
                else:
                    resampled_df = class_df
                    print(f"  Class {label} ({Config.LABEL_MAP[label]}): Kept at {len(class_df)}")
            
            balanced_dfs.append(resampled_df)
        
        train_data = pd.concat(balanced_dfs, ignore_index=True)
    
    # Calculate class weights for imbalance
    class_counts = train_data['label'].value_counts().sort_index().values
    total_samples = len(train_data)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\nLabel distribution in training data:")
    print(train_data['label'].value_counts().sort_index())
    print("\nCalculated class weights:")
    for i, weight in enumerate(class_weights.cpu().numpy()):
        print(f"  {Config.LABEL_MAP[i]}: {weight:.4f}")
    
    return train_data, val_data, test_data, class_weights


def create_data_loader(texts, labels, tokenizer, batch_size, shuffle=True):
    """Create DataLoader for training/validation"""
    dataset = ToxicityDataset(
        texts=texts.values,
        labels=labels.values if labels is not None else None,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )


def train_epoch(model, data_loader, optimizer, scheduler, device, class_weights=None):
    """Train for one epoch"""
    model.train()
    losses = []
    correct_predictions = torch.tensor(0).to(device)
    total_predictions = 0
    
    # Define loss function - weights will be handled inside the loop to match dtype
    loss_fn = nn.CrossEntropyLoss(reduction='none') 
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        
        # Apply class weights with matching dtype
        if class_weights is not None:
            # Use Focal Loss if class weights are provided
            curr_loss_fn = FocalLoss(alpha=class_weights.to(logits.dtype), gamma=2.0)
            loss = curr_loss_fn(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': np.mean(losses), 'acc': (correct_predictions.float() / total_predictions).item()})
    
    return np.mean(losses), (correct_predictions.float() / total_predictions).item()


def eval_model(model, data_loader, device, class_weights=None):
    """Evaluate model on validation set"""
    model.eval()
    losses = []
    predictions = []
    true_labels = []
    
    # Define loss function - weights will be handled inside the loop
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            if class_weights is not None:
                curr_loss_fn = FocalLoss(alpha=class_weights.to(logits.dtype), gamma=2.0)
                loss = curr_loss_fn(logits, labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            losses.append(loss.item())
            
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    
    return np.mean(losses), f1_macro, f1_weighted, predictions, true_labels


def train_model(model_name, model_key, train_data, val_data, class_weights):
    """Train a single model"""
    print(f"\n{'='*80}")
    print(f"Training {model_name} ({model_key})")
    print(f"{'='*80}\n")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_key,
        num_labels=Config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model = model.to(Config.DEVICE)
    
    # Enable Multi-GPU support
    if Config.GPUS > 1:
        print(f"Using {Config.GPUS} GPUs for training")
        model = nn.DataParallel(model)
    
    # Create data loaders
    train_loader = create_data_loader(
        train_data['message'],
        train_data['label'],
        tokenizer,
        Config.BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = create_data_loader(
        val_data['message'],
        val_data['label'],
        tokenizer,
        Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Optimizer and scheduler with Weight Decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': Config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)
    
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1_macro': [],
        'val_f1_weighted': []
    }
    
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch + 1}/{Config.EPOCHS}')
        print('-' * 80)
        
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            Config.DEVICE,
            class_weights=class_weights
        )
        
        val_loss, val_f1_macro, val_f1_weighted, val_preds, val_labels = eval_model(
            model,
            val_loader,
            Config.DEVICE,
            class_weights=class_weights
        )
        
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val F1 (Macro): {val_f1_macro:.4f} | Val F1 (Weighted): {val_f1_weighted:.4f}')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)
        
        # Save best model and Early Stopping
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            # Handle DataParallel state_dict
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), f'best_model_{model_name}.pt')
            print(f'✓ Saved best model with F1 (Macro): {best_f1:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                print(f'\n! Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Load best model for final evaluation
    # Handle DataParallel for loading
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(f'best_model_{model_name}.pt'))
    
    # Final evaluation
    val_loss, val_f1_macro, val_f1_weighted, val_preds, val_labels = eval_model(
        model,
        val_loader,
        Config.DEVICE,
        class_weights=class_weights
    )
    
    print(f'\n{"="*80}')
    print(f'Final Results for {model_name}')
    print(f'{"="*80}')
    print(f'Best F1 (Macro): {val_f1_macro:.4f}')
    print(f'Best F1 (Weighted): {val_f1_weighted:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(val_labels, val_preds, target_names=[Config.LABEL_MAP[i] for i in range(Config.NUM_CLASSES)]))
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'history': history,
        'best_f1_macro': val_f1_macro,
        'best_f1_weighted': val_f1_weighted,
        'val_preds': val_preds,
        'val_labels': val_labels,
        'epoch_logs': [
            {
                'epoch': i + 1,
                'train_loss': history['train_loss'][i],
                'train_acc': history['train_acc'][i],
                'val_loss': history['val_loss'][i],
                'val_f1_macro': history['val_f1_macro'][i],
                'val_f1_weighted': history['val_f1_weighted'][i]
            } for i in range(len(history['train_loss']))
        ]
    }


def generate_predictions(model, tokenizer, test_data, output_file='predictions.csv'):
    """Generate predictions for test data"""
    print(f"\nGenerating predictions for test data...")
    
    # Ensure model is on device and handled by DataParallel if necessary
    model.eval()
    if Config.GPUS > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    
    test_loader = create_data_loader(
        test_data['message'],
        None,
        tokenizer,
        Config.BATCH_SIZE,
        shuffle=False
    )
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    # Create submission file
    submission = pd.DataFrame({
        'index': test_data['index'].values,
        'label': predictions
    })
    
    # Sort by index in ascending order
    submission = submission.sort_values('index')
    submission.to_csv(output_file, index=False)
    
    print(f"✓ Predictions saved to {output_file}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Prediction distribution:")
    print(submission['label'].value_counts().sort_index())
    
    return submission


def main():
    """Main training pipeline"""
    print(f"Device: {Config.DEVICE}")
    print(f"Number of GPUs: {Config.GPUS}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load data
    train_data, val_data, test_data, class_weights = load_data()
    
    # Store results for all models
    results = {}
    
    # Train each model
    for model_name, model_key in Config.MODELS.items():
        try:
            result = train_model(model_name, model_key, train_data, val_data, class_weights)
            results[model_name] = result
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {str(e)}")
            continue
    
    # Compare models and generate predictions
    if results:
        print(f"\n{'='*80}")
        print("Model Comparison")
        print(f"{'='*80}\n")
        
        comparison = []
        for model_name, result in results.items():
            comparison.append({
                'Model': model_name,
                'F1 (Macro)': result['best_f1_macro'],
                'F1 (Weighted)': result['best_f1_weighted']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('F1 (Macro)', ascending=False)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('model_comparison.csv', index=False)
        print(f"\n✓ Model comparison saved to model_comparison.csv")
        
        # Generate predictions using best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n{'='*80}")
        print(f"Generating final predictions using best model: {best_model_name}")
        print(f"{'='*80}")
        
        best_result = results[best_model_name]
        submission = generate_predictions(
            best_result['model'],
            best_result['tokenizer'],
            test_data,
            'predictions.csv'
        )
        
        # Save training summary
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(Config.DEVICE),
            'gpus': Config.GPUS,
            'models_trained': list(results.keys()),
            'best_model': best_model_name,
            'best_f1_macro': float(comparison_df.iloc[0]['F1 (Macro)']),
            'config': {
                'max_length': Config.MAX_LENGTH,
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'learning_rate': Config.LEARNING_RATE
            }
        }
        
        with open('training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Training summary saved to training_summary.json")

        # Save detailed training log to CSV
        detailed_logs = []
        for model_name, result in results.items():
            model_key = Config.MODELS[model_name]
            for epoch_log in result['epoch_logs']:
                log_entry = {
                    'Timestamp': summary['timestamp'],
                    'Model Name': model_name,
                    'Model Key': model_key,
                    **epoch_log,
                    'Max Length': Config.MAX_LENGTH,
                    'Batch Size': Config.BATCH_SIZE,
                    'Total Epochs': Config.EPOCHS,
                    'Learning Rate': Config.LEARNING_RATE,
                    'Weight Decay': Config.WEIGHT_DECAY,
                    'Early Stopping Patience': Config.EARLY_STOPPING_PATIENCE,
                    'Use Augmented': Config.USE_AUGMENTED,
                    'Resampling Enabled': Config.RESAMPLING_ENABLED,
                    'Undersample Target': Config.UNDERSAMPLE_TARGET,
                    'Oversample Minimum': Config.OVERSAMPLE_MINIMUM,
                    'Device': str(Config.DEVICE),
                    'Num GPUs': Config.GPUS
                }
                detailed_logs.append(log_entry)
        
        detailed_logs_df = pd.DataFrame(detailed_logs)
        detailed_logs_df.to_csv('training_log.csv', index=False)
        print(f"✓ Detailed training log saved to training_log.csv")
        
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}")
        print(f"\nFiles generated:")
        print(f"  - predictions.csv (submission file)")
        print(f"  - model_comparison.csv")
        print(f"  - training_summary.json")
        print(f"  - best_model_*.pt (saved model weights)")
        print(f"\nTo create submission zip:")
        print(f"  zip predictions.zip predictions.csv")
    else:
        print(f"\n{'!'*80}")
        print("CRITICAL ERROR: No models were successfully trained.")
        print("Please check the error messages above to identify why the models failed.")
        print(f"{'!'*80}")


if __name__ == "__main__":
    main()
