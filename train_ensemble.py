"""
train_ensemble.py
Implements an ensemble of mmBERT models using K-Fold Cross-Validation.
Averages the predicted probabilities (soft voting) across all folds.
"""

import os
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from tqdm import tqdm
import yaml
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set random seeds to prevent identical initialization across folds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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

class ConfigLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.TRAIN_TEXT = self.cfg['data']['train_text']
        self.TRAIN_LABEL = self.cfg['data']['train_label']
        self.VAL_TEXT = self.cfg['data']['val_text']
        self.VAL_LABEL = self.cfg['data']['val_label']
        self.TEST_TEXT = self.cfg['data']['test_text']
        
        self.MAX_LENGTH = self.cfg['training']['max_length']
        self.BATCH_SIZE = self.cfg['training']['batch_size']
        self.EPOCHS = self.cfg['training']['epochs']
        self.LEARNING_RATE = float(self.cfg['training']['learning_rate'])
        self.WEIGHT_DECAY = float(self.cfg['training']['weight_decay'])
        self.EARLY_STOPPING_PATIENCE = self.cfg['training'].get('early_stopping_patience', 5)
        self.NUM_CLASSES = self.cfg['training']['num_classes']
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.MODEL_KEY = self.cfg['models'].get('mmbert', 'jhu-clsp/mmBERT-base')
        self.LABEL_MAP = {int(k): v for k, v in self.cfg['labels'].items()}

Config = ConfigLoader()

class ToxicityDataset(Dataset):
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
    if not isinstance(text, str): return ""
    text = text.replace("#ERROR!", "").replace("#NAME?", "")
    return " ".join(text.split())

def train_epoch(model, data_loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    losses = []
    
    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if class_weights is not None:
            curr_loss_fn = FocalLoss(alpha=class_weights.to(logits.dtype), gamma=2.0)
            loss = curr_loss_fn(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': np.mean(losses)})
    
    return np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    return true_labels, predictions

def predict_proba(model, data_loader, device):
    """Returns soft probabilities for ensembling"""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting Probabilities', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    return np.vstack(all_probs)

def main():
    print(f"Device: {Config.DEVICE}")
    print(f"Base Model: {Config.MODEL_KEY}")
    
    # 1. Load Data
    train_text = pd.read_csv(Config.TRAIN_TEXT)
    train_label = pd.read_csv(Config.TRAIN_LABEL)
    t_df = pd.merge(train_text, train_label, on='index')
    
    val_text = pd.read_csv(Config.VAL_TEXT)
    val_label = pd.read_csv(Config.VAL_LABEL)
    v_df = pd.merge(val_text, val_label, on='index')
    
    # Combine training and validation sets for K-Fold
    all_train_df = pd.concat([t_df, v_df], ignore_index=True)
    all_train_df['message'] = all_train_df['message'].apply(preprocess_text)
    
    test_df = pd.read_csv(Config.TEST_TEXT)
    test_df['message'] = test_df['message'].apply(preprocess_text)
    
    print(f"Total Combined Training Samples: {len(all_train_df)}")
    
    X_all = all_train_df['message'].values
    y_all = all_train_df['label'].values.astype(int)
    
    # Global class weights based on all data
    class_counts = all_train_df['label'].value_counts().sort_index().values
    total_samples = len(all_train_df)
    class_weights_np = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(Config.DEVICE)
    
    # 2. Setup Tokenizer and Test Loader
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_KEY)
    test_dataset = ToxicityDataset(test_df['message'].values, None, tokenizer, Config.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 3. K-Fold Cross Validation
    K = 2
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    
    fold_f1_scores = []
    fold_test_probs = []
    
    best_overall_f1 = 0
    start_time = time.time()
    
    print(f"\nStarting {K}-Fold Ensemble Training...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{K}")
        print(f"{'='*40}")
        
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]
        
        train_loader = DataLoader(ToxicityDataset(X_tr, y_tr, tokenizer, Config.MAX_LENGTH), batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(ToxicityDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH), batch_size=Config.BATCH_SIZE, shuffle=False)
        
        # Initialize fresh model for this fold
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_KEY, 
            num_labels=Config.NUM_CLASSES
        ).to(Config.DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        best_fold_f1 = 0
        patience_counter = 0
        
        # Train fold
        for epoch in range(Config.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE, class_weights)
            true_labels, preds = eval_model(model, val_loader, Config.DEVICE)
            
            val_f1_macro = f1_score(true_labels, preds, average='macro')
            
            print(f"  Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val F1 (Macro): {val_f1_macro:.4f}")
            
            if val_f1_macro > best_fold_f1:
                best_fold_f1 = val_f1_macro
                patience_counter = 0
                # Save best fold model
                torch.save(model.state_dict(), f"best_model_fold_{fold+1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping triggered for fold {fold+1}")
                    break
                
        fold_f1_scores.append(best_fold_f1)
        print(f"Best Macro F1 for Fold {fold + 1}: {best_fold_f1:.4f}")
        
        # Load best folder model to make test predictions
        model.load_state_dict(torch.load(f"best_model_fold_{fold+1}.pt", weights_only=True))
        test_probs = predict_proba(model, test_loader, Config.DEVICE)
        fold_test_probs.append(test_probs)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    avg_f1 = np.mean(fold_f1_scores)
    print(f"\n========================================")
    print(f"Average CV Macro F1: {avg_f1:.4f} (+/- {np.std(fold_f1_scores):.4f})")
    print(f"Training Time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"========================================\n")
    
    # 4. Bagging: Average probabilities across folds
    print("Ensembling predictions across folds (Bagging)...")
    avg_probs = np.mean(fold_test_probs, axis=0) # Shape: (Num_Test_Samples, Num_Classes)
    final_test_preds = np.argmax(avg_probs, axis=1)
    
    # 5. Save Results
    submission = pd.DataFrame({
        'index': test_df['index'].values,
        'label': final_test_preds
    }).sort_values('index')
    
    output_file = 'predictions_mmbert_ensemble.csv'
    submission.to_csv(output_file, index=False)
    print(f"✓ Ensemble predictions saved to {output_file}")
    
    # 6. Logging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'Timestamp': timestamp,
        'Model Name': 'mmBERT-Ensemble-KFold',
        'Model Key': Config.MODEL_KEY,
        'epoch': f'{K} Folds',
        'train_loss': 'N/A',
        'train_acc': 'N/A',
        'val_loss': 'N/A',
        'val_f1_macro': avg_f1,
        'val_f1_weighted': 'N/A',
        'Max Length': Config.MAX_LENGTH,
        'Batch Size': Config.BATCH_SIZE,
        'Learning Rate': Config.LEARNING_RATE,
        'Device': str(Config.DEVICE)
    }
    
    log_df = pd.DataFrame([log_entry])
    log_file = 'training_log.csv'
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
    print(f"✓ Results appended to {log_file}")

if __name__ == "__main__":
    main()
