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
from torch.optim import AdamWthat 
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import warnings
import yaml
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class ConfigLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        # Data paths
        self.MERGED_TEXT = "data/merged/merged_text.csv"
        self.MERGED_LABEL = "data/merged/merged_label.csv"
        self.TEST_TEXT = self.cfg['data']['test_text']
        
        # Training parameters
        self.MAX_LENGTH = self.cfg['training']['max_length']
        self.BATCH_SIZE = self.cfg['training']['batch_size']
        self.EPOCHS = self.cfg['training']['epochs']
        self.LEARNING_RATE = self.cfg['training']['learning_rate']
        self.WEIGHT_DECAY = self.cfg['training']['weight_decay']
        self.NUM_CLASSES = self.cfg['training']['num_classes']
        
        # Runtime settings
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GPUS = torch.cuda.device_count()
        
        # Label mapping
        self.LABEL_MAP = {int(k): v for k, v in self.cfg['labels'].items()}
        
        # Model path
        self.MODEL_KEY = "jhu-clsp/mmBERT-base"

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

def load_data():
    print("Loading merged training data...")
    train_text = pd.read_csv(Config.MERGED_TEXT)
    train_label = pd.read_csv(Config.MERGED_LABEL)
    train_data = pd.merge(train_text, train_label, on='index')
    
    print("Loading test data...")
    test_data = pd.read_csv(Config.TEST_TEXT)
    
    for df in [train_data, test_data]:
        df['message'] = df['message'].apply(preprocess_text)
    
    train_data['label'] = train_data['label'].astype(int)
    
    # Calculate class weights
    class_counts = train_data['label'].value_counts().sort_index().values
    class_weights = torch.tensor(len(train_data) / (len(class_counts) * class_counts), dtype=torch.float).to(Config.DEVICE)
    
    return train_data, test_data, class_weights

def create_data_loader(texts, labels, tokenizer, batch_size, shuffle=True):
    dataset = ToxicityDataset(texts.values, labels.values if labels is not None else None, tokenizer, Config.MAX_LENGTH)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, data_loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    losses = []
    correct, total = 0, 0
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(logits.dtype) if class_weights is not None else None)
        loss = loss_fn(logits, labels)
        
        _, preds = torch.max(logits, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': np.mean(losses), 'acc': correct/total})
    
    return np.mean(losses), correct/total

def generate_predictions(model, tokenizer, test_data, output_file='predictions.csv'):
    print(f"\nGenerating predictions for test data...")
    model.eval()
    test_loader = create_data_loader(test_data['message'], None, tokenizer, Config.BATCH_SIZE, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            input_ids, attention_mask = batch['input_ids'].to(Config.DEVICE), batch['attention_mask'].to(Config.DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    submission = pd.DataFrame({'index': test_data['index'].values, 'label': predictions}).sort_values('index')
    submission.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to {output_file}")
    return submission

def main():
    print(f"Device: {Config.DEVICE}")
    train_data, test_data, class_weights = load_data()
    
    print(f"\nTraining mmBERT: {Config.MODEL_KEY}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_KEY)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_KEY, num_labels=Config.NUM_CLASSES).to(Config.DEVICE)
    if Config.GPUS > 1: model = nn.DataParallel(model)
    
    train_loader = create_data_loader(train_data['message'], train_data['label'], tokenizer, Config.BATCH_SIZE)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * Config.EPOCHS)
    
    all_logs = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch + 1}/{Config.EPOCHS}')
        loss, acc = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE, class_weights)
        
        log_entry = {
            'Timestamp': timestamp, 'Model Name': 'mmbert-merged', 'Model Key': Config.MODEL_KEY,
            'epoch': epoch + 1, 'train_loss': loss, 'train_acc': acc,
            'val_loss': None, 'val_f1_macro': None, 'val_f1_weighted': None, # No validation
            'Max Length': Config.MAX_LENGTH, 'Batch Size': Config.BATCH_SIZE,
            'Learning Rate': Config.LEARNING_RATE, 'Device': str(Config.DEVICE)
        }
        all_logs.append(log_entry)
        
    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), 'best_model_mmbert_merged.pt')
    
    # Generate predictions
    generate_predictions(model, tokenizer, test_data, 'predictions.csv')
    
    # Log results
    log_df = pd.DataFrame(all_logs)
    log_file = 'training_log.csv'
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
    print(f"✓ Results appended to {log_file}")

if __name__ == "__main__":
    main()
