"""
Hybrid Training Script: mmBERT Embeddings + Random Forest
Combines Transformer feature extraction with Random Forest classification.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import yaml
import json
from datetime import datetime

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class ConfigLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.MERGED_TEXT = "data/merged/merged_text.csv"
        self.MERGED_LABEL = "data/merged/merged_label.csv"
        self.TEST_TEXT = self.cfg['data']['test_text']
        self.MAX_LENGTH = self.cfg['training']['max_length']
        self.BATCH_SIZE = self.cfg['training']['batch_size']
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_KEY = self.cfg['models'].get('mmbert', 'jhu-clsp/mmBERT-small')
        self.NUM_CLASSES = self.cfg['training']['num_classes']
        self.LABEL_MAP = {int(k): v for k, v in self.cfg['labels'].items()}

Config = ConfigLoader()

class ToxicityDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'attention_mask': encoding['attention_mask'].flatten()
        }

def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting embeddings'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
    return np.vstack(embeddings)

def main():
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_KEY}")
    
    # Load data
    train_text = pd.read_csv(Config.MERGED_TEXT)
    train_label = pd.read_csv(Config.MERGED_LABEL)
    train_df = pd.merge(train_text, train_label, on='index')
    
    test_df = pd.read_csv(Config.TEST_TEXT)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize BERT
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_KEY)
    model = AutoModel.from_pretrained(Config.MODEL_KEY).to(Config.DEVICE)
    
    # Create loaders
    train_loader = DataLoader(ToxicityDataset(train_df['message'], tokenizer, Config.MAX_LENGTH), batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ToxicityDataset(test_df['message'], tokenizer, Config.MAX_LENGTH), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Extract
    print("\nStep 1: Extracting mmBERT embeddings...")
    X_train = extract_embeddings(model, train_loader, Config.DEVICE)
    y_train = train_df['label'].values
    
    X_test = extract_embeddings(model, test_loader, Config.DEVICE)
    
    # Train Random Forest
    print("\nStep 2: Training Random Forest (Ensemble)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train, y_train)
    
    # Predictions
    print("\nStep 3: Generating predictions...")
    train_preds = rf.predict(X_train)
    test_preds = rf.predict(X_test)
    
    # Metrics on train (since we merged val)
    print("\nMetrics on Training (Merged) Data:")
    print(classification_report(y_train, train_preds, target_names=[Config.LABEL_MAP[i] for i in range(Config.NUM_CLASSES)]))
    
    f1_macro = f1_score(y_train, train_preds, average='macro')
    print(f"Train Macro F1: {f1_macro:.4f}")
    
    # Save predictions
    submission = pd.DataFrame({
        'index': test_df['index'].values,
        'label': test_preds
    }).sort_values('index')
    
    submission.to_csv('predictions_hybrid.csv', index=False)
    print(f"✓ Predictions saved to predictions_hybrid.csv")
    
    # Log to training_log.csv
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'Timestamp': timestamp,
        'Model Name': 'Hybrid-RF-mmBERT',
        'Model Key': Config.MODEL_KEY,
        'epoch': 'N/A',
        'train_loss': 'N/A',
        'train_acc': rf.score(X_train, y_train),
        'val_loss': 'N/A',
        'val_f1_macro': 'N/A', # No dedicated val set in merged mode
        'val_f1_weighted': 'N/A',
        'Max Length': Config.MAX_LENGTH,
        'Batch Size': Config.BATCH_SIZE,
        'Learning Rate': 'N/A (RF)',
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
