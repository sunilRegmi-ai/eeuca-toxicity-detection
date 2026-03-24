"""
Robust Hybrid Training Script: mmBERT Embeddings + XGBoost with K-Fold CV
Prevents overfitting and uses bagging for better generalization.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm
import yaml
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

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
            # [CLS] token
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

def main():
    print(f"Device: {Config.DEVICE}")
    print(f"Base Model: {Config.MODEL_KEY}")
    
    # 1. Load Data
    train_text = pd.read_csv(Config.MERGED_TEXT)
    train_label = pd.read_csv(Config.MERGED_LABEL)
    train_df = pd.merge(train_text, train_label, on='index')
    test_df = pd.read_csv(Config.TEST_TEXT)
    
    # 2. Extract Embeddings (Once)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_KEY)
    model = AutoModel.from_pretrained(Config.MODEL_KEY).to(Config.DEVICE)
    
    train_loader = DataLoader(ToxicityDataset(train_df['message'], tokenizer, Config.MAX_LENGTH), batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ToxicityDataset(test_df['message'], tokenizer, Config.MAX_LENGTH), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print("\nExtracting mmBERT embeddings...")
    X_train_all = extract_embeddings(model, train_loader, Config.DEVICE)
    y_train_all = train_df['label'].values.astype(int)
    X_test = extract_embeddings(model, test_loader, Config.DEVICE)
    
    # 3. K-Fold Cross-Validation
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    
    cv_scores = []
    fold_predictions = []
    
    print(f"\nStarting {K}-Fold Cross-Validation with XGBoost...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all)):
        print(f"\n--- Fold {fold+1}/{K} ---")
        X_tr, X_val = X_train_all[train_idx], X_train_all[val_idx]
        y_tr, y_val = y_train_all[train_idx], y_train_all[val_idx]
        
        # Calculate samples weights for training set only
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
        
        # XGBoost Classifier
        clf = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=Config.NUM_CLASSES,
            random_state=SEED,
            n_jobs=-1,
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            early_stopping_rounds=50
        )
        
        # Train with early stopping
        clf.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        # Evaluate
        val_preds = clf.predict(X_val)
        f1_m = f1_score(y_val, val_preds, average='macro')
        cv_scores.append(f1_m)
        print(f"Fold {fold+1} Macro F1: {f1_m:.4f}")
        
        # Predict on Test for Bagging
        test_probs = clf.predict_proba(X_test)
        fold_predictions.append(test_probs)
    
    avg_cv_score = np.mean(cv_scores)
    print(f"\n========================================")
    print(f"Average CV Macro F1: {avg_cv_score:.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"========================================\n")
    
    # 4. Bagging: Average probabilities across folds
    print("Ensembling predictions across folds (Bagging)...")
    avg_probs = np.mean(fold_predictions, axis=0)
    final_test_preds = np.argmax(avg_probs, axis=1)
    
    # 5. Save Results
    submission = pd.DataFrame({
        'index': test_df['index'].values,
        'label': final_test_preds
    }).sort_values('index')
    
    output_file = 'predictions_hybrid_robust.csv'
    submission.to_csv(output_file, index=False)
    print(f"✓ Robust predictions saved to {output_file}")
    
    # 6. Logging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'Timestamp': timestamp,
        'Model Name': 'Hybrid-XGBoost-KFold',
        'Model Key': Config.MODEL_KEY,
        'epoch': f'{K} Folds',
        'train_loss': 'N/A',
        'train_acc': 'N/A',
        'val_loss': 'N/A',
        'val_f1_macro': avg_cv_score,
        'val_f1_weighted': 'N/A',
        'Max Length': Config.MAX_LENGTH,
        'Batch Size': Config.BATCH_SIZE,
        'Learning Rate': '0.05 (XGB)',
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
