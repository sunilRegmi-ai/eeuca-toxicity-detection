"""
train_smote_mmbert.py

This script extracts [CLS] embeddings from mmBERT and uses SMOTE
to synthetically balance minority classes in the embedding space.
It uses Stratified K-Fold cross-validation, applying SMOTE *only* on the training folds
to rigorously avoid data leakage, and trains a classifier head (XGBoost) on the balanced data.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from tqdm import tqdm
import yaml
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
            
        # Use merged data as base (without conda augmentation) to test pure SMOTE effects
        self.MERGED_TEXT = "data/merged/merged_text.csv"
        self.MERGED_LABEL = "data/merged/merged_label.csv"
        self.TEST_TEXT = self.cfg['data']['test_text']
        self.MAX_LENGTH = self.cfg['training']['max_length']
        self.BATCH_SIZE = self.cfg['training']['batch_size']
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_KEY = self.cfg['models'].get('mmbert', 'jhu-clsp/mmBERT-base')
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
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting mmBERT embeddings'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token
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
    
    # 2. Setup Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_KEY)
    model = AutoModel.from_pretrained(Config.MODEL_KEY).to(Config.DEVICE)
    
    train_loader = DataLoader(ToxicityDataset(train_df['message'], tokenizer, Config.MAX_LENGTH), batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ToxicityDataset(test_df['message'], tokenizer, Config.MAX_LENGTH), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print("\n[Step 1/4] Extracting embeddings from original data...")
    X_train_all = extract_embeddings(model, train_loader, Config.DEVICE)
    y_train_all = train_df['label'].values.astype(int)
    X_test = extract_embeddings(model, test_loader, Config.DEVICE)
    
    # Free up VRAM to prevent memory errors during XGBoost device='cuda'
    del model
    torch.cuda.empty_cache()
    
    # 3. K-Fold Cross-Validation with SMOTE inner loop
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    
    cv_scores = []
    fold_predictions = []
    
    print(f"\n[Step 2/4] Starting {K}-Fold CV with SMOTE balancing...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all)):
        print(f"\n--- Fold {fold+1}/{K} ---")
        X_tr_raw, X_val = X_train_all[train_idx], X_train_all[val_idx]
        y_tr_raw, y_val = y_train_all[train_idx], y_train_all[val_idx]
        
        # --- CRITICAL: Apply SMOTE ONLY on the training fold to prevent data leakage ---
        print(f"  > Original fold classes: {np.bincount(y_tr_raw)}")
        
        target_samples = 2000
        # Stratify logic based on the 2000 target
        under_strategy = {c: target_samples for c in np.unique(y_tr_raw) if np.sum(y_tr_raw == c) > target_samples}
        over_strategy = {c: target_samples for c in np.unique(y_tr_raw) if np.sum(y_tr_raw == c) < target_samples}
        
        # We need special handling: Some minority classes might have fewer than 6 samples, 
        # which breaks SMOTE's default k_neighbors=5. We'll dynamically adjust k_neighbors.
        min_samples = np.min(np.bincount(y_tr_raw))
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        
        pipeline = Pipeline([
            ('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=SEED)),
            ('smote', SMOTE(sampling_strategy=over_strategy, random_state=SEED, k_neighbors=k_neighbors))
        ])
        
        X_tr_smote, y_tr_smote = pipeline.fit_resample(X_tr_raw, y_tr_raw)
        print(f"  > Balanced classes (Target=2000): {np.bincount(y_tr_smote)}")
        
        # Classifier Head Options
        clf = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
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
        
        # Train on SMOTE data, validate on untouched fold data
        clf.fit(
            X_tr_smote, y_tr_smote,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        # Evaluate
        val_preds = clf.predict(X_val)
        f1_m = f1_score(y_val, val_preds, average='macro')
        cv_scores.append(f1_m)
        print(f"  > Fold {fold+1} Validation Macro F1: {f1_m:.4f}")
        
        # Predict on Test for Bagging
        test_probs = clf.predict_proba(X_test)
        fold_predictions.append(test_probs)
    
    avg_cv_score = np.mean(cv_scores)
    print(f"\n========================================")
    print(f"Average CV Macro F1 (SMOTE): {avg_cv_score:.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"========================================\n")
    
    # 4. Bagging: Average probabilities across folds
    print("[Step 3/4] Ensembling predictions across folds (Bagging)...")
    avg_probs = np.mean(fold_predictions, axis=0)
    final_test_preds = np.argmax(avg_probs, axis=1)
    
    # 5. Save Results
    print(f"[Step 4/4] Saving results...")
    submission = pd.DataFrame({
        'index': test_df['index'].values,
        'label': final_test_preds
    }).sort_values('index')
    
    output_file = 'predictions_smote_mmbert.csv'
    submission.to_csv(output_file, index=False)
    print(f"✓ SMOTE predictions saved to {output_file}")
    
    # 6. Logging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'Timestamp': timestamp,
        'Model Name': 'mmBERT-SMOTE-XGBoost',
        'Model Key': Config.MODEL_KEY,
        'epoch': f'{K} Folds',
        'train_loss': 'N/A',
        'train_acc': 'N/A',
        'val_loss': 'N/A',
        'val_f1_macro': avg_cv_score,
        'val_f1_weighted': 'N/A',
        'Max Length': Config.MAX_LENGTH,
        'Batch Size': Config.BATCH_SIZE,
        'Learning Rate': 'N/A (XGB)',
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
