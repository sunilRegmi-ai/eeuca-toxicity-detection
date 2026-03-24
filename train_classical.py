"""
Classical ML Script: TF-IDF + Random Forest / XGBoost
A strong baseline for imbalanced text classification.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import yaml
from datetime import datetime

# Set seed
SEED = 42
np.random.seed(SEED)

class ConfigLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.MERGED_TEXT = "data/merged/merged_text.csv"
        self.MERGED_LABEL = "data/merged/merged_label.csv"
        self.TEST_TEXT = self.cfg['data']['test_text']
        self.NUM_CLASSES = self.cfg['training']['num_classes']
        self.LABEL_MAP = {int(k): v for k, v in self.cfg['labels'].items()}

Config = ConfigLoader()

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.replace("#ERROR!", "").replace("#NAME?", "")
    return " ".join(text.lower().split()) # Basic lowercase and space normalization

def main():
    print("Loading data...")
    train_text = pd.read_csv(Config.MERGED_TEXT)
    train_label = pd.read_csv(Config.MERGED_LABEL)
    train_df = pd.merge(train_text, train_label, on='index')
    
    test_df = pd.read_csv(Config.TEST_TEXT)
    
    print(f"Total train samples (Merged): {len(train_df)}")
    
    # Feature Extraction
    print("\nExtracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    
    X_train_full = tfidf.fit_transform(train_df['message'].astype(str))
    y_train_full = train_df['label'].values
    
    X_test = tfidf.transform(test_df['message'].astype(str))
    
    # Optional: Split for local validation even if merged
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full)
    
    # Model 1: Random Forest
    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        n_jobs=-1,
        random_state=SEED,
        verbose=1
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    val_preds = rf.predict(X_val)
    print("\nRandom Forest - Local Validation Report (10% subset):")
    print(classification_report(y_val, val_preds, target_names=[Config.LABEL_MAP[i] for i in range(Config.NUM_CLASSES)]))
    
    macro_f1 = f1_score(y_val, val_preds, average='macro')
    print(f"Local Validation Macro F1: {macro_f1:.4f}")
    
    # Generate predictions on full test set
    print("\nGenerating final predictions on full test set...")
    # Refit on full data for best performance
    rf.fit(X_train_full, y_train_full)
    test_preds = rf.predict(X_test)
    
    submission = pd.DataFrame({
        'index': test_df['index'].values,
        'label': test_preds
    }).sort_values('index')
    
    output_file = 'predictions_classical_rf.csv'
    submission.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to {output_file}")
    
    # Logging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'Timestamp': timestamp,
        'Model Name': 'Classical-TFIDF-RF',
        'Model Key': 'TF-IDF-10k',
        'epoch': 'N/A',
        'train_loss': 'N/A',
        'train_acc': accuracy_score(y_val, val_preds),
        'val_loss': 'N/A',
        'val_f1_macro': macro_f1,
        'val_f1_weighted': f1_score(y_val, val_preds, average='weighted'),
        'Max Length': 'N/A',
        'Batch Size': 'N/A',
        'Learning Rate': 'N/A',
        'Device': 'CPU'
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
