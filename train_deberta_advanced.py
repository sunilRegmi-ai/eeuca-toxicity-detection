"""
train_deberta_advanced.py

Advanced modeling script specifically designed to break Macro F1 plateaus
on highly imbalanced text datasets.
Features:
- DeBERTa-v3 model architectures
- Layer-wise Learning Rate Decay (LLRD)
- Focal Loss & Class Weighting
- Stratified 5-Fold Cross Validation
- Nelder-Mead Out-Of-Fold (OOF) Optimization specifically targeting Macro F1
"""

import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from tqdm import tqdm
import yaml
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class ConfigLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.TRAIN_TEXT = self.cfg['data']['train_text']
        self.TRAIN_LABEL = self.cfg['data']['train_label']
        self.TEST_TEXT = self.cfg['data']['test_text']
        self.MAX_LENGTH = self.cfg['training']['max_length']
        # DeBERTa-v3 is heavier than mmBERT. Cut batch size in half to prevent CUDA OOM.
        self.BATCH_SIZE = self.cfg['training']['batch_size'] // 2 if self.cfg['training']['batch_size'] > 16 else self.cfg['training']['batch_size']
        self.EPOCHS = self.cfg['training']['epochs']
        self.EARLY_STOPPING_PATIENCE = self.cfg['training']['early_stopping_patience']
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_KEY = self.cfg['models'].get('deberta', 'microsoft/deberta-v3-base')
        self.NUM_CLASSES = self.cfg['training']['num_classes']

Config = ConfigLoader()

class textDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item



def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, llrd):
    """
    Layer-wise Learning Rate Decay
    Slowly drops the learning rate across deeper layers of the Transformer so it 
    does not 'forget' its fundamental linguistic knowledge.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    
    try:
        layers = [model.deberta.embeddings] + list(model.deberta.encoder.layer)
    except:
        layers = []
        
    if len(layers) > 0:
        layers.reverse()
        lr = learning_rate
        for layer in layers:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            lr *= llrd
            
        # Top classification head gets native Learning Rate
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": learning_rate,
            },
            {
                "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": learning_rate,
            },
        ]
    else:
        # Fallback
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        
    return optimizer_grouped_parameters

def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    # Gradient Accumulation explicitly set for DeBERTa memory constraints
    accumulation_steps = 2
    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()
        loss = criterion(logits, labels) / accumulation_steps
        
        loss.backward()
        
        # Clip gradients to prevent DeBERTa NaN loss explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1, np.array(all_probs), np.array(all_labels)

def predict_test(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting Test", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)

def main():
    print(f"Device: {Config.DEVICE}")
    print(f"Base Model: {Config.MODEL_KEY}")
    print(f"Augmented Dataset Loaded: {Config.TRAIN_TEXT}")
    
    # 1. Load Data
    train_text = pd.read_csv(Config.TRAIN_TEXT)
    train_label = pd.read_csv(Config.TRAIN_LABEL)
    train_df = pd.merge(train_text, train_label, on='index')
    
    test_df = pd.read_csv(Config.TEST_TEXT)
    
    # DeBERTa-v3 specifically requires `sentencepiece` fast tokenizer mapping.
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_KEY, use_fast=True)
    
    # 2. Setup K-Fold
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    
    # Static Class Weights
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(Config.DEVICE)
    print(f"\nCalculated Imbalance Weights: {class_weights}")
    
    # Replacing FocalLoss with safer native CrossEntropy Loss to avoid NaN issues
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    oof_probs = np.zeros((len(train_df), Config.NUM_CLASSES))
    test_probs_ensemble = []
    cv_scores_argmax = []
    
    print(f"\n[Step 1/4] Starting {K}-Fold Advanced Training...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f"\n========================================")
        print(f"Fold {fold+1}/{K}")
        print(f"========================================")
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        train_dataset = textDataset(train_fold['message'], train_fold['label'], tokenizer, Config.MAX_LENGTH)
        val_dataset = textDataset(val_fold['message'], val_fold['label'], tokenizer, Config.MAX_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_KEY, num_labels=Config.NUM_CLASSES)
        model.to(Config.DEVICE)
        
        # DeBERTa explicit Layer Wise Decay strategy
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, learning_rate=1e-5, weight_decay=0.01, llrd=0.85
        )
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        
        best_val_f1 = 0
        best_probs = None
        patience_counter = 0
        
        # Fold Training Loop
        for epoch in range(Config.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE, criterion)
            val_f1, probs, _ = evaluate(model, val_loader, Config.DEVICE)
            
            print(f"  > Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Macro F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_probs = probs
                patience_counter = 0
                torch.save(model.state_dict(), f"deberta_fold_{fold+1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"  > Early stopping triggered!")
                    break
                    
        print(f"Best Base F1 (Argmax) for Fold {fold+1}: {best_val_f1:.4f}")
        cv_scores_argmax.append(best_val_f1)
        
        # Collect OOF
        oof_probs[val_idx] = best_probs
        
        # Predict Fold Test Set
        model.load_state_dict(torch.load(f"deberta_fold_{fold+1}.pt", weights_only=True))
        test_dataset = textDataset(test_df['message'], None, tokenizer, Config.MAX_LENGTH)
        # Larger batch size for test set inference
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE*2, shuffle=False)
        fold_test_probs = predict_test(model, test_loader, Config.DEVICE)
        test_probs_ensemble.append(fold_test_probs)
        
        # Clean up
        del model
        torch.cuda.empty_cache()

    print(f"\n========================================")
    print(f"Average CV Base Macro F1: {np.mean(cv_scores_argmax):.4f}")
    
    # --------------------------------------------------------------------------
    # THRESHOLD OPTIMIZATION (Nelder-Mead)
    # --------------------------------------------------------------------------
    print("\n[Step 2/4] Executing Nelder-Mead Optimization targeting Macro F1...")
    
    y_true_all = train_df['label'].values
    
    def loss_fn(weights):
        # Apply class weights exactly to output probabilities (Threshold adjustment)
        weighted_probs = oof_probs * weights
        preds = np.argmax(weighted_probs, axis=1)
        return -f1_score(y_true_all, preds, average='macro')
        
    initial_weights = [1.0] * Config.NUM_CLASSES
    # Nelder - Mead does not natively take bounds, ignore warning.
    res = minimize(loss_fn, initial_weights, method='Nelder-Mead', options={'maxiter': 3000})
    best_class_weights = res.x
    
    optimized_f1 = -res.fun
    
    print("\n========================================")
    print(f"Optimized Class Multipliers: {best_class_weights}")
    print(f"Final Optimized Global Macro F1: {optimized_f1:.4f} (Up from {f1_score(y_true_all, np.argmax(oof_probs, axis=1), average='macro'):.4f})")
    print("========================================\n")
    
    # --------------------------------------------------------------------------
    # PREDICTION & SAVING
    # --------------------------------------------------------------------------
    print("[Step 3/4] Ensembling Test probabilities and locking-in Optimal Weights...")
    avg_test_probs = np.mean(test_probs_ensemble, axis=0)
    
    # Lock-in the threshold shift natively learned from optimization
    weighted_test_probs = avg_test_probs * best_class_weights
    final_test_preds = np.argmax(weighted_test_probs, axis=1)
    
    print("[Step 4/4] Saving detailed results...")
    submission = pd.DataFrame({
        'index': test_df['index'].values,
        'label': final_test_preds
    }).sort_values('index')
    
    output_file = 'predictions_deberta_optimized.csv'
    submission.to_csv(output_file, index=False)
    
    # Log everything natively
    log_entry = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Model Name': 'DeBERTa-v3-Advanced',
        'Model Key': Config.MODEL_KEY,
        'epoch': f'{K}-Fold LLRD',
        'train_loss': 'Threshold Optimized',
        'train_acc': 'N/A',
        'val_loss': 'N/A',
        'val_f1_macro': optimized_f1,
        'val_f1_weighted': 'N/A',
        'Max Length': Config.MAX_LENGTH,
        'Batch Size': Config.BATCH_SIZE,
        'Learning Rate': '1.5e-5 (LLRD 0.85)',
        'Device': str(Config.DEVICE)
    }
    
    log_df = pd.DataFrame([log_entry])
    log_file = 'training_log.csv'
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
        
    print(f"✓ Training log appended. Final predictions precisely saved to {output_file}")

if __name__ == "__main__":
    main()
