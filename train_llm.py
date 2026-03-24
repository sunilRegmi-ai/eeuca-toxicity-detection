"""
Toxicity Detection Training Script for Multilingual LLMs (Qwen2.5, Llama, etc.)
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, BitsAndBytesConfig
)
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
except ImportError:
    print("WARNING: 'peft' library not found. LoRA training will not be available.")
    print("Please install it using: pip install peft bitsandbytes accelerate")

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
        self.MODELS = self.cfg.get('llm_models', {
            'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct'
        })
        
        # Training parameters
        self.MAX_LENGTH = self.cfg['training']['max_length']
        self.BATCH_SIZE = self.cfg['training']['batch_size']
        self.EPOCHS = self.cfg['training']['epochs']
        self.LEARNING_RATE = self.cfg['training']['learning_rate']
        self.WEIGHT_DECAY = self.cfg['training']['weight_decay']
        self.EARLY_STOPPING_PATIENCE = self.cfg['training']['early_stopping_patience']
        self.NUM_CLASSES = self.cfg['training']['num_classes']
        
        # LoRA parameters
        self.LORA_R = self.cfg.get('lora', {}).get('r', 8)
        self.LORA_ALPHA = self.cfg.get('lora', {}).get('alpha', 16)
        self.LORA_DROPOUT = self.cfg.get('lora', {}).get('dropout', 0.05)
        
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

# Reuse ToxicityDataset from train.py logic
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
    print("Loading data...")
    if Config.USE_AUGMENTED:
        train_text_path, train_label_path = Config.AUGMENTED_TRAIN_TEXT, Config.AUGMENTED_TRAIN_LABEL
    else:
        train_text_path, train_label_path = Config.TRAIN_TEXT, Config.TRAIN_LABEL

    train_data = pd.merge(pd.read_csv(train_text_path), pd.read_csv(train_label_path), on='index')
    val_data = pd.merge(pd.read_csv(Config.VAL_TEXT), pd.read_csv(Config.VAL_LABEL), on='index')
    test_data = pd.read_csv(Config.TEST_TEXT)
    
    for df in [train_data, val_data, test_data]:
        df['message'] = df['message'].apply(preprocess_text)
    
    train_data['label'] = train_data['label'].astype(int)
    val_data['label'] = val_data['label'].astype(int)
    
    if Config.RESAMPLING_ENABLED:
        balanced_dfs = []
        for label in range(Config.NUM_CLASSES):
            class_df = train_data[train_data['label'] == label]
            if len(class_df) == 0: continue
            if label == 0:
                n_samples = min(len(class_df), Config.UNDERSAMPLE_TARGET)
                resampled_df = class_df.sample(n=n_samples, random_state=SEED)
            else:
                if len(class_df) < Config.OVERSAMPLE_MINIMUM:
                    resampled_df = class_df.sample(n=Config.OVERSAMPLE_MINIMUM, replace=True, random_state=SEED)
                else:
                    resampled_df = class_df
            balanced_dfs.append(resampled_df)
        train_data = pd.concat(balanced_dfs, ignore_index=True)
    
    class_counts = train_data['label'].value_counts().sort_index().values
    class_weights = torch.tensor(len(train_data) / (len(class_counts) * class_counts), dtype=torch.float).to(Config.DEVICE)
    
    return train_data, val_data, test_data, class_weights

def create_data_loader(texts, labels, tokenizer, batch_size, shuffle=True):
    dataset = ToxicityDataset(texts.values, labels.values if labels is not None else None, tokenizer, Config.MAX_LENGTH)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, data_loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    losses = []
    correct = 0
    total = 0
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
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

def eval_model(model, data_loader, device, class_weights=None):
    model.eval()
    losses = []
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(logits.dtype) if class_weights is not None else None)
            losses.append(loss_fn(logits, labels).item())
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    return np.mean(losses), f1_macro, f1_weighted, predictions, true_labels

def train_model(model_name, model_key, train_data, val_data, class_weights):
    print(f"\nTraining LLM: {model_name} ({model_key})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_key,
        num_labels=Config.NUM_CLASSES,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare for k-bit training and LoRA
    if 'peft' in globals():
        print("Preparing model for k-bit training and applying LoRA...")
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # DataParallel handles only non-quantized models usually, 
    # but with device_map="auto", it handles multi-GPU internally.
    
    train_loader = create_data_loader(train_data['message'], train_data['label'], tokenizer, Config.BATCH_SIZE)
    val_loader = create_data_loader(val_data['message'], val_data['label'], tokenizer, Config.BATCH_SIZE, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * Config.EPOCHS)
    
    best_f1 = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_f1_macro': [], 'val_f1_weighted': []}
    
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch + 1}/{Config.EPOCHS}')
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE, class_weights)
        val_loss, val_f1_macro, val_f1_weighted, val_preds, val_labels = eval_model(model, val_loader, Config.DEVICE, class_weights)
        
        print(f'Train Loss: {train_loss:.4f} | Val F1 (Macro): {val_f1_macro:.4f}')
        
        for k, v in zip(history.keys(), [train_loss, train_acc, val_loss, val_f1_macro, val_f1_weighted]):
            history[k].append(v)
            
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(f'best_llm_{model_name}')
            print(f'✓ Saved best LLM with F1 (Macro): {best_f1:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                print(f'! Early stopping at epoch {epoch + 1}')
                break
                
    return {
        'model_name': model_name,
        'model_key': model_key,
        'history': history,
        'best_f1_macro': best_f1,
        'epoch_logs': [
            {'epoch': i+1, 'train_loss': history['train_loss'][i], 'train_acc': history['train_acc'][i], 
             'val_loss': history['val_loss'][i], 'val_f1_macro': history['val_f1_macro'][i], 'val_f1_weighted': history['val_f1_weighted'][i]}
            for i in range(len(history['train_loss']))
        ]
    }

def main():
    print(f"Device: {Config.DEVICE} ({Config.GPUS} GPUs)")
    train_data, val_data, test_data, class_weights = load_data()
    
    results = {}
    for model_name, model_key in Config.MODELS.items():
        try:
            results[model_name] = train_model(model_name, model_key, train_data, val_data, class_weights)
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    if results:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        detailed_logs = []
        for model_name, result in results.items():
            for epoch_log in result['epoch_logs']:
                detailed_logs.append({
                    'Timestamp': timestamp, 'Model Name': model_name, 'Model Key': result['model_key'],
                    **epoch_log, 'Max Length': Config.MAX_LENGTH, 'Batch Size': Config.BATCH_SIZE,
                    'Learning Rate': Config.LEARNING_RATE, 'LORA_R': Config.LORA_R, 'Device': str(Config.DEVICE)
                })
        
        log_df = pd.DataFrame(detailed_logs)
        log_file = 'training_log.csv'
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
        print(f"✓ Results appended to {log_file}")

if __name__ == "__main__":
    main()
