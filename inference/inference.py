"""
Inference script for Toxicity Detection
Uses a trained model to generate predictions on test data
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import yaml
import json
import argparse
import zipfile

class InferenceConfig:
    def __init__(self, config_path="../config/config.yaml"):
        # Adjust path if called from inside inference folder
        if not os.path.exists(config_path):
             config_path = "config/config.yaml"
             
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.TEST_TEXT = self.cfg['data']['test_text']
        self.MAX_LENGTH = self.cfg['training']['max_length']
        self.BATCH_SIZE = self.cfg['training']['batch_size']
        self.NUM_CLASSES = self.cfg['training']['num_classes']
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GPUS = torch.cuda.device_count()

class ToxicityTestDataset(Dataset):
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

def run_inference(model_key, weight_path, test_data_path, output_file):
    cfg = InferenceConfig()
    print(f"Loading data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    
    # Preprocessing
    def preprocess_text(text):
        if not isinstance(text, str): return ""
        text = text.replace("#ERROR!", "").replace("#NAME?", "")
        return " ".join(text.split())
    
    test_df['message'] = test_df['message'].apply(preprocess_text)
    
    print(f"Loading model {model_key} and weights from {weight_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_key, num_labels=cfg.NUM_CLASSES
    )
    
    # Load state dict
    state_dict = torch.load(weight_path, map_location=cfg.DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(cfg.DEVICE)
    
    if cfg.GPUS > 1:
        print(f"Using {cfg.GPUS} GPUs for inference")
        model = nn.DataParallel(model)
    
    model.eval()
    
    test_dataset = ToxicityTestDataset(test_df['message'].values, tokenizer, cfg.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(cfg.DEVICE)
            attention_mask = batch['attention_mask'].to(cfg.DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            
    submission = pd.DataFrame({
        'index': test_df['index'].values,
        'label': predictions
    })
    submission = submission.sort_values('index')
    submission.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to {output_file}")
    
    # Create ZIP
    zip_file = output_file.replace('.csv', '.zip')
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, os.path.basename(output_file))
    print(f"✓ Zipped predictions to {zip_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--weights", type=str, default="best_model_roberta.pt")
    parser.add_argument("--test_data", type=str, default="../data/test_index_text.csv")
    parser.add_argument("--output", type=str, default="predictions.csv")
    args = parser.parse_args()
    
    run_inference(args.model, args.weights, args.test_data, args.output)
