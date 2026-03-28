import os
import pandas as pd
import numpy as np
import torch
import yaml
from torch import nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

class ConfigLoader:
    def __init__(self, config_path="../config/config.yaml"):
        if not os.path.exists(config_path):
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
            
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        # Data Paths
        self.TRAIN_TEXT = self.cfg['data']['train_text']
        self.TRAIN_LABEL = self.cfg['data']['train_label']
        self.VAL_TEXT = self.cfg['data']['val_text']
        self.VAL_LABEL = self.cfg['data']['val_label']
        self.TEST_TEXT = self.cfg['data']['test_text']
        
        self.TEST_LABEL = self.cfg['data'].get('test_label')
        if not self.TEST_LABEL:
            if "test_release" in self.TEST_TEXT:
                self.TEST_LABEL = self.TEST_TEXT.replace("index_text", "index_label")
            else:
                self.TEST_LABEL = "/home/sure/projects/eeuca/data/test_release-20260327T051813Z-3-001/test_release/test_index_label.csv"

        # Models Dictionary
        self.MODELS = self.cfg.get('models', {})
        
        # Hyperparameters
        self.MAX_LENGTH = int(self.cfg['training']['max_length'])
        self.BATCH_SIZE = int(self.cfg['training']['batch_size'])
        self.EPOCHS = int(self.cfg['training']['epochs'])
        self.LEARNING_RATE = float(self.cfg['training']['learning_rate'])
        self.WEIGHT_DECAY = float(self.cfg['training'].get('weight_decay', 0.01))
        self.EARLY_STOPPING_PATIENCE = int(self.cfg['training']['early_stopping_patience'])
        self.NUM_CLASSES = int(self.cfg['training']['num_classes'])
        
        self.OUTPUT_DIR = "./results"
        self.RESULTS_CSV = "multilingual_results.csv"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Cast class_weights to match the device and dtype of logits (especially for fp16/Half)
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(device=labels.device, dtype=logits.dtype))
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_and_merge(text_path, label_path):
    root_dir = "/home/sure/projects/eeuca"
    if not os.path.isabs(text_path):
        text_path = os.path.join(root_dir, text_path)
    if label_path and not os.path.isabs(label_path):
        label_path = os.path.join(root_dir, label_path)
        
    df_text = pd.read_csv(text_path)
    if label_path and os.path.exists(label_path):
        df_label = pd.read_csv(label_path)
        return pd.merge(df_text, df_label, on='index')
    return df_text

def train_single_model(model_key, model_name, config, train_df, val_df, test_df, class_weights_tensor):
    print(f"\n" + "="*50)
    print(f"STARTING TRAINING: {model_key} ({model_name})")
    print("="*50)
    
    model_output_dir = os.path.join(config.OUTPUT_DIR, model_key)
    os.makedirs(model_output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=config.MAX_LENGTH)
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=os.path.join(model_output_dir, "logs"),
        logging_steps=50,
        fp16=torch.cuda.is_available() and "deberta" not in model_name.lower(),
        report_to="none"
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)]
    )
    
    trainer.train()
    
    test_results = {}
    if 'label' in test_df.columns:
        print(f"Evaluating {model_key} on Test Set...")
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        test_results = trainer.evaluate(test_dataset)
        print(f"Test F1-Macro for {model_key}: {test_results['eval_f1_macro']:.4f}")
    
    eval_results = trainer.evaluate()
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_key': model_key,
        'model_name': model_name,
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'max_length': config.MAX_LENGTH,
        'val_f1_macro': eval_results.get('eval_f1_macro', 'N/A'),
        'val_precision_macro': eval_results.get('eval_precision_macro', 'N/A'),
        'val_recall_macro': eval_results.get('eval_recall_macro', 'N/A'),
        'test_f1_macro': test_results.get('eval_f1_macro', 'N/A'),
        'test_precision_macro': test_results.get('eval_precision_macro', 'N/A'),
        'test_recall_macro': test_results.get('eval_recall_macro', 'N/A'),
        'test_accuracy': test_results.get('eval_accuracy', 'N/A'),
    }
    
    results_df = pd.DataFrame([results])
    if os.path.exists(config.RESULTS_CSV):
        results_df.to_csv(config.RESULTS_CSV, mode='a', header=False, index=False)
    else:
        results_df.to_csv(config.RESULTS_CSV, index=False)
    
    final_model_path = os.path.join(model_output_dir, "final_best")
    trainer.save_model(final_model_path)
    print(f"Best model for {model_key} saved to {final_model_path}")
    
    # Cleanup memory
    del model
    del trainer
    torch.cuda.empty_cache()

def main():
    config = ConfigLoader()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading and merging data...")
    train_df = load_and_merge(config.TRAIN_TEXT, config.TRAIN_LABEL)
    val_df = load_and_merge(config.VAL_TEXT, config.VAL_LABEL)
    test_df = load_and_merge(config.TEST_TEXT, config.TEST_LABEL)
    
    for df in [train_df, val_df, test_df]:
        if 'message' in df.columns:
            df.rename(columns={'message': 'text'}, inplace=True)
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)
    
    labels = train_df['label'].values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    if not config.MODELS:
        print("No models found in config.yaml under 'models' key.")
        return
        
    for model_key, model_name in config.MODELS.items():
        try:
            train_single_model(model_key, model_name, config, train_df, val_df, test_df, class_weights_tensor)
        except Exception as e:
            print(f"ERROR training model {model_key}: {e}")
            continue

if __name__ == "__main__":
    main()
