"""
Quick test script to verify data loading and model setup
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("="*80)
print("GameTox Toxicity Detection - Setup Verification")
print("="*80)

# Check PyTorch and CUDA
print(f"\n1. PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")

# Check data files
print("\n2. Checking data files...")
data_files = [
    "data/train/train_index_text.csv",
    "data/train/train_index_label.csv",
    "data/val/val_index_text.csv",
    "data/val/val_index_label.csv",
    "data/test_index_text.csv"
]

all_exist = True
for file in data_files:
    try:
        df = pd.read_csv(file)
        print(f"   ✓ {file} ({len(df)} rows)")
    except Exception as e:
        print(f"   ✗ {file} - Error: {str(e)}")
        all_exist = False

if not all_exist:
    print("\n   ERROR: Some data files are missing or corrupted!")
    exit(1)

# Load and check training data
print("\n3. Loading training data...")
train_text = pd.read_csv("data/train/train_index_text.csv")
train_label = pd.read_csv("data/train/train_index_label.csv")
train_data = pd.merge(train_text, train_label, on='index')

print(f"   Total training samples: {len(train_data)}")
print(f"   Label distribution:")
for label, count in train_data['label'].value_counts().sort_index().items():
    label_names = {
        0: "Non-toxic",
        1: "Insults and Flaming",
        2: "Other Offensive Texts",
        3: "Hate and Harassment",
        4: "Threats",
        5: "Extremism"
    }
    print(f"      {int(label)}: {label_names.get(int(label), 'Unknown')} - {count} samples ({count/len(train_data)*100:.1f}%)")

# Test model loading
print("\n4. Testing model loading...")
try:
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=6
    )
    print("   ✓ Model loading successful (tested with DistilBERT)")
    
    # Test tokenization
    sample_text = train_data['message'].iloc[0]
    encoding = tokenizer(
        sample_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print(f"   ✓ Tokenization successful")
    print(f"      Sample text: '{sample_text}'")
    print(f"      Token IDs shape: {encoding['input_ids'].shape}")
    
except Exception as e:
    print(f"   ✗ Model loading failed: {str(e)}")
    exit(1)

print("\n" + "="*80)
print("Setup Verification Complete!")
print("="*80)
print("\nYou're ready to start training. Run:")
print("  python train.py")
print("\nOr use the quick start script:")
print("  bash quickstart.sh")
print("="*80)
