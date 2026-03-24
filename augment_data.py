import pandas as pd
import requests
import io
import os
import numpy as np

def augment_dataset(url, output_dir):
    print(f"Downloading CONDA dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return
    
    conda_df = pd.read_csv(io.StringIO(response.text))
    
    # Mapping CONDA labels to GameTox labels
    # CONDA: O: Ordinary, I: Insult, A: Abusive, E: Expletive
    # GameTox: 0: Non-toxic, 1: Insults, 2: Other Offensive, 3: Hate/Harassment, 4: Threats, 5: Extremism
    mapping = {
        'O': 0, # Ordinary -> Non-toxic
        'I': 1, # Insult -> Insulting
        'E': 2, # Expletive -> Other Offensive
        'A': 3, # Abusive -> Hate/Harassment
    }
    
    conda_filtered = conda_df[conda_df['intentClass'].isin(mapping.keys())].copy()
    conda_filtered['label'] = conda_filtered['intentClass'].map(mapping)
    conda_filtered = conda_filtered.rename(columns={'utterance': 'message'})
    
    # Load original training data
    orig_text = pd.read_csv("data/train/train_index_text.csv")
    orig_label = pd.read_csv("data/train/train_index_label.csv")
    orig_train = pd.merge(orig_text, orig_label, on='index')
    
    print(f"\nOriginal training distribution:")
    print(orig_train['label'].value_counts().sort_index())

    # --- BALANCING LOGIC ---
    print("\nApplying smart balancing...")
    
    # 1. Undersample Class 0 from the combined pool
    # We want to keep the total Class 0 count reasonable (e.g. 15,000)
    target_class0 = 15000
    
    # Original Class 0 samples
    orig_0 = orig_train[orig_train['label'] == 0]
    # CONDA Class 0 samples
    conda_0 = conda_filtered[conda_filtered['label'] == 0]
    
    # Mix them and sample
    all_0 = pd.concat([orig_0, conda_0])
    balanced_0 = all_0.sample(n=min(len(all_0), target_class0), random_state=42)
    
    # 2. Augment Classes 1, 2, 3 with ALL available CONDA samples
    # These classes are minority in GameTox but plentiful in CONDA
    balanced_123 = pd.concat([
        orig_train[orig_train['label'].isin([1, 2, 3])],
        conda_filtered[conda_filtered['label'].isin([1, 2, 3])]
    ])
    
    # 3. Oversample Classes 4 and 5 (Not present in CONDA)
    # Target at least 2000 samples for these rare classes to give model a chance
    target_minority = 2000
    
    orig_4 = orig_train[orig_train['label'] == 4]
    if len(orig_4) > 0:
        balanced_4 = orig_4.sample(n=target_minority, replace=True, random_state=42)
    else:
        balanced_4 = orig_4 # Empty
        
    orig_5 = orig_train[orig_train['label'] == 5]
    if len(orig_5) > 0:
        balanced_5 = orig_5.sample(n=target_minority, replace=True, random_state=42)
    else:
        balanced_5 = orig_5 # Empty

    # Combine everything
    augmented_train = pd.concat([balanced_0, balanced_123, balanced_4, balanced_5], ignore_index=True)
    
    # Re-assign indices to avoid collisions
    augmented_train['index'] = range(len(augmented_train))
    
    print(f"\nFinal Balanced Training Distribution:")
    dist = augmented_train['label'].value_counts().sort_index()
    print(dist)
    
    # Map for printing
    label_names = {0: "Non-toxic", 1: "Insults", 2: "Other Offensive", 
                  3: "Hate/Harassment", 4: "Threats", 5: "Extremism"}
    
    print("\nFinal counts:")
    for label, count in dist.items():
        print(f"  Class {label} ({label_names[label]}): {count} instances")

    # Save to new folder
    os.makedirs(output_dir, exist_ok=True)
    augmented_train[['index', 'message']].to_csv(os.path.join(output_dir, "train_index_text.csv"), index=False)
    augmented_train[['index', 'label']].to_csv(os.path.join(output_dir, "train_index_label.csv"), index=False)
    
    print(f"\n✓ Balanced augmented dataset saved to {output_dir}")
    return dist

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/usydnlp/CONDA/refs/heads/main/data/CONDA_train.csv"
    output_dir = "data/augmented"
    augment_dataset(url, output_dir)
