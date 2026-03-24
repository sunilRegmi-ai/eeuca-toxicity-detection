import pandas as pd
import numpy as np
import os
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    # Remove excessive punctuation and game-specific chat noise
    text = re.sub(r'[^\w\s\?!\']', ' ', text)
    return " ".join(text.lower().split())

def augment_data():
    print("Starting data augmentation using CONDA subset...")
    
    # 1. Load our current merged data
    curr_text = pd.read_csv("data/merged/merged_text.csv")
    curr_label = pd.read_csv("data/merged/merged_label.csv")
    curr_df = pd.merge(curr_text, curr_label, on='index')
    
    # 2. Load CONDA
    conda_df = pd.read_csv("data/CONDA.csv", sep=';')
    # Filter only toxic messages (category_id 0)
    conda_tox = conda_df[conda_df['category_id'] == 0].copy()
    conda_tox['full_text'] = conda_tox['full_text'].astype(str)
    
    # 3. Define keyword-based extraction
    # Class 3: Hate and Harassment
    hate_keywords = r'nigger|faggot|retard|cunt|slut|bitch|whore|mother|son of a|mongoloid|jew|gay'
    # Class 4: Threats
    threat_keywords = r'kill|die|death|murder|hang|shoot|stab|slit|burn'
    # Class 5: Extremism
    extreme_keywords = r'nazi|hitler|isis|jihad|terrorist|white power|heil|holocaust'
    
    def get_class(text):
        if not isinstance(text, str): return None
        if re.search(extreme_keywords, text, re.I): return 5
        if re.search(threat_keywords, text, re.I): return 4
        if re.search(hate_keywords, text, re.I): return 3
        return None # General toxicity we don't necessarily need if we want balance

    conda_tox['target_label'] = conda_tox['full_text'].apply(get_class)
    
    # Filter only those that mapped to our minority classes
    new_samples = conda_tox[conda_tox['target_label'].notnull()].copy()
    
    print("\nSamples found in CONDA for minority classes:")
    print(new_samples['target_label'].value_counts().sort_index())
    
    # 4. Prepare for merging
    # Format new samples to match our columns
    # We'll use a unique index prefix to avoid collisions
    new_samples['index'] = "conda_" + new_samples.index.astype(str)
    new_samples = new_samples.rename(columns={'full_text': 'message'})
    
    aug_df = new_samples[['index', 'message', 'target_label']].rename(columns={'target_label': 'label'})
    
    # Clean text
    aug_df['message'] = aug_df['message'].apply(clean_text)
    
    # 5. Merge and save
    final_df = pd.concat([curr_df, aug_df], ignore_index=True)
    
    print(f"\nFinal Class Distribution (Old + CONDA Augmentation):")
    print(final_df['label'].value_counts().sort_index())
    
    output_dir = "data/augmented_conda"
    os.makedirs(output_dir, exist_ok=True)
    
    final_df[['index', 'message']].to_csv(os.path.join(output_dir, "conda_augmented_text.csv"), index=False)
    final_df[['index', 'label']].to_csv(os.path.join(output_dir, "conda_augmented_label.csv"), index=False)
    
    print(f"\n✓ Augmented dataset saved to {output_dir}")

if __name__ == "__main__":
    augment_data()
