"""
augment_jigsaw.py

Augments the original training data with samples from the 
Jigsaw Toxic Comment Classification Challenge dataset (data/train.csv).
Specifically targets the minority classes:
- Class 3 (Hate) 
- Class 4 (Threats)
- Class 5 (Extremism)
Strictly ignores all other classes to maintain balance.
"""

import os
import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^\w\s\?!\']', ' ', text)
    return " ".join(text.lower().split())

def augment_data():
    print("Loading Jigsaw Dataset (`data/train.csv`)...")
    jigsaw_df = pd.read_csv("data/train.csv")
    
    # Fill nan just in case
    jigsaw_df['comment_text'] = jigsaw_df['comment_text'].fillna("").astype(str)
    
    print("\nExtracting minority classes from Jigsaw...")
    
    # Empty DataFrame to store our newly mapped samples
    new_samples = []
    
    # 1. Threats (Class 4)
    # Jigsaw has a 'threat' column
    threats = jigsaw_df[jigsaw_df['threat'] == 1].copy()
    print(f"  > Found {len(threats)} Threat samples")
    for text in threats['comment_text']:
        new_samples.append({'message': clean_text(text), 'label': 4})
        
    # 2. Hate (Class 3) and Extremism (Class 5)
    # Jigsaw has an 'identity_hate' column which groups both
    id_hate = jigsaw_df[(jigsaw_df['identity_hate'] == 1) & (jigsaw_df['threat'] == 0)].copy()
    print(f"  > Found {len(id_hate)} Identity Hate samples")
    
    extreme_keywords = r'nazi|hitler|isis|jihad|terrorist|white power|heil|holocaust'
    
    for text in id_hate['comment_text']:
        cleaned = clean_text(text)
        # If it contains extremist keywords, designate as Class 5
        if re.search(extreme_keywords, cleaned, re.I):
            new_samples.append({'message': cleaned, 'label': 5})
        else:
            new_samples.append({'message': cleaned, 'label': 3})
            
    aug_df = pd.DataFrame(new_samples)
    
    # Remove extremely short strings or empty
    aug_df = aug_df[aug_df['message'].str.len() > 5]
    
    # Provide a unique index
    aug_df['index'] = ["jigsaw_" + str(i) for i in range(len(aug_df))]
    
    print("\nSummary of Augmented Samples:")
    counts = aug_df['label'].value_counts().sort_index()
    print(f"Class 3 (Hate): {counts.get(3, 0)}")
    print(f"Class 4 (Threats): {counts.get(4, 0)}")
    print(f"Class 5 (Extremism): {counts.get(5, 0)}")
    
    # 3. Merge with Original Dataset
    print("\nMerging with original training dataset...")
    curr_text = pd.read_csv("data/train/train_index_text.csv")
    curr_label = pd.read_csv("data/train/train_index_label.csv")
    curr_df = pd.merge(curr_text, curr_label, on='index')
    
    final_df = pd.concat([curr_df, aug_df], ignore_index=True)
    
    print("\nFinal Merged Class Distribution:")
    print(final_df['label'].value_counts().sort_index())
    
    # 4. Save results
    output_dir = "data/augmented_jigsaw"
    os.makedirs(output_dir, exist_ok=True)
    
    final_df[['index', 'message']].to_csv(os.path.join(output_dir, "jigsaw_augmented_text.csv"), index=False)
    final_df[['index', 'label']].to_csv(os.path.join(output_dir, "jigsaw_augmented_label.csv"), index=False)
    
    print(f"\n✓ Jigsaw Augmented dataset saved to {output_dir}")

if __name__ == "__main__":
    augment_data()
