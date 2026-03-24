"""
smart_augment_conda.py
Uses Semantic Similarity (Sentence Transformers) to dynamically select
CONDA samples (Classes 3, 4, 5 only) that best match the test dataset distribution.
"""

import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    # Remove excessive punctuation and game-specific chat noise
    text = re.sub(r'[^\w\s\?!\']', ' ', text)
    return " ".join(text.lower().split())

def augment_data():
    print("Starting Smart Semantic Augmentation...")
    
    # 1. Load Test Set
    test_df = pd.read_csv("data/test_index_text.csv")
    test_df['message'] = test_df['message'].apply(clean_text)
    test_texts = test_df['message'].tolist()
    
    # 2. Load CONDA and strict filter
    conda_df = pd.read_csv("data/CONDA.csv", sep=';')
    conda_tox = conda_df[conda_df['category_id'] == 0].copy()
    conda_tox['full_text'] = conda_tox['full_text'].astype(str)
    
    hate_keywords = r'nigger|faggot|retard|cunt|slut|bitch|whore|mother|son of a|mongoloid|jew|gay'
    threat_keywords = r'kill|die|death|murder|hang|shoot|stab|slit|burn'
    extreme_keywords = r'nazi|hitler|isis|jihad|terrorist|white power|heil|holocaust'
    
    def get_class(text):
        if not isinstance(text, str): return None
        if re.search(extreme_keywords, text, re.I): return 5
        if re.search(threat_keywords, text, re.I): return 4
        if re.search(hate_keywords, text, re.I): return 3
        # Strict rule: DO NOT return 0, 1, or 2.
        return None 

    conda_tox['target_label'] = conda_tox['full_text'].apply(get_class)
    candidates = conda_tox[conda_tox['target_label'].notnull()].copy()
    candidates['message'] = candidates['full_text'].apply(clean_text)
    
    candidate_texts = candidates['message'].tolist()
    
    print(f"Test Set shape: {len(test_texts)}")
    print(f"Candidate CONDA samples (Classes 3,4,5): {len(candidate_texts)}")
    
    if len(candidate_texts) == 0:
        print("No candidates found for classes 3, 4, 5. Exiting.")
        return

    # 3. Compute Embeddings
    print("\nLoading SentenceTransformer model (all-MiniLM-L6-v2) for fast semantic matching...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("Computing embeddings for test set...")
    test_embeddings = model.encode(test_texts, show_progress_bar=True, device=device)
    
    print("Computing embeddings for CONDA candidates...")
    candidate_embeddings = model.encode(candidate_texts, show_progress_bar=True, device=device)
    
    # 4. Calculate Cosine Similarity
    print("\nCalculating cosine similarity matrix...")
    similarities = cosine_similarity(candidate_embeddings, test_embeddings)
    
    # We want candidates that have at least one highly similar match in the test set
    max_sim_scores = np.max(similarities, axis=1)
    candidates['sim_score'] = max_sim_scores
    
    # 5. Select Best Matches
    N_SAMPLES_PER_CLASS = {
        3: 800,  # Hate
        4: 200,  # Threats
        5: 50    # Extremism (limited candidate pool)
    }
    
    selected_dfs = []
    print("\nSelection Results:")
    for cls, n_samples in N_SAMPLES_PER_CLASS.items():
        cls_candidates = candidates[candidates['target_label'] == cls]
        cls_candidates = cls_candidates.sort_values(by='sim_score', ascending=False)
        top_candidates = cls_candidates.head(n_samples)
        selected_dfs.append(top_candidates)
        print(f"  Class {cls}: Selected {len(top_candidates)} samples (Max Similarity: {top_candidates['sim_score'].max():.4f}, Min: {top_candidates['sim_score'].min():.4f})")
        
    final_candidates = pd.concat(selected_dfs)
    print(f"\nFinal Selected Smart Augmented Samples: {len(final_candidates)}")
    print(f"Average Semantic Match Score: {final_candidates['sim_score'].mean():.4f}")
    
    # 6. Merge with Original Data
    final_candidates['index'] = "smart_conda_" + final_candidates.index.astype(str)
    aug_df = final_candidates[['index', 'message', 'target_label']].rename(columns={'target_label': 'label'})
    
    curr_text = pd.read_csv("data/train/train_index_text.csv")
    curr_label = pd.read_csv("data/train/train_index_label.csv")
    curr_df = pd.merge(curr_text, curr_label, on='index')
    
    smart_final_df = pd.concat([curr_df, aug_df], ignore_index=True)
    
    # 7. Save output
    output_dir = "data/augmented_smart"
    os.makedirs(output_dir, exist_ok=True)
    
    smart_final_df[['index', 'message']].to_csv(os.path.join(output_dir, "smart_augmented_text.csv"), index=False)
    smart_final_df[['index', 'label']].to_csv(os.path.join(output_dir, "smart_augmented_label.csv"), index=False)
    
    print(f"\n✓ Smart Augmented dataset saved to {output_dir}")

if __name__ == "__main__":
    augment_data()
