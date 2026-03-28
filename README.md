# Multilingual Toxicity Detection Pipeline

This directory contains a flexible, configuration-driven training pipeline for multilingual toxicity detection models. This README is generated systematically based on the execution script (`train_multilingual.py`), configurations (`../config/config.yaml`), and our evaluated empirical results (`multilingual_results.csv`).

## Overview

The `train_multilingual.py` script leverages the Hugging Face `Trainer` API to fine-tune pre-trained transformer models. The entire training flow—from datasets and models to hyperparameters—is coordinated through `config.yaml`.

### Key Features
- **Sequential Multi-Model Training:** Automatically iterate over and train multiple architectures defined in the `models` block (e.g., `mmBERT`, `m-bert`, `xlm-roberta`).
- **Automated Data Merging:** Automatically loads text and label CSVs and merges them on the common `index` column. Checks fallback paths if `test_label` is omitted.
- **Class Imbalance Mitigation:** Computes balanced class weights dynamically via `sklearn` based on the training dataset. These weights are then applied to a customized `WeightedTrainer` cross-entropy loss function.
- **Early Stopping & Checkpoint Management:** Implements an `EarlyStoppingCallback` to monitor validation performance metrics (Macro F1) to prevent overfitting.
- **Comprehensive Logging:** Exports training hyperparameters, evaluation metrics, and timestamps to `multilingual_results.csv`.
- **FP16 Mixed Precision:** Activates FP16 training when CUDA is available (bypassed for DeBERTa variants to prevent nan loss issues).

---

## Experimental Results

The following section summarizes the peak results of the trained models, extracted directly from `multilingual_results.csv`. The pipeline automatically calculated test data metrics for macro accuracy and F1 scores under the given hyperparameters.

| Model Alias          | Base Model Path                          | Val F1 (Macro) | Test F1 (Macro) | Test Accuracy | Learning Rate | Batch Size / Seq Len | 
|----------------------|------------------------------------------|----------------|-----------------|---------------|---------------|----------------------|
| **mmbert**           | `jhu-clsp/mmBERT-base`                   | **0.5882**     | **0.4282**      | **0.8634**    | `1e-05`       | 64 / 32              |
| **m-bert**           | `bert-base-multilingual-uncased`         | 0.4064         | 0.4239          | 0.8305        | `3e-06`       | 32 / 64              |
| **xlm-roberta**      | `xlm-roberta-base`                       | 0.3830         | 0.3839          | 0.8130        | `3e-06`       | 32 / 64              |
| **m-distilbert**     | `distilbert-base-multilingual-cased`     | 0.3907         | 0.3578          | 0.7942        | `3e-06`       | 32 / 64              |
| **toxic-xlm-roberta**| `unitary/multilingual-toxic-xlm-roberta` | 0.3558         | 0.3520          | 0.8281        | `3e-06`       | 32 / 64              |

*(Note: Data derived from the highest F1 run per model variant. Parameters like `m-bert` and `xlm-roberta` were initially run using older parameters (`LR=3e-06, BS=32, MAX_LEN=64`) before being upgraded in later `mmBERT` runs)*

---

## Configuration (`config.yaml`)

The training script dynamically acts on parameters from `config.yaml`.

### Data Parameters
Target data partitions allocated for training:
- `train_text`, `train_label`: Training subset files
- `val_text`, `val_label`: Validation setup files
- `test_text`, `test_label`: Held-out testing files

### Target Models
The pipeline is currently configured to evaluate the following models:
- `xlm-roberta`: `"xlm-roberta-base"`
- `m-bert`: `"bert-base-multilingual-uncased"`
- `deberta`: `"microsoft/deberta-v3-base"`
- `mmbert`: `"jhu-clsp/mmBERT-base"`
- `m-distilbert`: `"distilbert-base-multilingual-cased"`
- `m-deberta`: `"microsoft/mdeberta-v3-base"`
- `toxic-xlm-roberta`: `"unitary/multilingual-toxic-xlm-roberta"`

### Current Hyperparameters
Settings for newly compiled executions:
- **`max_length`:** 32 (Sequence truncation limit)
- **`batch_size`:** 64
- **`epochs`:** 10 (Max number of epochs prior to Early Stopping)
- **`learning_rate`:** 1e-5
- **`weight_decay`:** 0.01 (L2 regularization factor)
- **`early_stopping_patience`:** 3
- **`num_classes`:** 6 toxicity categories

### Class Mapping
GameTox toxicity taxonomy:
- `0`: Non-toxic
- `1`: Insults and Flaming
- `2`: Other Offensive Texts
- `3`: Hate and Harassment  
- `4`: Threats
- `5`: Extremism

---

## Usage

Run the file directly. Checkpoints are dynamically saved into `./results/` partitioned by the defined model alias keys.

```bash
cd multilingual_training
python train_multilingual.py
```
