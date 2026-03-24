# GameTox Toxicity Detection

Training script for detecting toxicity in online gaming communities using the GameTox dataset from NAACL 2025.

## Dataset

The GameTox dataset contains 53,000 game chat utterances from World of Tanks, labeled with 6 toxicity categories:

- **0**: Non-toxic
- **1**: Insults and Flaming
- **2**: Other Offensive Texts
- **3**: Hate and Harassment
- **4**: Threats
- **5**: Extremism

## Data Preparation

**Note**: The experiments strictly use the original GameTox dataset. No external datasets or augmented data were used in the final methodology.

## Methodology

We explore various transformer-based architectures for classifying game chat toxicity. Based on our training logs, our best performing architecture utilizes the **mmBERT** model (`jhu-clsp/mmBERT-small` and `jhu-clsp/mmBERT-base`).

### Training Configuration
The optimal single mmBERT model was trained with the following hyperparameters:
- **Maximum Sequence Length**: 64
- **Batch Size**: 32
- **Epochs**: 30 (with Early Stopping patience of 5)
- **Learning Rate**: 2e-05
- **Optimizer Weight Decay**: 0.01
- **Hardware Setup**: 2 GPUs (CUDA)

### Experimental Results
Evaluation is based on the **Macro F1-score**, giving equal weight to all toxicity classes. 
- The best single `mmBERT` model achieved a Macro F1 score of **0.6789** during validation.
- Additional methodologies explored include K-Fold Ensembling (`mmBERT-base` achieving up to 0.657 Macro F1 on 2 folds) and SMOTE paired with XGBoost.

## Installation

```bash
pip install -r requirements.txt
```

## Data Structure

Ensure your data is organized as follows:

```
data/
├── train/
│   ├── train_index_text.csv
│   └── train_index_label.csv
├── val/
│   ├── val_index_text.csv
│   └── val_index_label.csv
└── test_index_text.csv
```

## Usage

### Training

Run the training script:

```bash
python train.py
```

The script will:
1. Load and preprocess the data
2. Train all 4 models for 3 epochs each
3. Evaluate on validation set using F1-score (Macro)
4. Compare model performance
5. Generate predictions using the best model

### Output Files

After training, the following files will be generated:

- `predictions.csv` - Predictions for test data (ready for submission)
- `model_comparison.csv` - Performance comparison of all models
- `training_summary.json` - Training configuration and results
- `best_model_*.pt` - Saved model weights for each model

### Creating Submission

To create the submission zip file:

```bash
zip predictions.zip predictions.csv
```

**Important**: Ensure the zip file contains only `predictions.csv` with no subdirectories.

## Submission Format

The `predictions.csv` file will have the following format:

```csv
index,label
12345,0
15001,1
20524,1
35231,0
65102,1
```

- Indices are sorted in ascending order
- Labels are integers from 0-5

## Evaluation Metric

Models are ranked by **F1-score (Macro)**, which gives equal weight to all classes regardless of their frequency.

## Configuration

You can modify training parameters in the `Config` class in `train.py`:

- `MAX_LENGTH`: Maximum sequence length (default: 128)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `EPOCHS`: Number of training epochs (default: 3)
- `LEARNING_RATE`: Learning rate for optimizer (default: 2e-5)

## Hardware Requirements

- **GPU**: Recommended for faster training (CUDA-compatible)
- **RAM**: At least 8GB
- **Storage**: ~2GB for model weights and data

## Training Time

Approximate training time per model (on GPU):
- BERT: ~30-40 minutes
- RoBERTa: ~30-40 minutes
- DistilBERT: ~20-30 minutes
- ALBERT: ~25-35 minutes

Total training time: ~2-3 hours for all 4 models

## Tips for Better Performance

1. **Increase epochs**: Try 5-10 epochs for better convergence
2. **Adjust batch size**: Larger batches (32-64) may improve stability
3. **Fine-tune learning rate**: Try 1e-5 or 3e-5
4. **Data augmentation**: Consider back-translation or synonym replacement
5. **Ensemble methods**: Combine predictions from multiple models

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` to 8 or 4
- Reduce `MAX_LENGTH` to 64

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce number of models to train
- Use DistilBERT for faster training

## References

- GameTox Dataset: [GitHub Repository](https://github.com/therealthapa/eeuca-toxicity)
- Paper: Naseem et al., NAACL 2025

## License

Please refer to the original dataset repository for licensing information.
