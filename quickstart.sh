#!/bin/bash

# Quick start script for toxicity detection training

echo "=========================================="
echo "GameTox Toxicity Detection - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check data directory
if [ ! -d "data" ]; then
    echo "ERROR: data directory not found!"
    echo "Please ensure your data is organized as:"
    echo "  data/"
    echo "  ├── train/"
    echo "  │   ├── train_index_text.csv"
    echo "  │   └── train_index_label.csv"
    echo "  ├── val/"
    echo "  │   ├── val_index_text.csv"
    echo "  │   └── val_index_label.csv"
    echo "  └── test_index_text.csv"
    exit 1
fi

echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Run training
python train.py

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - predictions.csv (ready for submission)"
echo "  - model_comparison.csv"
echo "  - training_summary.json"
echo ""
echo "To create submission zip:"
echo "  zip predictions.zip predictions.csv"
echo ""
