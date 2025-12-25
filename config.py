"""Configuration file for digital watermarking experiments."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model hyperparameters
MODEL_CONFIG = {
    "input_channels": 1,
    "num_classes": 10,
    "hidden_size": 128,
}

# Training hyperparameters
TRAIN_CONFIG = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 10,
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

# Watermark parameters
WATERMARK_CONFIG = {
    "length": 128,  # Binary watermark length
    "lambda": 0.1,  # Embedding strength (regularization coefficient)
    "target_layer": "conv2",  # Layer to embed watermark in
    "seed": 42,  # Random seed for reproducibility
}

# Robustness test parameters
ROBUSTNESS_CONFIG = {
    "fine_tune_epochs": 5,
    "fine_tune_lr": 0.0001,
    "pruning_ratio": 0.3,  # Percentage of parameters to prune
}

# File paths
PATHS = {
    "baseline_model": MODELS_DIR / "baseline_model.pth",
    "watermarked_model": MODELS_DIR / "watermarked_model.pth",
    "watermark_key": MODELS_DIR / "watermark_key.pth",
    "results_json": RESULTS_DIR / "results.json",
    "results_txt": RESULTS_DIR / "results.txt",
}

