"""Robustness testing script."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import config
from src.model import SimpleCNN
from src.watermark import WatermarkGenerator
from src.robustness import test_robustness
from src.utils import load_model, get_mnist_loaders


def main():
    """Main robustness testing function."""
    print("="*80)
    print("ROBUSTNESS TESTING")
    print("="*80)
    
    device = torch.device(config.TRAIN_CONFIG["device"])
    
    # Load watermark key
    print("\nLoading watermark key...")
    key_data = torch.load(config.PATHS["watermark_key"], map_location="cpu")
    watermark = key_data["watermark"]
    key = key_data["key"]
    watermark_length = key_data["length"]
    
    # Load watermarked model
    print("Loading watermarked model...")
    watermarked_model = load_model(
        SimpleCNN,
        config.PATHS["watermarked_model"],
        num_classes=config.MODEL_CONFIG["num_classes"]
    )
    
    # Initialize watermark generator
    watermark_gen = WatermarkGenerator(
        length=watermark_length,
        seed=config.WATERMARK_CONFIG["seed"]
    )
    
    # Load data
    print("Loading data...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=config.TRAIN_CONFIG["batch_size"],
        data_dir=str(config.DATA_DIR)
    )
    
    # Run robustness tests
    robustness_results = test_robustness(
        watermarked_model,
        train_loader,
        test_loader,
        watermark_gen,
        watermark,
        key,
        target_layer=config.WATERMARK_CONFIG["target_layer"],
        fine_tune_epochs=config.ROBUSTNESS_CONFIG["fine_tune_epochs"],
        fine_tune_lr=config.ROBUSTNESS_CONFIG["fine_tune_lr"],
        pruning_ratio=config.ROBUSTNESS_CONFIG["pruning_ratio"],
        device=device
    )
    
    print("\n" + "="*80)
    print("ROBUSTNESS TEST SUMMARY")
    print("="*80)
    print(f"\nFine-tuning Attack:")
    print(f"  Accuracy: {robustness_results['fine_tuning']['accuracy']:.2f}%")
    print(f"  Watermark Bit Accuracy: {robustness_results['fine_tuning']['bit_accuracy']:.4f}")
    print(f"  Watermark Survived: {robustness_results['fine_tuning']['survived']}")
    
    print(f"\nPruning Attack:")
    print(f"  Accuracy: {robustness_results['pruning']['accuracy']:.2f}%")
    print(f"  Watermark Bit Accuracy: {robustness_results['pruning']['bit_accuracy']:.4f}")
    print(f"  Watermark Survived: {robustness_results['pruning']['survived']}")
    
    return robustness_results


if __name__ == "__main__":
    main()

