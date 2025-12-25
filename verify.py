"""Watermark verification script."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import config
from src.model import SimpleCNN
from src.watermark import WatermarkGenerator
from src.extractor import verify_watermark
from src.utils import load_model, evaluate_model, get_mnist_loaders


def main():
    """Main verification function."""
    print("="*80)
    print("WATERMARK VERIFICATION")
    print("="*80)
    
    # Load watermark key
    print("\nLoading watermark key...")
    key_data = torch.load(config.PATHS["watermark_key"], map_location="cpu")
    watermark = key_data["watermark"]
    key = key_data["key"]
    watermark_length = key_data["length"]
    
    print(f"Watermark length: {watermark_length}")
    
    # Load watermarked model
    print("\nLoading watermarked model...")
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
    
    # Verify watermark
    print("\nExtracting and verifying watermark...")
    verification_results = verify_watermark(
        watermarked_model,
        watermark_gen,
        watermark,
        key,
        target_layer=config.WATERMARK_CONFIG["target_layer"]
    )
    
    # Display results
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    print(f"Bit Accuracy: {verification_results['bit_accuracy']:.4f} ({verification_results['bit_accuracy']*100:.2f}%)")
    print(f"Exact Match: {verification_results['exact_match']}")
    print(f"Matches: {verification_results['matches']}/{verification_results['total']}")
    print(f"Hamming Distance: {verification_results['hamming_distance']}")
    
    # Evaluate model accuracy
    print("\nEvaluating model accuracy...")
    _, test_loader = get_mnist_loaders(
        batch_size=config.TRAIN_CONFIG["batch_size"],
        data_dir=str(config.DATA_DIR)
    )
    accuracy = evaluate_model(watermarked_model, test_loader)
    print(f"Model Accuracy: {accuracy:.2f}%")
    
    return verification_results


if __name__ == "__main__":
    main()

