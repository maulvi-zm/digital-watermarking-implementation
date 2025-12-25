"""Comprehensive evaluation script."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import config
from src.model import SimpleCNN
from src.watermark import WatermarkGenerator
from src.evaluator import evaluate_fidelity, evaluate_reliability
from src.utils import load_model, get_mnist_loaders, save_results


def main():
    """Main evaluation function."""
    print("="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    
    device = torch.device(config.TRAIN_CONFIG["device"])
    
    # Load models
    print("\nLoading models...")
    baseline_model = load_model(
        SimpleCNN,
        config.PATHS["baseline_model"],
        num_classes=config.MODEL_CONFIG["num_classes"]
    )
    
    watermarked_model = load_model(
        SimpleCNN,
        config.PATHS["watermarked_model"],
        num_classes=config.MODEL_CONFIG["num_classes"]
    )
    
    # Load watermark key
    print("Loading watermark key...")
    key_data = torch.load(config.PATHS["watermark_key"], map_location="cpu")
    watermark = key_data["watermark"]
    key = key_data["key"]
    watermark_length = key_data["length"]
    
    # Initialize watermark generator
    watermark_gen = WatermarkGenerator(
        length=watermark_length,
        seed=config.WATERMARK_CONFIG["seed"]
    )
    
    # Load test data
    print("Loading test data...")
    _, test_loader = get_mnist_loaders(
        batch_size=config.TRAIN_CONFIG["batch_size"],
        data_dir=str(config.DATA_DIR)
    )
    
    # Evaluate fidelity
    print("\nEvaluating model fidelity...")
    fidelity_results = evaluate_fidelity(
        baseline_model,
        watermarked_model,
        test_loader,
        device=device
    )
    
    # Evaluate reliability
    print("Evaluating watermark reliability...")
    reliability_results = evaluate_reliability(
        watermarked_model,
        watermark_gen,
        watermark,
        key,
        target_layer=config.WATERMARK_CONFIG["target_layer"]
    )
    
    # Compile results
    results = {
        "baseline": {
            "accuracy": fidelity_results["baseline_accuracy"]
        },
        "watermarked": {
            "accuracy": fidelity_results["watermarked_accuracy"],
            "fidelity": {
                "accuracy_drop": fidelity_results["accuracy_drop"],
                "relative_drop": fidelity_results["relative_drop"]
            }
        },
        "verification": reliability_results
    }
    
    # Save results
    print("\nSaving results...")
    save_results(
        results,
        config.PATHS["results_json"],
        config.PATHS["results_txt"]
    )
    
    # Display summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nBaseline Model:")
    print(f"  Accuracy: {fidelity_results['baseline_accuracy']:.2f}%")
    
    print(f"\nWatermarked Model:")
    print(f"  Accuracy: {fidelity_results['watermarked_accuracy']:.2f}%")
    print(f"  Accuracy Drop: {fidelity_results['accuracy_drop']:.2f}%")
    print(f"  Relative Drop: {fidelity_results['relative_drop']:.2f}%")
    
    print(f"\nWatermark Verification:")
    print(f"  Bit Accuracy: {reliability_results['bit_accuracy']:.4f}")
    print(f"  Exact Match: {reliability_results['exact_match']}")
    print(f"  Matches: {reliability_results['matches']}/{reliability_results['total']}")
    
    return results


if __name__ == "__main__":
    main()

