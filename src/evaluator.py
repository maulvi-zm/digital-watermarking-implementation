"""Evaluation metrics for model fidelity and watermark reliability."""

from src.utils import evaluate_model
from src.extractor import verify_watermark


def evaluate_fidelity(baseline_model, watermarked_model, test_loader, device="cpu"):
    """
    Evaluate model fidelity (accuracy difference).
    
    Args:
        baseline_model: Non-watermarked model
        watermarked_model: Watermarked model
        test_loader: Test data loader
        device: Device to run on
    
    Returns:
        Dictionary with fidelity metrics
    """
    baseline_acc = evaluate_model(baseline_model, test_loader, device)
    watermarked_acc = evaluate_model(watermarked_model, test_loader, device)
    
    accuracy_drop = baseline_acc - watermarked_acc
    
    return {
        "baseline_accuracy": baseline_acc,
        "watermarked_accuracy": watermarked_acc,
        "accuracy_drop": accuracy_drop,
        "relative_drop": (accuracy_drop / baseline_acc * 100) if baseline_acc > 0 else 0
    }


def evaluate_reliability(model, watermark_gen, original_watermark, key, target_layer="conv2"):
    """
    Evaluate watermark reliability.
    
    Args:
        model: Watermarked model
        watermark_gen: WatermarkGenerator instance
        original_watermark: Original watermark
        key: Secret key
        target_layer: Layer to extract from
    
    Returns:
        Dictionary with reliability metrics
    """
    verification = verify_watermark(
        model,
        watermark_gen,
        original_watermark,
        key,
        target_layer
    )
    
    return {
        "bit_accuracy": verification["bit_accuracy"],
        "exact_match": verification["exact_match"],
        "matches": verification["matches"],
        "total": verification["total"],
        "hamming_distance": verification["hamming_distance"]
    }

