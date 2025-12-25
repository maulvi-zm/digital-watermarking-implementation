"""Watermark extraction and verification."""

from src.watermark import WatermarkGenerator


def extract_watermark(model, watermark_gen, key, watermark_length, target_layer="conv2"):
    """
    Extract watermark from trained model.
    
    Args:
        model: Trained PyTorch model
        watermark_gen: WatermarkGenerator instance
        key: Secret key used for embedding
        watermark_length: Length of watermark to extract
        target_layer: Layer to extract watermark from
    
    Returns:
        Extracted watermark vector
    """
    # Get parameters from target layer
    params = model.get_target_layer_params(target_layer)
    
    # Decode watermark
    extracted = watermark_gen.decode(params, key, watermark_length)
    
    return extracted


def verify_watermark(model, watermark_gen, original_watermark, key, target_layer="conv2"):
    """
    Verify watermark in model.
    
    Args:
        model: Trained PyTorch model
        watermark_gen: WatermarkGenerator instance
        original_watermark: Original watermark vector
        key: Secret key used for embedding
        target_layer: Layer to extract watermark from
    
    Returns:
        Dictionary with verification results
    """
    # Extract watermark
    extracted = extract_watermark(
        model,
        watermark_gen,
        key,
        len(original_watermark),
        target_layer
    )
    
    # Compare with original
    comparison = watermark_gen.compare(original_watermark, extracted)
    
    return {
        "extracted_watermark": extracted.tolist(),
        "original_watermark": original_watermark.tolist(),
        **comparison
    }


