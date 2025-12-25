"""Watermark generation and encoding/decoding functions."""

import torch
import numpy as np


class WatermarkGenerator:
    """Generate and manage binary watermarks."""
    
    def __init__(self, length=128, seed=42):
        """
        Initialize watermark generator.
        
        Args:
            length: Length of binary watermark vector
            seed: Random seed for reproducibility
        """
        self.length = length
        self.seed = seed
        self.watermark = None
        self.key = None
    
    def generate(self):
        """Generate a random binary watermark vector."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Generate random binary watermark
        self.watermark = torch.randint(0, 2, (self.length,), dtype=torch.float32)
        
        # Generate secret key (random indices for embedding)
        self.key = torch.randperm(self.length)
        
        return self.watermark.clone(), self.key.clone()
    
    def encode(self, params, watermark, key, embedding_strength=0.1):
        """
        Encode watermark into model parameters using fixed normalization.
        
        Args:
            params: Model parameters (flattened tensor)
            watermark: Binary watermark vector
            key: Secret key for embedding
            embedding_strength: Strength of watermark embedding (0-1)
        
        Returns:
            Encoded parameters
        """
        if len(params) < len(watermark):
            raise ValueError("Parameters vector too short for watermark")
        
        encoded_params = params.clone()
        
        # Select parameters using key
        selected_indices = key[:len(watermark)].long() % len(params)
        selected_params = params[selected_indices]
        
        # Normalize selected parameters to [0, 1] range using FIXED min/max
        param_min = selected_params.min()
        param_max = selected_params.max()
        
        if param_max > param_min:
            normalized_params = (selected_params - param_min) / (param_max - param_min)
        else:
            # If all params are same, can't normalize - use small perturbation
            normalized_params = selected_params.clone()
        
        # Encode watermark by modifying normalized parameters to match watermark bits
        # Use embedding_strength to control how strongly we push towards watermark values
        for i, bit in enumerate(watermark):
            target_value = bit.item()  # Target normalized value (0 or 1)
            current_value = normalized_params[i].item()
            
            # Move normalized parameter towards target watermark bit
            # Use embedding_strength to control the magnitude of change
            new_normalized = current_value + embedding_strength * (target_value - current_value)
            normalized_params[i] = new_normalized
        
        # Denormalize back to original parameter scale
        if param_max > param_min:
            encoded_selected = normalized_params * (param_max - param_min) + param_min
        else:
            encoded_selected = normalized_params
        
        # Update encoded_params with modified selected parameters
        for i, idx in enumerate(selected_indices):
            encoded_params[idx] = encoded_selected[i]
        
        return encoded_params
    
    def decode(self, params, key, watermark_length):
        """
        Decode watermark from model parameters.
        
        Args:
            params: Model parameters (flattened tensor)
            key: Secret key used for embedding
            watermark_length: Length of watermark to extract
        
        Returns:
            Extracted binary watermark vector
        """
        if len(params) < watermark_length:
            raise ValueError("Parameters vector too short for watermark extraction")
        
        extracted = torch.zeros(watermark_length)
        selected_indices = key[:watermark_length].long() % len(params)
        selected_params = params[selected_indices]
        
        # Normalize selected parameters to [0, 1] range (same as embedding)
        param_min = selected_params.min()
        param_max = selected_params.max()
        if param_max > param_min:
            normalized_params = (selected_params - param_min) / (param_max - param_min)
        else:
            normalized_params = selected_params
        
        # Extract watermark by thresholding normalized parameters
        # Since watermark bits are 0 or 1, use 0.5 as threshold on normalized [0,1] values
        for i in range(watermark_length):
            if normalized_params[i] > 0.5:
                extracted[i] = 1.0
            else:
                extracted[i] = 0.0
        
        return extracted
    
    def compare(self, original, extracted):
        """
        Compare original and extracted watermarks.
        
        Args:
            original: Original watermark vector
            extracted: Extracted watermark vector
        
        Returns:
            Dictionary with comparison metrics
        """
        # Convert to binary
        orig_binary = (original > 0.5).float()
        extr_binary = (extracted > 0.5).float()
        
        # Calculate metrics
        matches = (orig_binary == extr_binary).sum().item()
        total = len(original)
        bit_accuracy = matches / total
        exact_match = (orig_binary == extr_binary).all().item()
        
        return {
            "bit_accuracy": bit_accuracy,
            "exact_match": exact_match,
            "matches": matches,
            "total": total,
            "hamming_distance": (orig_binary != extr_binary).sum().item()
        }

