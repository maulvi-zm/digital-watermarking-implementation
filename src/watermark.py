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
    
    def encode(self, params, watermark, key):
        """
        Encode watermark into model parameters.
        
        Args:
            params: Model parameters (flattened tensor)
            watermark: Binary watermark vector
            key: Secret key for embedding
        
        Returns:
            Encoded parameters
        """
        if len(params) < len(watermark):
            raise ValueError("Parameters vector too short for watermark")
        
        encoded_params = params.clone()
        
        # Select parameters using key
        selected_indices = key[:len(watermark)].long() % len(params)
        
        # Encode watermark by modifying selected parameters
        # Simple encoding: add small perturbation based on watermark bit
        for i, bit in enumerate(watermark):
            idx = selected_indices[i].item()
            if bit > 0.5:  # Bit is 1
                encoded_params[idx] += 0.01
            else:  # Bit is 0
                encoded_params[idx] -= 0.01
        
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
        
        # Decode by checking parameter values
        # Threshold-based decoding
        threshold = params[selected_indices].mean()
        
        for i in range(watermark_length):
            idx = selected_indices[i].item()
            if params[idx] > threshold:
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

