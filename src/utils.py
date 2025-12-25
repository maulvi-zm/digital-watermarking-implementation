"""Utility functions for data loading, model saving, and visualization."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from pathlib import Path


def get_mnist_loaders(batch_size=64, data_dir="./data"):
    """
    Get MNIST data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/download data
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader


def save_model(model, path, metadata=None):
    """
    Save model to file.
    
    Args:
        model: PyTorch model
        path: Path to save model
        metadata: Optional metadata dictionary
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    
    if metadata:
        save_dict["metadata"] = metadata
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(model_class, path, **model_kwargs):
    """
    Load model from file.
    
    Args:
        model_class: Model class to instantiate
        path: Path to saved model
        **model_kwargs: Keyword arguments for model initialization
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(path, map_location="cpu")
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
    
    Returns:
        Test accuracy (float)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def save_results(results, json_path, txt_path):
    """
    Save experimental results to JSON and text files.
    
    Args:
        results: Dictionary containing results
        json_path: Path to save JSON file
        txt_path: Path to save text file
    """
    # Save as JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save as human-readable text
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENTAL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        if "baseline" in results:
            f.write("BASELINE MODEL:\n")
            f.write(f"  Test Accuracy: {results['baseline']['accuracy']:.2f}%\n\n")
        
        if "watermarked" in results:
            f.write("WATERMARKED MODEL:\n")
            f.write(f"  Test Accuracy: {results['watermarked']['accuracy']:.2f}%\n")
            f.write(f"  Model Fidelity: {results['watermarked'].get('fidelity', 'N/A')}\n\n")
        
        if "verification" in results:
            f.write("WATERMARK VERIFICATION:\n")
            verif = results["verification"]
            f.write(f"  Bit Accuracy: {verif['bit_accuracy']:.4f}\n")
            f.write(f"  Exact Match: {verif['exact_match']}\n")
            f.write(f"  Matches: {verif['matches']}/{verif['total']}\n")
            f.write(f"  Hamming Distance: {verif['hamming_distance']}\n\n")
        
        if "robustness" in results:
            f.write("ROBUSTNESS TESTS:\n")
            robust = results["robustness"]
            
            if "fine_tuning" in robust:
                ft = robust["fine_tuning"]
                f.write(f"  Fine-tuning Attack:\n")
                f.write(f"    Accuracy After Attack: {ft['accuracy']:.2f}%\n")
                f.write(f"    Watermark Bit Accuracy: {ft['bit_accuracy']:.4f}\n")
                f.write(f"    Watermark Survived: {ft['survived']}\n\n")
            
            if "pruning" in robust:
                pr = robust["pruning"]
                f.write(f"  Pruning Attack:\n")
                f.write(f"    Accuracy After Attack: {pr['accuracy']:.2f}%\n")
                f.write(f"    Watermark Bit Accuracy: {pr['bit_accuracy']:.4f}\n")
                f.write(f"    Watermark Survived: {pr['survived']}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Results saved to {json_path} and {txt_path}")

