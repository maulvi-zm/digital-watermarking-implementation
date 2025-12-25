"""Robustness testing: fine-tuning and pruning attacks."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils import evaluate_model
from src.extractor import verify_watermark


def fine_tune_attack(
    model,
    train_loader,
    test_loader,
    epochs=5,
    lr=0.0001,
    device="cpu",
    subset_ratio=0.5
):
    """
    Fine-tuning attack: retrain model on subset of data.
    
    Args:
        model: Trained model to attack
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of fine-tuning epochs
        lr: Learning rate for fine-tuning
        device: Device to run on
        subset_ratio: Ratio of training data to use
    
    Returns:
        Fine-tuned model
    """
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use subset of training data
    total_batches = len(train_loader)
    subset_size = int(total_batches * subset_ratio)
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= subset_size:
                break
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / subset_size
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Fine-tuning Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Test Acc={test_acc:.2f}%")
    
    return model


def pruning_attack(model, pruning_ratio=0.3):
    """
    Pruning attack: remove low-magnitude parameters.
    
    Args:
        model: Trained model to attack
        pruning_ratio: Percentage of parameters to prune
    
    Returns:
        Pruned model
    """
    model = model.train()
    
    # Collect all parameters
    all_params = []
    for param in model.parameters():
        all_params.append(param.data.flatten())
    
    all_params_tensor = torch.cat(all_params)
    
    # Calculate threshold
    k = int(len(all_params_tensor) * pruning_ratio)
    threshold = torch.kthvalue(all_params_tensor.abs(), k)[0]
    
    # Prune parameters below threshold
    for param in model.parameters():
        mask = param.data.abs() > threshold
        param.data *= mask.float()
    
    return model


def test_robustness(
    model,
    train_loader,
    test_loader,
    watermark_gen,
    original_watermark,
    key,
    target_layer="conv2",
    fine_tune_epochs=5,
    fine_tune_lr=0.0001,
    pruning_ratio=0.3,
    device="cpu"
):
    """
    Test watermark robustness against attacks.
    
    Args:
        model: Watermarked model
        train_loader: Training data loader (for fine-tuning attack)
        test_loader: Test data loader
        watermark_gen: WatermarkGenerator instance
        original_watermark: Original watermark
        key: Secret key
        target_layer: Layer to extract watermark from
        fine_tune_epochs: Epochs for fine-tuning attack
        fine_tune_lr: Learning rate for fine-tuning
        pruning_ratio: Ratio for pruning attack
        device: Device to run on
    
    Returns:
        Dictionary with robustness test results
    """
    results = {}
    
    # Test fine-tuning attack
    print("\n" + "="*60)
    print("Testing Fine-tuning Attack")
    print("="*60)
    
    # Clone model for fine-tuning attack to avoid modifying original
    import copy
    fine_tune_model = copy.deepcopy(model)
    fine_tuned_model = fine_tune_attack(
        fine_tune_model,
        train_loader,
        test_loader,
        epochs=fine_tune_epochs,
        lr=fine_tune_lr,
        device=device
    )
    
    fine_tuned_acc = evaluate_model(fine_tuned_model, test_loader, device)
    fine_tune_verification = verify_watermark(
        fine_tuned_model,
        watermark_gen,
        original_watermark,
        key,
        target_layer
    )
    
    results["fine_tuning"] = {
        "accuracy": fine_tuned_acc,
        "bit_accuracy": fine_tune_verification["bit_accuracy"],
        "exact_match": fine_tune_verification["exact_match"],
        "survived": fine_tune_verification["bit_accuracy"] > 0.7  # Threshold for survival
    }
    
    print(f"Fine-tuning Results:")
    print(f"  Accuracy: {fine_tuned_acc:.2f}%")
    print(f"  Watermark Bit Accuracy: {fine_tune_verification['bit_accuracy']:.4f}")
    print(f"  Watermark Survived: {results['fine_tuning']['survived']}")
    
    # Test pruning attack
    print("\n" + "="*60)
    print("Testing Pruning Attack")
    print("="*60)
    
    # Clone model for pruning attack to avoid modifying original
    prune_model = copy.deepcopy(model)
    pruned_model = pruning_attack(prune_model, pruning_ratio=pruning_ratio)
    pruned_acc = evaluate_model(pruned_model, test_loader, device)
    prune_verification = verify_watermark(
        pruned_model,
        watermark_gen,
        original_watermark,
        key,
        target_layer
    )
    
    results["pruning"] = {
        "accuracy": pruned_acc,
        "bit_accuracy": prune_verification["bit_accuracy"],
        "exact_match": prune_verification["exact_match"],
        "survived": prune_verification["bit_accuracy"] > 0.7  # Threshold for survival
    }
    
    print(f"Pruning Results:")
    print(f"  Accuracy: {pruned_acc:.2f}%")
    print(f"  Watermark Bit Accuracy: {prune_verification['bit_accuracy']:.4f}")
    print(f"  Watermark Survived: {results['pruning']['survived']}")
    
    return results

