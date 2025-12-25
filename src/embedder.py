"""Watermark embedding during training."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.watermark import WatermarkGenerator
from src.utils import evaluate_model


def train_with_watermark(
    model,
    train_loader,
    test_loader,
    watermark_gen,
    watermark,
    key,
    lambda_reg=0.1,
    epochs=10,
    lr=0.001,
    device="cpu",
    target_layer="conv2"
):
    """
    Train model with watermark embedding.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        watermark_gen: WatermarkGenerator instance
        watermark: Watermark vector
        key: Secret key
        lambda_reg: Regularization strength
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        target_layer: Layer to embed watermark in
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            task_loss = criterion(outputs, labels)
            
            # Watermark loss
            params = model.get_target_layer_params(target_layer)
            watermark_loss = compute_watermark_loss(params, watermark, key, lambda_reg)
            
            # Combined loss
            total_loss = task_loss + watermark_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                "loss": f"{running_loss/(len(pbar)+1):.4f}",
                "acc": f"{100*correct/total:.2f}%"
            })
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
    
    return model, history


def compute_watermark_loss(params, watermark, key, lambda_reg):
    """
    Compute watermark embedding loss.
    
    Args:
        params: Model parameters (flattened)
        watermark: Target watermark vector
        key: Secret key
        lambda_reg: Regularization strength
    
    Returns:
        Watermark loss
    """
    if len(params) < len(watermark):
        # If params too short, pad or truncate watermark
        watermark = watermark[:len(params)]
        key = key[:len(params)]
    
    # Select parameters using key
    selected_indices = (key[:len(watermark)].long() % len(params))
    selected_params = params[selected_indices]
    
    # Normalize selected parameters to [0, 1] range for encoding
    param_min = selected_params.min()
    param_max = selected_params.max()
    if param_max > param_min:
        normalized_params = (selected_params - param_min) / (param_max - param_min)
    else:
        normalized_params = selected_params
    
    # Compute MSE loss between normalized params and watermark
    watermark_loss = lambda_reg * nn.functional.mse_loss(normalized_params, watermark)
    
    return watermark_loss


