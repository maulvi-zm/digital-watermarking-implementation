"""Training script for baseline and watermarked models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from src.model import SimpleCNN
from src.watermark import WatermarkGenerator
# Post-training embedding approach - no longer using train_with_watermark
from src.utils import get_mnist_loaders, save_model, evaluate_model


def train_baseline(model, train_loader, test_loader, epochs=10, lr=0.001, device="cpu"):
    """Train baseline (non-watermarked) model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                "loss": f"{running_loss/(len(pbar)+1):.4f}",
                "acc": f"{100*correct/total:.2f}%"
            })
        
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Test Acc={test_acc:.2f}%")
    
    return model


def main():
    """Main training function."""
    print("="*80)
    print("DIGITAL WATERMARKING - TRAINING")
    print("="*80)
    
    # Setup
    device = torch.device(config.TRAIN_CONFIG["device"])
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=config.TRAIN_CONFIG["batch_size"],
        data_dir=str(config.DATA_DIR)
    )
    
    # Train baseline model
    print("\n" + "="*80)
    print("TRAINING BASELINE MODEL")
    print("="*80)
    baseline_model = SimpleCNN(num_classes=config.MODEL_CONFIG["num_classes"])
    baseline_model = train_baseline(
        baseline_model,
        train_loader,
        test_loader,
        epochs=config.TRAIN_CONFIG["epochs"],
        lr=config.TRAIN_CONFIG["learning_rate"],
        device=device
    )
    
    baseline_acc = evaluate_model(baseline_model, test_loader, device)
    print(f"\nBaseline Model Final Accuracy: {baseline_acc:.2f}%")
    
    # Save baseline model
    save_model(baseline_model, config.PATHS["baseline_model"])
    
    # Generate watermark
    print("\n" + "="*80)
    print("GENERATING WATERMARK")
    print("="*80)
    watermark_gen = WatermarkGenerator(
        length=config.WATERMARK_CONFIG["length"],
        seed=config.WATERMARK_CONFIG["seed"]
    )
    watermark, key = watermark_gen.generate()
    print(f"Watermark length: {len(watermark)}")
    print(f"Watermark sample: {watermark[:10].tolist()}")
    
    # Save watermark key
    torch.save({
        "watermark": watermark,
        "key": key,
        "length": len(watermark),
        "seed": config.WATERMARK_CONFIG["seed"]
    }, config.PATHS["watermark_key"])
    print(f"Watermark key saved to {config.PATHS['watermark_key']}")
    
    # Train watermarked model (normally, without watermark loss)
    print("\n" + "="*80)
    print("TRAINING WATERMARKED MODEL")
    print("="*80)
    watermarked_model = SimpleCNN(num_classes=config.MODEL_CONFIG["num_classes"])
    watermarked_model = train_baseline(
        watermarked_model,
        train_loader,
        test_loader,
        epochs=config.TRAIN_CONFIG["epochs"],
        lr=config.TRAIN_CONFIG["learning_rate"],
        device=device
    )
    
    watermarked_acc = evaluate_model(watermarked_model, test_loader, device)
    print(f"\nWatermarked Model Accuracy (before embedding): {watermarked_acc:.2f}%")
    
    # Embed watermark post-training
    print("\n" + "="*80)
    print("EMBEDDING WATERMARK (POST-TRAINING)")
    print("="*80)
    target_layer = config.WATERMARK_CONFIG["target_layer"]
    
    # Get parameters from target layer
    params = watermarked_model.get_target_layer_params(target_layer)
    print(f"Original parameters shape: {params.shape}")
    print(f"Watermark length: {len(watermark)}")
    
    # Encode watermark into parameters
    # Use lambda as embedding strength (convert from regularization coefficient to embedding strength)
    embedding_strength = min(config.WATERMARK_CONFIG["lambda"], 1.0)
    encoded_params = watermark_gen.encode(params, watermark, key, embedding_strength=embedding_strength)
    
    # Set encoded parameters back into model
    watermarked_model.set_target_layer_params(encoded_params, target_layer)
    
    # Verify watermark was embedded
    print("Verifying watermark embedding...")
    from src.extractor import verify_watermark
    verification = verify_watermark(
        watermarked_model,
        watermark_gen,
        watermark,
        key,
        target_layer=target_layer
    )
    print(f"Watermark bit accuracy after embedding: {verification['bit_accuracy']:.4f} ({verification['bit_accuracy']*100:.2f}%)")
    print(f"Matches: {verification['matches']}/{verification['total']}")
    
    # Re-evaluate model accuracy after watermark embedding
    watermarked_acc_after = evaluate_model(watermarked_model, test_loader, device)
    print(f"\nWatermarked Model Accuracy (after embedding): {watermarked_acc_after:.2f}%")
    print(f"Accuracy change: {watermarked_acc_after - watermarked_acc:.2f}%")
    
    # Save watermarked model
    save_model(watermarked_model, config.PATHS["watermarked_model"])
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Watermarked Accuracy: {watermarked_acc:.2f}%")
    print(f"Accuracy Drop: {baseline_acc - watermarked_acc:.2f}%")


if __name__ == "__main__":
    main()

