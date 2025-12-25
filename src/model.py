"""Simple CNN model for MNIST classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN architecture for MNIST digit classification."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_target_layer_params(self, layer_name="conv2"):
        """Get parameters from a specific layer for watermark embedding."""
        if layer_name == "conv2":
            return self.conv2.weight.data.flatten()
        elif layer_name == "conv1":
            return self.conv1.weight.data.flatten()
        elif layer_name == "fc1":
            return self.fc1.weight.data.flatten()
        elif layer_name == "fc2":
            return self.fc2.weight.data.flatten()
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
    
    def set_target_layer_params(self, params, layer_name="conv2"):
        """Set parameters back into a specific layer after watermark embedding."""
        if layer_name == "conv2":
            self.conv2.weight.data = params.reshape(self.conv2.weight.data.shape)
        elif layer_name == "conv1":
            self.conv1.weight.data = params.reshape(self.conv1.weight.data.shape)
        elif layer_name == "fc1":
            self.fc1.weight.data = params.reshape(self.fc1.weight.data.shape)
        elif layer_name == "fc2":
            self.fc2.weight.data = params.reshape(self.fc2.weight.data.shape)
        else:
            raise ValueError(f"Unknown layer: {layer_name}")

