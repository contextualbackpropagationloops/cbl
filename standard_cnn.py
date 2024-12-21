import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StandardCNN, self).__init__()
        # CIFAR-10 input: (3,32,32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # (32,32,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64,32,32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (128,32,32)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dims by half
        # After first pool (applied after conv2): (64,16,16)
        # After second pool (applied after conv3): (128,8,8)
        # Flattened features: 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (64,16,16)
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # (128,8,8)
        x = x.view(x.size(0), -1)  # (N,8192)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
