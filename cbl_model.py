import torch
import torch.nn as nn
import torch.nn.functional as F

class CBL_CNN(nn.Module):
    def __init__(self, num_classes=10, T=2, alpha=0.0):
        super(CBL_CNN, self).__init__()
        # Similar CNN as the standard model
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # (32,32,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64,32,32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (128,32,32)
        self.pool = nn.MaxPool2d(2, 2)
        # After pooling twice: (128,8,8)
        # Flatten: 128*8*8=8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # CBL parameters
        self.T = T
        self.alpha = alpha
        self.z_dim = 64

        # Context vector from final output
        self.g = nn.Linear(num_classes, self.z_dim)

        # Adapters
        # We'll refine h1, h2, h3, h4 (outputs after each major stage)
        # h1 shape: (N,32,32,32)
        # h2 shape after pool: (N,64,16,16)
        # h3 shape after pool: (N,128,8,8)
        # h4 shape: (N,256)
        # We'll insert a single-channel z-scalar and run a 1x1 conv or linear adapter.

        self.adapter_conv1 = nn.Sequential(
            nn.Conv2d(32 + 1, 32, kernel_size=1),
            nn.ReLU()
        )
        self.adapter_conv2 = nn.Sequential(
            nn.Conv2d(64 + 1, 64, kernel_size=1),
            nn.ReLU()
        )
        self.adapter_conv3 = nn.Sequential(
            nn.Conv2d(128 + 1, 128, kernel_size=1),
            nn.ReLU()
        )
        self.adapter_fc1 = nn.Sequential(
            nn.Linear(256 + self.z_dim, 256),
            nn.ReLU()
        )

    def forward_once(self, x):
        # Forward pass to record intermediate states
        h1 = F.relu(self.conv1(x))       # (N,32,32,32)
        h2 = self.pool(F.relu(self.conv2(h1)))  # (N,64,16,16)
        h3 = self.pool(F.relu(self.conv3(h2)))  # (N,128,8,8)
        h3_flat = h3.view(h3.size(0), -1)       # (N,8192)
        h4 = F.relu(self.fc1(h3_flat))           # (N,256)
        y = self.fc2(h4)                         # (N,num_classes)
        return h1, h2, h3, h4, y

    def refine_step(self, h1, h2, h3, h4, y):
        # Compute context vector z
        z = self.g(y) # (N,z_dim)

        # Use mean of z to create a scalar channel
        z_scalar = z.mean(dim=1, keepdim=True) # (N,1)

        # Refine h1
        h1_input = torch.cat([h1, z_scalar.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h1.size(2), h1.size(3))], dim=1)
        h1_new = self.adapter_conv1(h1_input)
        h1_refined = self.alpha * h1 + (1 - self.alpha) * h1_new

        # Refine h2
        h2_input = torch.cat([h2, z_scalar.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h2.size(2), h2.size(3))], dim=1)
        h2_new = self.adapter_conv2(h2_input)
        h2_refined = self.alpha * h2 + (1 - self.alpha) * h2_new

        # Refine h3
        h3_input = torch.cat([h3, z_scalar.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h3.size(2), h3.size(3))], dim=1)
        h3_new = self.adapter_conv3(h3_input)
        h3_refined = self.alpha * h3 + (1 - self.alpha) * h3_new

        # Refine h4
        h4_input = torch.cat([h4, z], dim=1)  # (N,256+64)
        h4_new = self.adapter_fc1(h4_input)
        h4_refined = self.alpha * h4 + (1 - self.alpha) * h4_new

        # Recompute output
        y_new = self.fc2(h4_refined)
        return h1_refined, h2_refined, h3_refined, h4_refined, y_new

    def forward(self, x):
        # Initial forward pass
        h1, h2, h3, h4, y = self.forward_once(x)
        # Iterative refinement
        for _ in range(self.T):
            h1, h2, h3, h4, y = self.refine_step(h1, h2, h3, h4, y)
        return y
