import argparse
import torch
import torch.nn as nn
import timm
from thop import profile

class CBL_Transformer(nn.Module):
    """
    A Vision Transformer (ViT-B/16) that implements Contextual Backpropagation Loops (CBL).
    In each iteration (T steps), we derive a context vector z from the current output y
    and feed it back into the hidden states for refinement.
    """
    def __init__(self, 
                 model_name='vit_base_patch16_224',
                 pretrained=False,
                 num_classes=1000,
                 T=2,
                 alpha=0.0):
        """
        T: number of iterative refinement steps.
        alpha: blending factor for refinement.
        """
        super(CBL_Transformer, self).__init__()

        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes
        self.T = T
        self.alpha = alpha

        # Dimension of the ViT hidden representation. For ViT-B/16, typically 768.
        self.hidden_dim = self.vit.embed_dim

        # Dimension of the context vector z
        self.z_dim = 128

        # Projection layer that converts the output logits to a context vector z
        self.g = nn.Linear(num_classes, self.z_dim)

        # Adapter for each Transformer block
        self.adapter_blocks = nn.ModuleList([
            nn.Linear(self.hidden_dim + self.z_dim, self.hidden_dim)
            for _ in range(len(self.vit.blocks))
        ])

    def forward_once(self, x):
        """
        Single forward pass through ViT, storing intermediate hidden states.
        """
        # Patch embedding
        x = self.vit.patch_embed(x)  # (B, N_patches, hidden_dim)

        # Class token
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)

        # If model has dist_token (like DeiT), handle it; otherwise skip
        dist_token = getattr(self.vit, 'dist_token', None)
        if dist_token is not None:
            dist_token = dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding & dropout
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        # Pass through each Transformer block, storing hidden states
        hidden_states = []
        for block in self.vit.blocks:
            x = block(x)
            hidden_states.append(x)

        # Final norm + head
        x = self.vit.norm(x)
        y = self.vit.head(x[:, 0])  # (batch_size, num_classes)
        return hidden_states, y

    def refine_step(self, hidden_states, y):
        """
        One refinement step: 
          1) compute context z from y
          2) re-inject z into each hidden state
          3) re-run norm+head
        """
        # Derive context vector
        z = self.g(y)  # (batch_size, z_dim)

        new_hidden_states = []
        for i, h in enumerate(hidden_states):
            # h shape: (B, N, hidden_dim)
            # Expand z to match (B, N, z_dim)
            z_expanded = z.unsqueeze(1).expand(-1, h.size(1), -1)
            # Concat
            h_input = torch.cat([h, z_expanded], dim=-1)  # (B, N, hidden_dim + z_dim)
            # Pass through adapter
            h_new = self.adapter_blocks[i](h_input)
            # Blend old and new
            h_refined = self.alpha * h + (1 - self.alpha) * h_new
            new_hidden_states.append(h_refined)

        # Recompute final output from last hidden state
        x = self.vit.norm(new_hidden_states[-1])
        y_new = self.vit.head(x[:, 0])
        return new_hidden_states, y_new

    def forward(self, x):
        """
        Full forward with iterative refinement (T steps).
        """
        hidden_states, y = self.forward_once(x)
        for _ in range(self.T):
            hidden_states, y = self.refine_step(hidden_states, y)
        return y

