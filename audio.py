import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import ttest_ind
import scienceplots

plt.style.use(["science", "ieee", "no-latex"])


###############################################################################
# 1. Utility: PadOrTrim
###############################################################################
class PadOrTrim(nn.Module):
    """
    Pads (with zeros) or trims the waveform so it has exactly `max_len` samples.
    This ensures consistent input lengths.
    """
    def __init__(self, max_len=16000):
        super().__init__()
        self.max_len = max_len

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform shape = (channels, num_samples)
        num_samples = waveform.size(-1)
        if num_samples < self.max_len:
            # Pad on the end
            pad_len = self.max_len - num_samples
            waveform = F.pad(waveform, (0, pad_len))
        elif num_samples > self.max_len:
            # Trim from the end
            waveform = waveform[..., :self.max_len]
        return waveform


###############################################################################
# 2. Dataset: SpeechCommands with consistent shapes
###############################################################################
class SubsetSC(Dataset):
    """
    A helper dataset for SpeechCommands, selecting 'training', 'validation', or 'testing'.
    We do:
      (1) Resample to 16kHz (if needed)
      (2) Pad/Trim to 1 second = 16,000 samples
      (3) Convert to 64-bin MelSpectrogram (output shape => (1,64,time))
    """
    def __init__(self, subset: str = "training"):
        super().__init__()
        self.data = SPEECHCOMMANDS(root=".", download=True, subset=subset)

        # Collect all unique labels
        label_set = set()
        for waveform, sample_rate, label, _, _ in self.data:
            label_set.add(label)
        labels = sorted(list(label_set))
        self.label_dict = {lab: idx for idx, lab in enumerate(labels)}

        # We'll chain transforms:
        #   - PadOrTrim => ensures exactly 16000 samples
        #   - MelSpectrogram => 64-bin
        self.transform = nn.Sequential(
            PadOrTrim(max_len=16000),                       # 1 second at 16k
            T.MelSpectrogram(sample_rate=16000, n_mels=64)  # => (1,64,time)
        )

    def __getitem__(self, idx):
        waveform, sample_rate, label, _, _ = self.data[idx]

        # If sample_rate != 16k, resample first
        if sample_rate != 16000:
            resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample(waveform)

        # Now pad/trim + mel-spec
        mel_spec = self.transform(waveform)
        label_idx = self.label_dict[label]
        return mel_spec, label_idx

    def __len__(self):
        return len(self.data)


###############################################################################
# 3. Standard CNN
###############################################################################
class StandardCNN(nn.Module):
    """
    A simple CNN for (B,1,64,time) MelSpectrogram inputs.
    We do:
      - 3 conv layers, each with ReLU+pool
      - flatten
      - 2 FC layers (256->num_classes)
    """
    def __init__(self, num_classes=35):
        super().__init__()
        # conv1 in_channels=1, out_channels=32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d((2,2))

        # after 3 pool steps along freq=64 => 64/2/2/2=8, time dimension => time/8
        # we want a consistent shape, so let's do an adaptive pool to (8,8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8,8))

        # final => 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (B,1,64,time)
        x = F.relu(self.conv1(x))  # => (B,32,64,time)
        x = self.pool1(x)         # => (B,32,32,time/2)
        x = F.relu(self.conv2(x)) # => (B,64,32,time/2)
        x = self.pool2(x)         # => (B,64,16,time/4)
        x = F.relu(self.conv3(x)) # => (B,128,16,time/4)
        x = self.pool3(x)         # => (B,128,8,time/8)

        # adaptively pool time => 8 => final shape => (B,128,8,8)
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1) # => (B,8192)
        x = F.relu(self.fc1(x))   # => (B,256)
        out = self.fc2(x)         # => (B,num_classes)
        return out


###############################################################################
# 4. CBL CNN
###############################################################################
class CBL_CNN(nn.Module):
    """
    Same base CNN, but includes CBL:
      - track states h1, h2, h3, h4
      - refine them with top-down context from y
      - T refinement steps, alpha blend
    """
    def __init__(self, num_classes=35, T=2, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.z_dim = 64

        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d((2,2))
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d((2,2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8,8))

        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # context mapping
        self.g = nn.Linear(num_classes, self.z_dim)

        # adapters
        self.adapter_conv1 = nn.Sequential(
            nn.Conv2d(32+1, 32, kernel_size=1),
            nn.ReLU()
        )
        self.adapter_conv2 = nn.Sequential(
            nn.Conv2d(64+1, 64, kernel_size=1),
            nn.ReLU()
        )
        self.adapter_conv3 = nn.Sequential(
            nn.Conv2d(128+1, 128, kernel_size=1),
            nn.ReLU()
        )
        self.adapter_fc1 = nn.Sequential(
            nn.Linear(256 + self.z_dim, 256),
            nn.ReLU()
        )

    def forward_once(self, x):
        """
        One forward pass to gather intermediate states h1,h2,h3,h4 => y
        """
        # x => (B,1,64,time)
        a1 = F.relu(self.conv1(x))     # => (B,32,64,time)
        h1 = self.pool1(a1)            # => (B,32,32,time/2)

        a2 = F.relu(self.conv2(h1))    # => (B,64,32,time/2)
        h2 = self.pool2(a2)            # => (B,64,16,time/4)

        a3 = F.relu(self.conv3(h2))    # => (B,128,16,time/4)
        h3 = self.pool3(a3)            # => (B,128,8,time/8)

        # adaptive pool => (B,128,8,8)
        h3_pool = self.adaptive_pool(h3)
        h3_flat = h3_pool.view(h3_pool.size(0), -1)  # => (B,8192)

        h4 = F.relu(self.fc1(h3_flat))  # => (B,256)
        y = self.fc2(h4)                # => (B,num_classes)
        return h1, h2, h3, h4, y

    def refine_step(self, h1, h2, h3, h4, y):
        """
        Given states + y => compute context => refine => produce new states, new y
        """
        z = self.g(y)  # => (B,z_dim)
        z_scalar = z.mean(dim=1, keepdim=True)  # => (B,1)

        # Refine h1
        z_sc1 = z_scalar.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h1.size(2), h1.size(3))
        h1_in = torch.cat([h1, z_sc1], dim=1)
        h1_new = self.adapter_conv1(h1_in)
        h1_refined = self.alpha*h1 + (1-self.alpha)*h1_new

        # Refine h2
        z_sc2 = z_scalar.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h2.size(2), h2.size(3))
        h2_in = torch.cat([h2, z_sc2], dim=1)
        h2_new = self.adapter_conv2(h2_in)
        h2_refined = self.alpha*h2 + (1-self.alpha)*h2_new

        # Refine h3
        z_sc3 = z_scalar.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h3.size(2), h3.size(3))
        h3_in = torch.cat([h3, z_sc3], dim=1)
        h3_new = self.adapter_conv3(h3_in)
        h3_refined = self.alpha*h3 + (1-self.alpha)*h3_new

        # Refine h4
        h4_in = torch.cat([h4, z], dim=1)  # (B,256+64)
        h4_new = self.adapter_fc1(h4_in)   # => (B,256)
        h4_refined = self.alpha*h4 + (1-self.alpha)*h4_new

        # re-output
        y_new = self.fc2(h4_refined)
        return h1_refined, h2_refined, h3_refined, h4_refined, y_new

    def forward(self, x):
        # initial pass
        h1, h2, h3, h4, y = self.forward_once(x)
        # T loops
        for _ in range(self.T):
            h1, h2, h3, h4, y = self.refine_step(h1, h2, h3, h4, y)
        return y


###############################################################################
# 5. Train & Evaluate
###############################################################################
def train_one_epoch(model, loader, optimizer, criterion, device="cpu"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)
    for data, target in loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, preds = out.max(dim=1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)
        loop.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return running_loss/total, correct/total

def evaluate(model, loader, criterion, device="cpu"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion(out, target)

            running_loss += loss.item() * data.size(0)
            _, preds = out.max(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

    return running_loss/total, correct/total


###############################################################################
# 6. Main Experiment
###############################################################################
def main_experiment(num_epochs=5, device="cpu"):
    # 1) Prepare datasets & dataloaders
    train_data = SubsetSC("training")
    valid_data = SubsetSC("validation")
    test_data  = SubsetSC("testing")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

    num_classes = len(train_data.label_dict)
    print(f"Number of classes: {num_classes}")

    # 2) Models
    model_std = StandardCNN(num_classes=num_classes).to(device)
    model_cbl = CBL_CNN(num_classes=num_classes, T=2, alpha=0.025).to(device)

    # 3) Optimizers & Criterion
    optimizer_std = torch.optim.Adam(model_std.parameters(), lr=1e-3)
    optimizer_cbl = torch.optim.Adam(model_cbl.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 4) Loop
    results = {
        "epoch": [],
        "loss_std": [],
        "acc_std": [],
        "loss_cbl": [],
        "acc_cbl": []
    }

    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Training
        train_loss_std, train_acc_std = train_one_epoch(
            model_std, train_loader, optimizer_std, criterion, device
        )
        train_loss_cbl, train_acc_cbl = train_one_epoch(
            model_cbl, train_loader, optimizer_cbl, criterion, device
        )

        # Validation
        val_loss_std, val_acc_std = evaluate(model_std, valid_loader, criterion, device)
        val_loss_cbl, val_acc_cbl = evaluate(model_cbl, valid_loader, criterion, device)

        print(f"  [Validation] StandardCNN Loss: {val_loss_std:.4f}, Acc: {val_acc_std:.4f} | "
              f"CBL_CNN Loss: {val_loss_cbl:.4f}, Acc: {val_acc_cbl:.4f}")

        # Record
        results["epoch"].append(epoch)
        results["loss_std"].append(val_loss_std)
        results["acc_std"].append(val_acc_std)
        results["loss_cbl"].append(val_loss_cbl)
        results["acc_cbl"].append(val_acc_cbl)

    # Convert to DataFrame
    df_loss = pd.DataFrame({
        "Epoch": results["epoch"] * 2,
        "Loss": results["loss_std"] + results["loss_cbl"],
        "Model": ["Standard"]*len(results["epoch"]) + ["CBL"]*len(results["epoch"])
    })
    df_acc = pd.DataFrame({
        "Epoch": results["epoch"] * 2,
        "Accuracy": results["acc_std"] + results["acc_cbl"],
        "Model": ["Standard"]*len(results["epoch"]) + ["CBL"]*len(results["epoch"])
    })

    # 5) Final test evaluation
    test_loss_std, test_acc_std = evaluate(model_std, test_loader, criterion, device)
    test_loss_cbl, test_acc_cbl = evaluate(model_cbl, test_loader, criterion, device)

    print("\nFinal Test Results:")
    print(f"  StandardCNN - Loss: {test_loss_std:.4f}, Acc: {test_acc_std:.4f}")
    print(f"  CBL_CNN     - Loss: {test_loss_cbl:.4f}, Acc: {test_acc_cbl:.4f}")

    # 6) Plot
    plt.figure(figsize=(6,4))
    sns.lineplot(data=df_loss, x="Epoch", y="Loss", hue="Model", marker="o")
    plt.title("Validation Loss Over Epochs")
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    sns.lineplot(data=df_acc, x="Epoch", y="Accuracy", hue="Model", marker="o")
    plt.title("Validation Accuracy Over Epochs")
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=300)
    plt.close()

    # 7) T-test on validation accuracy
    std_acc_vals = df_acc[df_acc["Model"]=="Standard"]["Accuracy"].values
    cbl_acc_vals = df_acc[df_acc["Model"]=="CBL"]["Accuracy"].values
    t_stat, p_val = ttest_ind(std_acc_vals, cbl_acc_vals, equal_var=False)
    print(f"\nT-test on validation accuracy:\n  t-stat: {t_stat:.4f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("CBL CNN shows a statistically significant difference compared to Standard CNN at alpha=0.05.")
    else:
        print("No statistically significant difference at alpha=0.05.")


###############################################################################
# 7. Entry Point
###############################################################################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main_experiment(num_epochs=10, device=device)
