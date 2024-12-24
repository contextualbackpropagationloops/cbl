import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import os

from standard_cnn import StandardCNN
from cbl_model import CBL_CNN

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Optional nicer style (scienceplots)
import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])

# For statistical tests
import random
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

###############################################################################
# Training and Evaluation Routines
###############################################################################
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

###############################################################################
# Single-Experiment Routine
###############################################################################
def run_single_experiment(args, run_id=1):
    """
    Runs one experiment:
      - Loads CIFAR-10
      - Splits into train/val/test
      - Trains StandardCNN and CBL_CNN
      - Returns final test accuracies for each
      - Also generates and saves plots (only recommended for the final run)
    """
    device = args.device
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    print(f"\n=== Run {run_id}: Training Standard CNN and CBL CNN (T={args.T} ===")

    # To ensure reproducibility for this run (if you vary seeds between runs)
    seed_for_this_run = 1234 + run_id  # or any scheme you like
    random.seed(seed_for_this_run)
    np.random.seed(seed_for_this_run)
    torch.manual_seed(seed_for_this_run)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_for_this_run)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])

    # Download / load datasets
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create train/val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    ############################################################################
    # 1) Train Standard CNN
    ############################################################################
    print("Training Standard CNN...")
    standard_model = StandardCNN(num_classes=10).to(device)
    optimizer_std = optim.Adam(standard_model.parameters(), lr=lr)

    std_train_losses = []
    std_train_accs = []
    std_val_losses = []
    std_val_accs = []

    for epoch in range(epochs):
        print(f"[StandardCNN] Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(standard_model, train_loader, optimizer_std, criterion, device)
        val_loss, val_acc = evaluate(standard_model, val_loader, criterion, device)
        std_train_losses.append(train_loss)
        std_train_accs.append(train_acc)
        std_val_losses.append(val_loss)
        std_val_accs.append(val_acc)
        print(f"Standard CNN - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    test_loss, test_acc = evaluate(standard_model, test_loader, criterion, device)
    print(f"[StandardCNN] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    ############################################################################
    # 2) Train CBL CNN
    ############################################################################
    print("\nTraining CBL CNN...")
    cbl_model = CBL_CNN(num_classes=10, T=args.T).to(device)
    optimizer_cbl = optim.Adam(cbl_model.parameters(), lr=lr)

    cbl_train_losses = []
    cbl_train_accs = []
    cbl_val_losses = []
    cbl_val_accs = []

    for epoch in range(epochs):
        print(f"[CBL_CNN] Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(cbl_model, train_loader, optimizer_cbl, criterion, device)
        val_loss, val_acc = evaluate(cbl_model, val_loader, criterion, device)
        cbl_train_losses.append(train_loss)
        cbl_train_accs.append(train_acc)
        cbl_val_losses.append(val_loss)
        cbl_val_accs.append(val_acc)
        print(f"CBL CNN - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    test_loss_cbl, test_acc_cbl = evaluate(cbl_model, test_loader, criterion, device)
    print(f"[CBL_CNN] Test Loss: {test_loss_cbl:.4f}, Test Acc: {test_acc_cbl:.2f}%")

    ############################################################################
    # 3) Plotting (only do detailed plotting if run_id=1 or you want every run)
    ############################################################################
    if run_id == 1:
        epochs_range = range(1, epochs+1)

        import matplotlib as mpl
        mpl.rcParams['lines.markersize'] = 5 # Increase marker size
        
        # Prepare data for plotting
        # Accuracy DataFrame
        df_acc = pd.DataFrame({
            'Epoch': list(epochs_range)*2,
            'Accuracy': std_val_accs + cbl_val_accs,
            'Model': ['Standard']*epochs + ['CBL']*epochs
        })

        # Loss DataFrame
        df_loss = pd.DataFrame({
            'Epoch': list(epochs_range)*2,
            'Loss': std_val_losses + cbl_val_losses,
            'Model': ['Standard']*epochs + ['CBL']*epochs
        })

        sns.set_style("whitegrid")

        # Plot Validation Accuracy
        plt.figure(figsize=(6,4))
        sns.lineplot(data=df_acc, x='Epoch', y='Accuracy', hue='Model', marker='o')
        plt.title('Validation Accuracy Comparison (CIFAR-10)')
        plt.savefig('plots/validation_accuracy_comparison_cifar10.png')
        plt.close()

        # Plot Validation Loss
        plt.figure(figsize=(6,4))
        sns.lineplot(data=df_loss, x='Epoch', y='Loss', hue='Model', marker='o')
        plt.title('Validation Loss Comparison (CIFAR-10)')
        plt.savefig('plots/validation_loss_comparison_cifar10.png')
        plt.close()

        # Plot Training vs Validation Accuracy for each model separately
        df_std_acc_all = pd.DataFrame({
            'Epoch': epochs_range,
            'Accuracy': std_train_accs,
            'Dataset': ['Train']*epochs
        })
        df_std_acc_all = pd.concat([df_std_acc_all, pd.DataFrame({
            'Epoch': epochs_range,
            'Accuracy': std_val_accs,
            'Dataset': ['Val']*epochs
        })])

        df_cbl_acc_all = pd.DataFrame({
            'Epoch': epochs_range,
            'Accuracy': cbl_train_accs,
            'Dataset': ['Train']*epochs
        })
        df_cbl_acc_all = pd.concat([df_cbl_acc_all, pd.DataFrame({
            'Epoch': epochs_range,
            'Accuracy': cbl_val_accs,
            'Dataset': ['Val']*epochs
        })])
        

        # Standard CNN Accuracy (Train vs Val)
        plt.figure(figsize=(10,6))
        sns.lineplot(data=df_std_acc_all, x='Epoch', y='Accuracy', hue='Dataset', marker='o')
        plt.title('Standard CNN Accuracy (Train vs Val)')
        plt.savefig('plots/standard_cnn_accuracy_train_val_cifar10.png')
        plt.close()

        # CBL CNN Accuracy (Train vs Val)
        plt.figure(figsize=(10,6))
        sns.lineplot(data=df_cbl_acc_all, x='Epoch', y='Accuracy', hue='Dataset', marker='o')
        plt.title('CBL CNN Accuracy (Train vs Val)')
        plt.savefig('plots/cbl_cnn_accuracy_train_val_cifar10.png')
        plt.close()

    return test_acc, test_acc_cbl


###############################################################################
# Main Function: Repeated Runs + Statistical Tests
###############################################################################
def main():
    """
    - Parses arguments (including num_runs).
    - If num_runs=1, just run once (original behavior).
    - If num_runs>1, run multiple times, gather final test accuracies, 
      and then do paired t-test and Wilcoxon signed-rank test.
    """
    parser = argparse.ArgumentParser(description="Train and test standard and CBL CNN on CIFAR-10")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=5,
                        help="Number of repeated runs for statistical significance tests.")
    args = parser.parse_args()

    # If we only want 1 run, do the original single experiment.
    if args.num_runs == 1:
        print("=== Single run mode ===")
        run_single_experiment(args, run_id=1)
        return

    # Otherwise, do multiple runs and collect final test accuracies
    print(f"=== Multiple runs mode (num_runs={args.num_runs}) ===")
    standard_accuracies = []
    cbl_accuracies = []

    for run_id in range(1, args.num_runs + 1):
        std_acc, cbl_acc = run_single_experiment(args, run_id=run_id)
        standard_accuracies.append(std_acc)
        cbl_accuracies.append(cbl_acc)
        print(f"[Run {run_id}] StandardCNN Test Acc: {std_acc:.2f}% | CBL_CNN Test Acc: {cbl_acc:.2f}%")

    # Convert to numpy arrays for statistical tests
    A_std = np.array(standard_accuracies)
    A_cbl = np.array(cbl_accuracies)

    print("\n=== Summary of final test accuracies across runs ===")
    print("StandardCNN:", A_std)
    print("CBL_CNN:    ", A_cbl)
    print(f"\nStandardCNN => mean={A_std.mean():.2f}%, std={A_std.std():.2f}%")
    print(f"CBL_CNN     => mean={A_cbl.mean():.2f}%, std={A_cbl.std():.2f}%")

    # 1) Paired t-test
    t_stat, p_val_t = ttest_rel(A_cbl, A_std)
    print(f"\nPaired t-test:\n  t-stat={t_stat:.3f}, p-value={p_val_t:.4f}")
    if p_val_t < 0.05:
        print("  => Reject H0: CBL is significantly different (often better) than StandardCNN.")
    else:
        print("  => Fail to reject H0: No significant difference at p<0.05.")

    # 2) Wilcoxon signed-rank test
    stat_w, p_val_w = wilcoxon(A_cbl, A_std, alternative='two-sided')
    print(f"\nWilcoxon signed-rank test:\n  stat={stat_w:.3f}, p-value={p_val_w:.4f}")
    if p_val_w < 0.05:
        print("  => Reject H0: CBL is significantly different (often better) than StandardCNN.")
    else:
        print("  => Fail to reject H0: No significant difference at p<0.05.")

    # Interpretation notes
    print("\nINTERPRETATION:")
    print("1) Paired t-test assumes differences in accuracies are normally distributed.")
    print("2) Wilcoxon is a non-parametric alternative, safer for small samples or non-normal data.")
    print("3) If either p-value < 0.05, we typically conclude that there's a statistically")
    print("   significant difference in final test accuracies between CBL and StandardCNN.")


if __name__ == "__main__":
    main()
