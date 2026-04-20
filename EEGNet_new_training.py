# # ============================================================================
# # Improved EEGNet Training Script
# # ============================================================================
# # Enhanced training with:
# # - Windowing strategy (creates more training samples)
# # - Learning rate scheduling
# # - Better logging (CSV + JSON)
# # - Early stopping
# # - Model checkpointing
# # - Works with preprocessed .npy files
# # ============================================================================

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training_artifacts import build_run_paths, save_classification_outputs, save_confusion_matrix_figure

PROJECT_ROOT = Path(__file__).resolve().parent

# Force flush output (Windows fix)
sys.stdout.flush()

# Import improved EEGNet model
from EEGNet_new_model import EEGNet


# ============================================================================
# DATASET CLASS WITH WINDOWING
# ============================================================================
class EEGDatasetWithWindowing(Dataset):
    """
    Dataset class that supports windowing strategy.
    Creates overlapping windows from longer trials to increase dataset size.
    """
    
    def __init__(self, X, y, window_size=500, step_size=250, augment=False):
        """
        Args:
            X: EEG data of shape (N, C, T) or (N, 1, C, T)
            y: Labels of shape (N,)
            window_size: Size of each window in samples (default: 250 = 1s at 250 Hz)
            step_size: Stride between windows (default: 75 = 0.3s, gives 70% overlap)
            augment: Whether to apply data augmentation (not implemented yet)
        """
        self.augment = augment
        
        # Handle different input shapes
        if X.ndim == 4:
            # (N, 1, C, T) -> (N, C, T)
            if X.shape[1] == 1:
                X = X.squeeze(1)
        
        # Create windows from each trial
        windows = []
        labels = []
        
        for i in range(len(X)):
            trial = X[i]  # (C, T)
            label = y[i]
            
            # Extract overlapping windows
            n_channels, n_timepoints = trial.shape
            
            for start in range(0, n_timepoints - window_size + 1, step_size):
                end = start + window_size
                window = trial[:, start:end]  # (C, window_size)
                windows.append(window)
                labels.append(label)
        
        self.X = np.array(windows)  # (N_windows, C, window_size)
        self.y = np.array(labels)
        
        print(f"Created {len(self.X)} windows from {len(X)} trials")
        print(f"  Window size: {window_size} samples ({window_size/250:.1f}s)")
        print(f"  Step size: {step_size} samples ({step_size/250:.1f}s)")
        print(f"  Overlap: {(1 - step_size/window_size)*100:.1f}%")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])  # (C, window_size)
        y = torch.LongTensor([self.y[idx]])[0]
        
        # Data augmentation (optional)
        if self.augment and np.random.rand() > 0.5:
            # Add small Gaussian noise
            noise = torch.randn_like(x) * 0.01 * x.std()
            x = x + noise
        
        return x, y


# ============================================================================
# EARLY STOPPING CLASS
# ============================================================================
class EarlyStopping:
    """
    Early stopping to stop training when validation loss/metric stops improving.
    """
    
    def __init__(self, monitor='val_loss', mode='min', patience=50, min_delta=0.0):
        """
        Args:
            monitor: Metric to monitor ('val_loss' or 'val_acc')
            mode: 'min' for loss, 'max' for accuracy
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_state_dict = None
        self.best_metrics = {}
        self.early_stop = False
    
    def step(self, metrics, model, epoch):
        """
        Check if we should stop training.
        
        Args:
            metrics: Dict with 'val_loss' and 'val_acc'
            model: PyTorch model
            epoch: Current epoch number
        
        Returns:
            improved: True if metric improved
        """
        if self.monitor == 'val_loss':
            current_score = metrics['val_loss']
            is_better = current_score < (self.best_score - self.min_delta) if self.best_score else True
        else:  # val_acc
            current_score = metrics['val_acc']
            is_better = current_score > (self.best_score + self.min_delta) if self.best_score else True
        
        if self.best_score is None or is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_state_dict = model.state_dict().copy()
            self.best_metrics = metrics.copy()
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    total = 0
    preds_all, labels_all = [], []
    
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)  # (B, C, T)
        y = y.to(device).long()
        
        optimizer.zero_grad()
        logits = model(x)  # (B, n_classes)
        loss = loss_fn(logits, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(y.cpu())
        total += x.size(0)
    
    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(labels_all, preds_all)
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    total = 0
    preds_all, labels_all = [], []
    
    for x, y in tqdm(dataloader, desc="Validating", leave=False):
        x = x.to(device)
        y = y.to(device).long()
        logits = model(x)
        loss = loss_fn(logits, y)
        
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(y.cpu())
        total += x.size(0)
    
    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(labels_all, preds_all)
    
    return epoch_loss, epoch_acc


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_eegnet(
    data_dir="",
    subject_id=None,
    use_windowing=True,
    window_size=250,      # 1 second at 250 Hz
    step_size=75,         # 0.3 seconds (70% overlap)
    batch_size=16,
    epochs=1500,
    lr=1e-4,
    weight_decay=1e-4,
    dropout_rate=0.5,
    F1=16,                # More filters
    patience=300,
    save_prefix="outputs/eegnet_new",
    seed=42
):
    """
    Main training function for improved EEGNet
    
    Args:
        data_dir: Directory containing .npy files
        use_windowing: Whether to use windowing strategy
        window_size: Size of each window in samples
        step_size: Stride between windows
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        dropout_rate: Dropout rate
        F1: Number of temporal filters
        patience: Early stopping patience
        save_prefix: Prefix for saving models and logs
        seed: Random seed
    """
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print("=" * 80)
    print("🚀 Improved EEGNet Training")
    print("=" * 80)
    
    # -----------------------------------------------
    # STEP 1 — LOAD DATA
    # -----------------------------------------------
    print("\n📂 Loading preprocessed EEG data...")
    
    X_train = np.load(os.path.join(data_dir, "X_train_eegnet.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train_eegnet.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_eegnet.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test_eegnet.npy"))
    
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Handle different input shapes
    if X_train.ndim == 4:
        if X_train.shape[1] == 1:
            X_train = X_train.squeeze(1)
            X_test = X_test.squeeze(1)
            print("   Reshaped from 4D to 3D: (N, 1, C, T) -> (N, C, T)")
    
    # Validate shape
    if X_train.ndim != 3:
        raise ValueError(f"Expected 3D array (N, C, T), got {X_train.ndim}D: {X_train.shape}")
    
    n_channels = X_train.shape[1]
    n_timepoints = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    results_paths = build_run_paths(
        PROJECT_ROOT,
        "EEGNet",
        data_dir,
        num_classes=num_classes,
        subject_no=subject_id,
    )
    save_prefix = str(results_paths["checkpoints_dir"] / "eegnet_new")
    figures_dir = results_paths["figures_dir"]
    metrics_dir = results_paths["metrics_dir"]
    run_dir = results_paths["run_dir"]
    
    print(f"✅ Data shape: (N, {n_channels}, {n_timepoints})")
    print(f"Results directory: {run_dir}")
    
    # -----------------------------------------------
    # STEP 2 — TRAIN/VAL SPLIT
    # -----------------------------------------------
    # Split training data into train and validation
    # (Test set is already separated in preprocessing)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=seed
    )
    
    print(f"\nDataset sizes:")
    print(f"Train: {X_train_split.shape[0]} trials")
    print(f"Val:   {X_val.shape[0]} trials")
    print(f"Test:  {X_test.shape[0]} trials")
    
    # -----------------------------------------------
    # STEP 3 — CREATE DATASETS WITH WINDOWING
    # -----------------------------------------------
    if use_windowing:
        print(f"\n📊 Using windowing strategy:")
        print(f"   Window size: {window_size} samples ({window_size/250:.1f}s)")
        print(f"   Step size: {step_size} samples ({step_size/250:.1f}s)")
        
        train_dataset = EEGDatasetWithWindowing(
            X_train_split, y_train_split,
            window_size=window_size,
            step_size=step_size,
            augment=False
        )
        val_dataset = EEGDatasetWithWindowing(
            X_val, y_val,
            window_size=window_size,
            step_size=window_size,  # No overlap for validation
            augment=False
        )
        test_dataset = EEGDatasetWithWindowing(
            X_test, y_test,
            window_size=window_size,
            step_size=window_size,  # No overlap for test
            augment=False
        )
        
        # Update n_timepoints to window_size
        n_timepoints = window_size
    else:
        print("\n📊 Using full trials (no windowing)")
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_split),
            torch.LongTensor(y_train_split)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
    
    # -----------------------------------------------
    # STEP 4 — DATA LOADERS
    # -----------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows safe
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # -----------------------------------------------
    # STEP 5 — BUILD MODEL
    # -----------------------------------------------
    print("\n🏗️ Building improved EEGNet model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = EEGNet(
        num_classes=num_classes,
        num_channels=n_channels,
        num_samples=n_timepoints,
        fs=250,
        dropout_rate=dropout_rate,
        F1=F1,
        D=2,
        F2=None,  # Auto = F1
        k_temporal=None,  # Auto = 0.5s = 125 samples
        use_spatial_dropout=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   F1 (temporal filters): {F1}")
    print(f"   Temporal kernel: {model.k_temporal} samples ({model.k_temporal/250:.2f}s)")
    print(f"   Spatial dropout: Enabled")
    
    # -----------------------------------------------
    # STEP 6 — OPTIMIZER + SCHEDULER
    # -----------------------------------------------
    # Class weights for imbalanced data
    train_counts = Counter(y_train_split)
    total = sum(train_counts.values())
    class_weights = torch.FloatTensor([
        total / (len(train_counts) * train_counts[i]) for i in sorted(train_counts.keys())
    ]).to(device)
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        min_lr=1e-6,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
        min_delta=0.0001
    )
    
    # -----------------------------------------------
    # STEP 7 — TRAINING LOOP
    # -----------------------------------------------
    print("\n🏁 Starting training...\n")
    
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }
    
    # Create output directory
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    best_path = f"{save_prefix}_best.pth"
    last_path = f"{save_prefix}_last.pth"
    log_path = f"{save_prefix}_log.csv"
    label_map_path = f"{save_prefix}_label_map.json"
    
    # Save label map
    label_map = {int(i): f"Class {i}" for i in sorted(np.unique(y_train))}
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"📄 Label map saved: {label_map_path}")
    
    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", leave=True)
    
    for epoch in epoch_bar:
        # Train
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(va_loss)
        
        # Save history
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Early stopping check
        improved = early_stopping.step(
            {"val_loss": va_loss, "val_acc": va_acc},
            model,
            epoch
        )
        
        if improved:
            torch.save(model.state_dict(), best_path)
        
        # Progress bar
        postfix_dict = {
            "Train Loss": f"{tr_loss:.4f}",
            "Train Acc": f"{tr_acc:.4f}",
            "Val Loss": f"{va_loss:.4f}",
            "Val Acc": f"{va_acc:.4f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.6f}",
        }
        
        if early_stopping.best_metrics:
            postfix_dict["Best"] = f"Loss={early_stopping.best_metrics['val_loss']:.4f} @ E{early_stopping.best_epoch}"
            postfix_dict["Status"] = (
                f"✅ Improved" if improved
                else f"⚠️ No improvement ({early_stopping.counter}/{patience})"
            )
        
        epoch_bar.set_postfix_str(str(postfix_dict))
        
        if early_stopping.early_stop:
            print(f"\n⛔ Early stopping at epoch {epoch}")
            break
    
    # Save final model and restore best
    torch.save(model.state_dict(), last_path)
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
        torch.save(model.state_dict(), best_path)
        print("🔁 Restored best model weights.")
    
    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(log_path, index=False)
    print(f"📊 Training history saved: {log_path}")
    
    # Summary
    if early_stopping.best_metrics:
        best_loss = early_stopping.best_metrics["val_loss"]
        best_acc = early_stopping.best_metrics["val_acc"]
        print(f"\n🏆 Best Validation Loss: {early_stopping.best_score:.6f} @ epoch {early_stopping.best_epoch}")
        print(f"   📉 Val Loss: {best_loss:.6f}")
        print(f"   📊 Val Acc:  {best_acc:.4f}")
    
    # -----------------------------------------------
    # STEP 8 — TEST EVALUATION
    # -----------------------------------------------
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)
    
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())
            
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
    
    test_acc = 100 * test_correct / test_total
    
    print(f"\n✅ Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
    
    # Classification report
    class_labels = [f"Class {i}" for i in range(num_classes)]
    print("\nClassification Report:")
    report_text = classification_report(all_true, all_preds, target_names=class_labels, zero_division=0)
    print(report_text)
    save_classification_outputs(metrics_dir, all_true, all_preds, class_labels, "eegnet_new")
    
    # -----------------------------------------------
    # STEP 9 — PLOTS AND VISUALIZATIONS
    # -----------------------------------------------
    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0, 0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history["train_acc"], label="Train Acc", linewidth=2)
    axes[0, 1].plot(history["val_acc"], label="Val Acc", linewidth=2)
    axes[0, 1].axhline(
        100 / num_classes,
        color='r',
        linestyle='--',
        label=f'Random ({100/num_classes:.1f}%)',
        alpha=0.5
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training & Validation Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history["lr"], linewidth=2, color='green')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting gap
    gap = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    axes[1, 1].plot(gap, linewidth=2, color='red')
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Train - Val Accuracy")
    axes[1, 1].set_title("Overfitting Monitor")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "eegnet_new_training_curves.png", dpi=300)
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"EEGNet Improved - Test Accuracy: {test_acc:.2f}%", fontsize=14)
    plt.colorbar()
    
    plt.xticks(range(num_classes), class_labels, rotation=45)
    plt.yticks(range(num_classes), class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Print values inside matrix
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12
            )
    
    plt.tight_layout()
    plt.savefig(figures_dir / "eegnet_new_confusion_matrix.png", dpi=300)
    plt.show()

    save_confusion_matrix_figure(
        figures_dir / "eegnet_new_confusion_matrix_clean.png",
        cm,
        class_labels,
        f"EEGNet Improved - Test Accuracy: {test_acc:.2f}%",
    )
    
    print(f"\n✅ All visualizations saved to {figures_dir}")
    print(f"✅ Best model saved: {best_path}")
    print(f"✅ Last model saved: {last_path}")
    print(f"✅ Metrics saved to: {metrics_dir}")
    
    return model, hist_df, best_path, test_acc


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved EEGNet with subject-aware outputs.")
    parser.add_argument("--data-dir", default="", help="Directory containing X_train_eegnet.npy and related files")
    parser.add_argument("--subject-id", type=int, default=None, help="Optional subject id used for results folder naming")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.7)
    parser.add_argument("--f1", type=int, default=16)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = {
        "data_dir": args.data_dir,
        "subject_id": args.subject_id,
        "use_windowing": True,
        "window_size": 250,
        "step_size": 75,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout_rate": args.dropout_rate,
        "F1": args.f1,
        "patience": args.patience,
        "seed": args.seed,
    }
    
    print("=" * 80)
    print("Improved EEGNet Training Configuration")
    print("=" * 80)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Run training
    model, history, best_path, test_acc = train_eegnet(**config)
    
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"Best Model: {best_path}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=" * 80)

# ============================================================================
# Improved EEGNet Training Script - Updated with Stable LR Logic
# ============================================================================

# ============================================================================
# Improved EEGNet Training Script - Fix for Older PyTorch Versions
# ============================================================================

# ============================================================================
# Improved EEGNet Training Script - Fix for Runtime Dimension Mismatch
# ============================================================================

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import os
# import sys
# import json
# import pandas as pd
# from tqdm import tqdm
# from collections import Counter

# # Force flush output (Windows fix)
# sys.stdout.flush()

# from EEGNet_new_model import EEGNet

# # ============================================================================
# # DATASET CLASS WITH WINDOWING
# # ============================================================================
# class EEGDatasetWithWindowing(Dataset):
#     def __init__(self, X, y, window_size=250, step_size=75, augment=False):
#         self.augment = augment
#         if X.ndim == 4:
#             if X.shape[1] == 1:
#                 X = X.squeeze(1)
        
#         windows = []
#         labels = []
#         for i in range(len(X)):
#             trial = X[i]
#             label = y[i]
#             n_channels, n_timepoints = trial.shape
#             for start in range(0, n_timepoints - window_size + 1, step_size):
#                 end = start + window_size
#                 window = trial[:, start:end]
#                 windows.append(window)
#                 labels.append(label)
        
#         self.X = np.array(windows)
#         self.y = np.array(labels)
#         print(f"Created {len(self.X)} windows from {len(X)} trials")

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         x = torch.FloatTensor(self.X[idx])
#         y = torch.LongTensor([self.y[idx]])[0]
#         if self.augment and np.random.rand() > 0.5:
#             noise = torch.randn_like(x) * 0.01 * x.std()
#             x = x + noise
#         return x, y

# # ============================================================================
# # EARLY STOPPING CLASS
# # ============================================================================
# class EarlyStopping:
#     def __init__(self, monitor='val_loss', mode='min', patience=50, min_delta=0.0):
#         self.monitor = monitor
#         self.mode = mode
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_score = None
#         self.best_epoch = 0
#         self.best_state_dict = None
#         self.best_metrics = {}
#         self.early_stop = False

#     def step(self, metrics, model, epoch):
#         if self.monitor == 'val_loss':
#             current_score = metrics['val_loss']
#             is_better = current_score < (self.best_score - self.min_delta) if self.best_score else True
#         else:
#             current_score = metrics['val_acc']
#             is_better = current_score > (self.best_score + self.min_delta) if self.best_score else True
        
#         if self.best_score is None or is_better:
#             self.best_score = current_score
#             self.best_epoch = epoch
#             self.best_state_dict = model.state_dict().copy()
#             self.best_metrics = metrics.copy()
#             self.counter = 0
#             return True
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#             return False

# # ============================================================================
# # TRAINING FUNCTIONS
# # ============================================================================
# def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scheduler=None):
#     model.train()
#     running_loss = 0.0
#     total = 0
#     preds_all, labels_all = [], []
    
#     for x, y in tqdm(dataloader, desc="Training", leave=False):
#         x = x.to(device)
#         y = y.to(device).long()
        
#         optimizer.zero_grad()
#         logits = model(x)
#         loss = loss_fn(logits, y)
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         with torch.no_grad():
#             if hasattr(model, 'depthwiseConv'):
#                 model.depthwiseConv[0].weight.data = torch.renorm(
#                     model.depthwiseConv[0].weight.data, p=2, dim=0, maxnorm=1.0
#                 )

#         if scheduler is not None:
#             scheduler.step()
        
#         running_loss += loss.item() * x.size(0)
#         preds = logits.argmax(dim=1)
#         preds_all.append(preds.cpu())
#         labels_all.append(y.cpu())
#         total += x.size(0)
    
#     preds_all = torch.cat(preds_all).numpy()
#     labels_all = torch.cat(labels_all).numpy()
#     return running_loss / total, accuracy_score(labels_all, preds_all)

# @torch.no_grad()
# def validate(model, dataloader, loss_fn, device):
#     model.eval()
#     running_loss = 0.0
#     total = 0
#     preds_all, labels_all = [], []
#     for x, y in tqdm(dataloader, desc="Validating", leave=False):
#         x = x.to(device)
#         y = y.to(device).long()
#         logits = model(x)
#         loss = loss_fn(logits, y)
#         running_loss += loss.item() * x.size(0)
#         preds_all.append(logits.argmax(dim=1).cpu())
#         labels_all.append(y.cpu())
#         total += x.size(0)
#     return running_loss / total, accuracy_score(torch.cat(labels_all), torch.cat(preds_all))

# # ============================================================================
# # MAIN TRAINING FUNCTION
# # ============================================================================
# def train_eegnet(
#     data_dir="",
#     use_windowing=True,
#     window_size=250,
#     step_size=75,
#     batch_size=32,
#     epochs=300,
#     lr=1e-3,
#     weight_decay=1e-3,
#     dropout_rate=0.5,
#     F1=16,
#     patience=50,
#     save_prefix="outputs/eegnet_new",
#     seed=42
# ):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     X_train = np.load(os.path.join(data_dir, "X_train_eegnet.npy"))
#     y_train = np.load(os.path.join(data_dir, "y_train_eegnet.npy"))
#     X_test = np.load(os.path.join(data_dir, "X_test_eegnet.npy"))
#     y_test = np.load(os.path.join(data_dir, "y_test_eegnet.npy"))

#     X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=seed)

#     train_ds = EEGDatasetWithWindowing(X_tr, y_tr, window_size, step_size)
#     val_ds = EEGDatasetWithWindowing(X_val, y_val, window_size, window_size)
#     test_ds = EEGDatasetWithWindowing(X_test, y_test, window_size, window_size)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

#     # Infer correct dimensions from actual data (handles both 3D and 4D formats)
#     # X_train can be (N, 1, C, T) or (N, C, T) - use dataset output for ground truth
#     sample_x, _ = train_ds[0]
#     n_channels = int(sample_x.shape[0])
#     n_samples = int(sample_x.shape[1])
#     print(f"Building model for {n_channels} channels, {n_samples} timepoints (window_size={window_size})")

#     model = EEGNet(num_classes=len(np.unique(y_train)), num_channels=n_channels, 
#                    num_samples=n_samples, dropout_rate=dropout_rate, F1=F1).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
#     early_stopping = EarlyStopping(patience=patience)

#     history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
#     os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

#     for epoch in range(1, epochs + 1):
#         tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
#         va_loss, va_acc = validate(model, val_loader, criterion, device)
        
#         history["epoch"].append(epoch)
#         history["train_loss"].append(tr_loss)
#         history["train_acc"].append(tr_acc)
#         history["val_loss"].append(va_loss)
#         history["val_acc"].append(va_acc)
#         history["lr"].append(optimizer.param_groups[0]["lr"])

#         if early_stopping.step({"val_loss": va_loss, "val_acc": va_acc}, model, epoch):
#             torch.save(model.state_dict(), f"{save_prefix}_best.pth")
        
#         if early_stopping.early_stop: break

#     # Plotting & Evaluation (Keep your existing visualization code)
#     # ...
#     return model, pd.DataFrame(history), f"{save_prefix}_best.pth", va_acc

# if __name__ == "__main__":
#     config = {
#         "data_dir": "",
#         "use_windowing": True,
#         "window_size": 250,
#         "step_size": 75,
#         "batch_size": 16,
#         "epochs": 300,
#         "lr": 1e-3,
#         "weight_decay": 1e-3,
#         "dropout_rate": 0.5,
#         "F1": 16,
#         "patience": 50,
#         "save_prefix": "outputs/eegnet_new"
#     }
#     train_eegnet(**config)