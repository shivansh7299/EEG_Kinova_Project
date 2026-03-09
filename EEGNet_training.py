# ============================================================================
# EEGNet Training Script for Motor Imagery Classification
# ============================================================================
# Training script for EEGNet model with visualization
# Supports: 8-channel, 250 Hz, 1-second EEG trials
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

# Force flush output (Windows fix)
sys.stdout.flush()

# Import EEGNet model
from EEGNet_model import EEGNet, EEGNetLite


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_eegnet():
    """
    Main training function for EEGNet model
    """
    print("=" * 80)
    print("🚀 EEGNet Training (8 ch, 250 Hz, 1s Window)")
    print("=" * 80)
    
    # -----------------------------------------------
    # STEP 1 — LOAD DATA
    # -----------------------------------------------
    # NOTE: Data should be preprocessed using preprocessing_3_class_EEGNet.ipynb
    # Expected format: (N, 1, C, T) or (N, C, T) where N=samples, C=8 channels, T=250 timepoints
    
    data_dir = ""  # put dataset folder path if needed
    
    print("\n📂 Loading preprocessed EEG data...")
    print("   Expected format: (N, 1, 8, 250) or (N, 8, 250) - (samples, channels, timepoints)")
    X_train = np.load(os.path.join(data_dir, "X_train_eegnet.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train_eegnet.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_eegnet.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test_eegnet.npy"))
    
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Validate data shape
    if X_train.ndim not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D array, got {X_train.ndim}D: {X_train.shape}")
    
    # Handle different input shapes
    if X_train.ndim == 4:
        # (N, 1, C, T) -> (N, C, T) for consistency
        if X_train.shape[1] == 1:
            X_train = X_train.squeeze(1)
            X_test = X_test.squeeze(1)
            print("   Reshaped from 4D to 3D: (N, 1, C, T) -> (N, C, T)")
    
    if X_train.shape[1] != 8:
        raise ValueError(f"Expected 8 channels, got {X_train.shape[1]} channels")
    if X_train.shape[2] != 250:
        print(f"⚠️  Warning: Expected 250 timepoints, got {X_train.shape[2]}. Model will adapt.")
    
    print("✅ Data shape validation passed!")
    print(f"Final train shape: {X_train.shape} (N, C, T)")
    
    num_classes = len(np.unique(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # -----------------------------------------------
    # STEP 2 — TRAIN/VAL SPLIT
    # -----------------------------------------------
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=42
    )
    
    print("\nDataset sizes:")
    print(f"Train: {X_train_split.shape[0]}")
    print(f"Val:   {X_val.shape[0]}")
    print(f"Test:  {X_test.shape[0]}")
    
    # Convert to tensors
    X_train_split = torch.FloatTensor(X_train_split)
    y_train_split = torch.LongTensor(y_train_split)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # -----------------------------------------------
    # STEP 3 — DATA LOADERS
    # -----------------------------------------------
    batch_size = 16  # Adjust based on your GPU memory
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_split, y_train_split),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # Windows safe
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # -----------------------------------------------
    # STEP 4 — BUILD MODEL
    # -----------------------------------------------
    print("\n🏗️ Building EEGNet model...")
    
    n_channels = X_train_split.shape[1]
    n_timepoints = X_train_split.shape[2]
    
    # Choose model based on dataset size
    if X_train_split.shape[0] < 2000:
        print("Using EEGNetLite (small dataset)")
        model = EEGNetLite(
            n_classes=num_classes,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            dropout_rate=0.5
        ).to(device)
    else:
        print("Using full EEGNet")
        model = EEGNet(
            n_classes=num_classes,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            F1=8,
            F2=16,
            D=2,
            dropout_rate=0.5
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # -----------------------------------------------
    # STEP 5 — OPTIMIZER + SCHEDULER
    # -----------------------------------------------
    # Add class weights to handle imbalance
    from collections import Counter
    train_counts = Counter(y_train_split.numpy())
    total = sum(train_counts.values())
    class_weights = torch.FloatTensor([
        total / (len(train_counts) * train_counts[i]) for i in sorted(train_counts.keys())
    ]).to(device)
    print(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # No learning rate scheduler - using constant learning rate
    initial_lr = optimizer.param_groups[0]['lr']
    print(f"Using constant learning rate: {initial_lr}")
    
    # -----------------------------------------------
    # STEP 6 — TRAINING LOOP
    # -----------------------------------------------
    num_epochs = 300
    patience = 100  # Early stopping patience
    patience_counter = 0
    best_val_acc = 0
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }
    
    print("\n🏁 Starting training...\n")
    
    for epoch in range(num_epochs):
        # ------- TRAIN -------
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (preds.argmax(1) == yb).sum().item()
            train_total += yb.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # ------- VALIDATE -------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                
                val_loss += loss.item()
                val_correct += (preds.argmax(1) == yb).sum().item()
                val_total += yb.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Learning rate remains constant (no scheduler)
        
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(
                f"Epoch {epoch+1:03d}/{num_epochs} | "
                f"Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}% | "
                f"T_Loss: {avg_train_loss:.4f} | V_Loss: {avg_val_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            sys.stdout.flush()
        
        # Best model saving + early stopping
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc
                },
                "eegnet_best_model.pth"
            )
            if (epoch + 1) % 5 == 0 or epoch < 10:
                print(f"   ✅ New best model saved! Val Acc = {best_val_acc:.2f}% (+{improvement:.2f}%)")
                sys.stdout.flush()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                print(f"   Best validation: {best_val_acc:.2f}%")
                sys.stdout.flush()
                break
    
    print("\nTraining Completed.")
    print(f"Best Validation Accuracy = {best_val_acc:.2f}%")
    
    # -----------------------------------------------
    # STEP 7 — TEST EVALUATION
    # -----------------------------------------------
    print("\nLoading best checkpoint...")
    ckpt = torch.load("eegnet_best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            pred_labels = preds.argmax(1)
            
            all_preds.extend(pred_labels.cpu().numpy())
            all_true.extend(yb.cpu().numpy())
            
            test_correct += (pred_labels == yb).sum().item()
            test_total += yb.size(0)
    
    test_acc = 100 * test_correct / test_total
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"✅ Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print("=" * 80 + "\n")
    
    # -----------------------------------------------
    # STEP 8 — PLOTS AND VISUALIZATIONS
    # -----------------------------------------------
    os.makedirs("output", exist_ok=True)
    
    # Classification Report
    class_labels = [f"Class {i}" for i in range(num_classes)]
    print("Classification Report:")
    print(classification_report(all_true, all_preds, target_names=class_labels))
    
    # Training curves - 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0, 0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
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
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Training & Validation Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot (constant)
    axes[1, 0].plot(history["lr"], linewidth=2, color='green')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate (Constant)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting gap plot
    gap = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    axes[1, 1].plot(gap, linewidth=2, color='red')
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Train - Val Accuracy (%)")
    axes[1, 1].set_title("Overfitting Monitor")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./output/eegnet_training_curves.png", dpi=300)
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"EEGNet Confusion Matrix (Test Acc {test_acc:.2f}%)", fontsize=14)
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
    plt.savefig("./output/eegnet_confusion_matrix.png", dpi=300)
    plt.show()
    
    print("\n✅ All visualizations saved to ./output/")
    
    return test_acc, best_val_acc


# ============================================================================
# MAIN ENTRY (REQUIRED FOR WINDOWS)
# ============================================================================
if __name__ == "__main__":
    test_acc, val_acc = train_eegnet()
    
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"Best Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Test Accuracy:      {test_acc:.2f}%")
    print("=" * 80)

