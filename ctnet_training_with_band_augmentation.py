# ============================================================================
# CTNet Training with Band-Based Data Augmentation
# ============================================================================
# This script augments training data by filtering into multiple frequency bands
# Only training data is augmented; validation and test data remain unchanged
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
from scipy.signal import butter, filtfilt, cheby2, cheb2ord

# Force flush output (Windows fix)
sys.stdout.flush()

# Import CTNet models
from ctnet_model import CTNet, CTNetLite


# ============================================================================
# BAND-BASED DATA AUGMENTATION
# ============================================================================
class BandFilterAugmentation:
    """
    Augments EEG data by filtering into multiple frequency bands.
    Each band-filtered version is treated as a separate augmented sample.
    """
    
    def __init__(self, sampling_rate=250, bands=None):
        """
        Initialize band filter augmentation.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling frequency in Hz (default: 250)
        bands : dict or None
            Dictionary of band names and frequency ranges.
            If None, uses standard EEG bands:
            - delta: 1-4 Hz
            - theta: 4-8 Hz
            - alpha: 8-13 Hz
            - beta: 13-30 Hz
            - gamma: 30-40 Hz
        """
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2.0
        
        # Define standard EEG frequency bands if not provided
        if bands is None:
            self.bands = {
                # 'delta': (1, 4),
                # 'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30)
                # 'gamma': (30, 40)
            }
        else:
            self.bands = bands
        
        # Pre-compute filter coefficients for each band
        self.filter_coeffs = {}
        self._compute_filter_coeffs()
    
    def _compute_filter_coeffs(self):
        """Pre-compute Butterworth bandpass filter coefficients for each band."""
        for band_name, (low, high) in self.bands.items():
            # Normalize frequencies to Nyquist
            low_norm = low / self.nyquist
            high_norm = high / self.nyquist
            
            # Design Butterworth bandpass filter
            # Using order=5 for good frequency response
            b, a = butter(5, [low_norm, high_norm], btype='band')
            self.filter_coeffs[band_name] = (b, a)
        
        print(f"✅ Initialized {len(self.bands)} frequency band filters:")
        for band_name, (low, high) in self.bands.items():
            print(f"   {band_name:8s}: {low:3.1f}-{high:3.1f} Hz")
    
    def filter_signal(self, signal, band_name):
        """
        Apply bandpass filter to a signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Signal to filter. Shape: (channels, timepoints) or (timepoints,)
        band_name : str
            Name of the frequency band to apply
        
        Returns:
        --------
        filtered : np.ndarray
            Filtered signal with same shape as input
        """
        if band_name not in self.filter_coeffs:
            raise ValueError(f"Unknown band: {band_name}. Available: {list(self.bands.keys())}")
        
        b, a = self.filter_coeffs[band_name]
        
        # Apply filtfilt for zero-phase filtering
        if signal.ndim == 1:
            # 1D signal (timepoints,)
            filtered = filtfilt(b, a, signal)
        elif signal.ndim == 2:
            # 2D signal (channels, timepoints)
            filtered = np.zeros_like(signal)
            for ch in range(signal.shape[0]):
                filtered[ch, :] = filtfilt(b, a, signal[ch, :])
        else:
            raise ValueError(f"Unsupported signal dimension: {signal.ndim}D")
        
        return filtered
    
    def augment_batch(self, X_batch):
        """
        Augment a batch of EEG data by creating band-filtered versions.
        
        Parameters:
        -----------
        X_batch : np.ndarray or torch.Tensor
            Batch of EEG data. Shape: (batch_size, channels, timepoints)
        
        Returns:
        --------
        augmented : np.ndarray
            Augmented data. Shape: (batch_size * n_bands, channels, timepoints)
            Original data is included as the first set of samples.
        """
        # Convert to numpy if tensor
        if isinstance(X_batch, torch.Tensor):
            X_batch = X_batch.cpu().numpy()
        
        batch_size, n_channels, n_timepoints = X_batch.shape
        n_bands = len(self.bands)
        
        # Initialize augmented array: original + all band-filtered versions
        augmented = np.zeros((batch_size * (n_bands + 1), n_channels, n_timepoints))
        
        # First, add original data (no augmentation)
        augmented[:batch_size] = X_batch
        
        # Then, add each band-filtered version
        band_names = list(self.bands.keys())
        for band_idx, band_name in enumerate(band_names):
            start_idx = batch_size * (band_idx + 1)
            end_idx = batch_size * (band_idx + 2)
            
            for sample_idx in range(batch_size):
                # Filter each sample
                filtered_sample = self.filter_signal(X_batch[sample_idx], band_name)
                augmented[start_idx + sample_idx] = filtered_sample
        
        return augmented
    
    def augment_dataset(self, X_data, y_data):
        """
        Augment entire dataset by creating band-filtered versions.
        
        Parameters:
        -----------
        X_data : np.ndarray
            EEG data. Shape: (n_samples, channels, timepoints)
        y_data : np.ndarray
            Labels. Shape: (n_samples,)
        
        Returns:
        --------
        X_augmented : np.ndarray
            Augmented data. Shape: (n_samples * (n_bands + 1), channels, timepoints)
        y_augmented : np.ndarray
            Augmented labels. Shape: (n_samples * (n_bands + 1),)
        """
        n_samples = X_data.shape[0]
        n_bands = len(self.bands)
        
        print(f"\n🔄 Augmenting dataset with {n_bands} frequency bands...")
        print(f"   Original samples: {n_samples}")
        
        # Initialize augmented arrays
        X_augmented = np.zeros((n_samples * (n_bands + 1), X_data.shape[1], X_data.shape[2]))
        y_augmented = np.zeros(n_samples * (n_bands + 1), dtype=y_data.dtype)
        
        # Add original data
        X_augmented[:n_samples] = X_data
        y_augmented[:n_samples] = y_data
        
        # Add band-filtered versions
        band_names = list(self.bands.keys())
        for band_idx, band_name in enumerate(band_names):
            start_idx = n_samples * (band_idx + 1)
            end_idx = n_samples * (band_idx + 2)
            
            print(f"   Filtering band {band_idx+1}/{n_bands}: {band_name} ({self.bands[band_name][0]}-{self.bands[band_name][1]} Hz)...")
            
            for sample_idx in range(n_samples):
                # Filter each sample
                filtered_sample = self.filter_signal(X_data[sample_idx], band_name)

                # ★ NORMALIZE AFTER FILTERING ★
                filtered_sample = (filtered_sample - np.mean(filtered_sample)) / (np.std(filtered_sample) + 1e-6)

                X_augmented[start_idx + sample_idx] = filtered_sample

                y_augmented[start_idx + sample_idx] = y_data[sample_idx]
            
            sys.stdout.flush()
        
        print(f"   ✅ Augmented samples: {X_augmented.shape[0]}")
        print(f"   Augmentation factor: {X_augmented.shape[0] / n_samples:.1f}x")
        
        return X_augmented, y_augmented


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_ctnet_with_band_augmentation():
    """
    Train CTNet with band-based data augmentation.
    Only training data is augmented; validation and test remain unchanged.
    """
    
    print("=" * 80)
    print("🚀 CTNet Training with Band-Based Data Augmentation")
    print("=" * 80)
    
    # -----------------------------------------------
    # STEP 1 — LOAD DATA
    # -----------------------------------------------
    data_dir = ""  # put dataset folder path if needed
    
    print("\n📂 Loading preprocessed EEG data...")
    print("   Expected format: (N, 8, 250) - (samples, channels, timepoints)")
    X_train = np.load(os.path.join(data_dir, "X_train_ctnet.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train_ctnet.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_ctnet.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test_ctnet.npy"))
    
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Validate data shape
    if X_train.ndim != 3:
        raise ValueError(f"Expected 3D array (N, C, T), got {X_train.ndim}D: {X_train.shape}")
    if X_train.shape[1] != 8:
        raise ValueError(f"Expected 8 channels, got {X_train.shape[1]} channels")
    if X_train.shape[2] != 250:
        print(f"⚠️  Warning: Expected 250 timepoints, got {X_train.shape[2]}. Model will adapt.")
    print("✅ Data shape validation passed!")
    
    # If 4D (e.g., FBMSNet output with frequency bands), fold bands into channels
    if X_train.ndim == 4:
        print("Detected 4D input → reshaping [bands, channels] into a flat channel axis...")
        n_bands = X_train.shape[1]
        n_channels = X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], n_bands * n_channels, X_train.shape[3])
        X_test = X_test.reshape(X_test.shape[0], n_bands * n_channels, X_test.shape[3])
    
    print(f"Final train shape: {X_train.shape} (N, C, T)")
    
    num_classes = len(np.unique(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # -----------------------------------------------
    # STEP 2 — TRAIN/VAL SPLIT (BEFORE AUGMENTATION)
    # -----------------------------------------------
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=42
    )
    
    print("\nDataset sizes (before augmentation):")
    print(f"Train: {X_train_split.shape[0]}")
    print(f"Val:   {X_val.shape[0]}")
    print(f"Test:  {X_test.shape[0]}")
    
    # -----------------------------------------------
    # STEP 3 — APPLY BAND-BASED AUGMENTATION (TRAINING DATA ONLY)
    # -----------------------------------------------
    print("\n" + "=" * 80)
    print("BAND-BASED DATA AUGMENTATION")
    print("=" * 80)
    
    # Initialize band filter augmentation
    # Using standard EEG bands: delta, theta, alpha, beta, gamma
    band_aug = BandFilterAugmentation(sampling_rate=250)
    
    # Augment ONLY training data
    X_train_augmented, y_train_augmented = band_aug.augment_dataset(
        X_train_split, y_train_split
    )
    
    print(f"\n✅ Augmentation complete!")
    print(f"   Training samples: {X_train_split.shape[0]} → {X_train_augmented.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]} (unchanged)")
    print(f"   Test samples: {X_test.shape[0]} (unchanged)")
    
    # Convert to tensors
    X_train_augmented = torch.FloatTensor(X_train_augmented)
    y_train_augmented = torch.LongTensor(y_train_augmented)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # -----------------------------------------------
    # STEP 4 — DATA LOADERS
    # -----------------------------------------------
    batch_size = 8  # Smaller batch for more gradient updates per epoch
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_augmented, y_train_augmented),
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
    # STEP 5 — BUILD MODEL
    # -----------------------------------------------
    print("\n Building CTNet model...")
    
    n_channels = X_train_augmented.shape[1]
    n_timepoints = X_train_augmented.shape[2]
    
    # Use full CTNet with optimized architecture
    print("Using CTNet with optimized architecture for augmented dataset.")
    # model = CTNet(
    #     n_channels=n_channels,
    #     n_timepoints=n_timepoints,
    #     n_classes=num_classes,
    #     sampling_rate=250,
    #     F1=8,
    #     D=2,
    #     F2=16,
    #     Kc1=None,
    #     Kc2=16,
    #     P1=8,
    #     P2=2,
    #     dropout_conv=0.5,  # Moderate dropout
    #     n_heads=4,
    #     n_layers=4,
    #     ff_mult=4,
    #     dropout_transformer=0.1,
    #     use_positional_encoding=True,
    #     classifier_dropout=0.3
    # ).to(device)
    model = CTNetLite(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=num_classes,
            #dropout_conv=0.7  # Validation Accuracy: 42.55% Test Accuracy:      29.17%
            dropout_conv=0.6 #-  Validation Accuracy = 57.45% Test Accuracy =  47%
            # Final Test Accuracy: 50.00% Best Val Accuracy:      61.70%
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # -----------------------------------------------
    # STEP 6 — OPTIMIZER + LOSS
    # -----------------------------------------------
    # Add class weights to handle any imbalance
    from collections import Counter
    train_counts = Counter(y_train_augmented.numpy())
    total = sum(train_counts.values())
    class_weights = torch.FloatTensor([
        total / (len(train_counts) * train_counts[i]) for i in sorted(train_counts.keys())
    ]).to(device)
    print(f"Class weights: {class_weights}")
    
    # Use Focal Loss to focus on hard examples and prevent class collapse
    class FocalLoss(nn.Module):
        """Focal Loss to address class imbalance and hard examples."""
        def __init__(self, class_weights, alpha=1.0, gamma=2.0):
            super().__init__()
            self.class_weights = class_weights
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, preds, targets):
            ce_loss = nn.functional.cross_entropy(preds, targets, weight=self.class_weights, reduction='none')
            pt = torch.exp(-ce_loss)  # Probability of true class
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
    
    # criterion = FocalLoss(class_weights, alpha=1.0, gamma=2.0)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Moderate LR
        weight_decay=1e-3  # Moderate weight decay
    )
    
    # -----------------------------------------------
    # STEP 7 — TRAINING LOOP
    # -----------------------------------------------
    num_epochs = 500
    scheduler = None  # No scheduler - fixed learning rate
    
    patience = 200  # Early stopping patience
    patience_counter = 0
    best_val_acc = 0
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }
    
    print("\n🏁 Starting training...\n")
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: CTNet (with band augmentation)")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial LR: 1e-3")
    print(f"Weight decay: 1e-3")
    print(f"Augmentation: {len(band_aug.bands)} frequency bands")
    print(f"Training samples: {X_train_augmented.shape[0]} (augmented)")
    print(f"Validation samples: {X_val.shape[0]} (original)")
    print(f"Test samples: {X_test.shape[0]} (original)")
    print("=" * 80 + "\n")
    
    for epoch in range(num_epochs):
        
        # ------- TRAIN -------
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            
            out = model(xb)
            preds = out[0] if isinstance(out, tuple) else out
            
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
                
                out = model(xb)
                preds = out[0] if isinstance(out, tuple) else out
                
                loss = criterion(preds, yb)
                val_loss += loss.item()
                val_correct += (preds.argmax(1) == yb).sum().item()
                val_total += yb.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update scheduler if one is used
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch < 10:
            # Calculate per-class predictions to diagnose class collapse
            model.eval()
            train_preds_all = []
            train_labels_all = []
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    preds = out[0] if isinstance(out, tuple) else out
                    train_preds_all.extend(preds.argmax(1).cpu().numpy())
                    train_labels_all.extend(yb.cpu().numpy())
            model.train()
            
            from collections import Counter
            pred_dist = Counter(train_preds_all)
            true_dist = Counter(train_labels_all)
            
            print(
                f"Epoch {epoch+1:03d}/{num_epochs} | "
                f"Train {train_acc:5.1f}% | Val {val_acc:5.1f}% | "
                f"LR {optimizer.param_groups[0]['lr']:.6f}"
            )
            print(f"   Predicted: {dict(sorted(pred_dist.items()))} | True: {dict(sorted(true_dist.items()))}")
        
        # Best model saving + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch,
                 "val_acc": val_acc},
                "ctnet_best_model_band_aug.pth"
            )
            if (epoch + 1) % 5 == 0 or epoch < 10:
                print(f"   ✅ New best model saved! Val Acc = {best_val_acc:.2f}%")
        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\n🛑 Early Stopping Triggered!")
                break
    
    print("\n✅ Training Completed.")
    print(f"Best Validation Accuracy = {best_val_acc:.2f}%")
    
    # -----------------------------------------------
    # STEP 8 — TEST EVALUATION
    # -----------------------------------------------
    print("\n📊 Loading best checkpoint for test evaluation...")
    ckpt = torch.load("ctnet_best_model_band_aug.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            out = model(xb)
            preds = out[0] if isinstance(out, tuple) else out
            pred_labels = preds.argmax(1)
            
            all_preds.extend(pred_labels.cpu().numpy())
            all_true.extend(yb.cpu().numpy())
            
            test_correct += (pred_labels == yb).sum().item()
            test_total += yb.size(0)
    
    test_acc = 100 * test_correct / test_total
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")
    print(f"Best Val Accuracy:      {best_val_acc:.2f}%")
    print(f"Generalization Gap:    {abs(best_val_acc - test_acc):.2f}%")
    print("=" * 80 + "\n")
    
    # Classification report
    class_labels = [f"Class {i}" for i in range(num_classes)]
    print("Classification Report:")
    print(classification_report(all_true, all_preds, target_names=class_labels))
    
    # -----------------------------------------------
    # STEP 9 — PLOTS
    # -----------------------------------------------
    os.makedirs("output", exist_ok=True)
    
    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["val_acc"], label="Val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history["lr"])
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].grid(True, alpha=0.3)
    
    gap = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    axes[1, 1].plot(gap)
    axes[1, 1].set_title("Overfitting Gap (Train - Val)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy Difference (%)")
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./output/ctnet_band_aug_training_curves.png", dpi=300)
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"CTNet with Band Augmentation - Test Accuracy: {test_acc:.2f}%")
    plt.colorbar()
    
    plt.xticks(range(num_classes), class_labels, rotation=45)
    plt.yticks(range(num_classes), class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    # Values inside matrix
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    
    plt.tight_layout()
    plt.savefig("./output/ctnet_band_aug_confusion_matrix.png", dpi=300)
    plt.show()
    
    print("\n✅ All visualizations saved to ./output/")
    
    return test_acc, best_val_acc


# ============================================================================
# MAIN ENTRY (REQUIRED FOR WINDOWS)
# ============================================================================
if __name__ == "__main__":
    test_acc, val_acc = train_ctnet_with_band_augmentation()
    
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY - CTNet with Band-Based Augmentation")
    print("=" * 80)
    print(f"Best Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Test Accuracy:      {test_acc:.2f}%")
    print("=" * 80)
    print("\nNote: Training data was augmented with frequency band filtering.")
    print("      Validation and test data remained unchanged (original).")
    print("=" * 80)

