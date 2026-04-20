# ===============================================================
# train_custom_fbmsnet.py — CORRECTED VERSION
# ===============================================================
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split

from codes.centralRepo.eegDataset_custom import EEGMultiBandDataset, create_dataloaders
from codes.centralRepo.networks import FBMSNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_artifacts import build_run_paths, save_classification_outputs, save_confusion_matrix_figure, save_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

parser = argparse.ArgumentParser(description="Train FBMSNet and save subject-aware results.")
parser.add_argument("--data-dir", default="data", help="Directory containing FBMSNet npy files")
parser.add_argument("--subject-id", type=int, default=None, help="Optional subject id used for results folder naming")
args = parser.parse_args()

# ===============================================================
# 1. Data
# ===============================================================
data_dir = args.data_dir
full_train = EEGMultiBandDataset(data_dir, 'train')
val_size   = int(0.1 * len(full_train))
train_size = len(full_train) - val_size
train_ds, val_ds = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,  batch_size=16, shuffle=False)
_, test_loader = create_dataloaders(data_dir, batch_size=16, shuffle=False)

y_train = np.load(f"{data_dir}/y_train_fbmsnet.npy")
num_classes = len(np.unique(y_train))
print(f"Detected number of classes: {num_classes}")
results_paths = build_run_paths(
    PROJECT_ROOT,
    "FBMSNet",
    data_dir,
    num_classes=num_classes,
    subject_no=args.subject_id,
)
figures_dir = results_paths["figures_dir"]
metrics_dir = results_paths["metrics_dir"]
checkpoints_dir = results_paths["checkpoints_dir"]
run_dir = results_paths["run_dir"]
print(f"Results directory: {run_dir}")

# ===============================================================
# 2. Model, loss, optimizer, scheduler
# ===============================================================
# FIXED: Correct model parameters
model = FBMSNet(
    nChan=8,                    # Number of EEG channels
    nTime=248,                  # Number of time samples (must be divisible by 4)
    nClass=num_classes,         # Number of classes
    temporalLayer='LogVarLayer', # Temporal aggregation layer
    num_Feat=36,               # Number of temporal features (was incorrectly 'm=32')
    dilatability=8,            # Spatial expansion factor (default is 8)
    dropoutP=0.6               # Dropout probability
).to(device)

# FBMSNet classifier head ends with LogSoftmax, so use NLLLoss.
criterion = nn.NLLLoss()

# Use a single fixed learning rate (no scheduler)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Verify input shape
print("\n🔍 Verifying data shapes...")
sample_X, sample_y = next(iter(train_loader))
print(f"Input shape: {sample_X.shape}")
print(f"Expected shape: (batch, 9, 8, 248)")
assert sample_X.shape[1:] == (9, 8, 248), f"❌ Shape mismatch! Got {sample_X.shape[1:]}, expected (9, 8, 248)"
print("✅ Shape verification passed!")

print("\n📊 Model Summary:")
print(model)

# ===============================================================
# 3. Training Loop
# ===============================================================
num_epochs = 300
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

best_val_acc = 0.0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---", flush=True)

    for batch_idx, (X, y) in enumerate(train_loader, start=1):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X)  # Model returns (logits with LogSoftmax, features)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(logits, 1)  # Get predictions from LogSoftmax
        total += y.size(0)
        correct += (preds == y).sum().item()

        # Print periodic batch progress so GUI log updates continuously.
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(train_loader):
            batch_acc = 100.0 * correct / max(total, 1)
            avg_loss_so_far = running_loss / batch_idx
            print(
                f"  Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {avg_loss_so_far:.4f} | Acc: {batch_acc:.2f}%",
                flush=True,
            )

    train_loss = running_loss / len(train_loader)
    train_acc  = 100.0 * correct / total

    # ---- Validation ----
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            loss = criterion(logits, y)
            val_loss += loss.item()
            _, preds = torch.max(logits, 1)
            val_total += y.size(0)
            val_correct += (preds == y).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%", flush=True)

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\n✅ Loaded best model with validation accuracy: {best_val_acc:.2f}%")

# ===============================================================
# 4. Test Evaluation
# ===============================================================
print("\n🧪 Evaluating on test set...")
model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model(X)
        _, preds = torch.max(logits, 1)
        test_total += y.size(0)
        test_correct += (preds == y).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

test_acc = 100.0 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

# ===============================================================
# 5. Plot Training Curves
# ===============================================================
epochs = np.arange(1, num_epochs+1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label="Train Acc", linewidth=2)
plt.plot(epochs, val_accs, label="Val Acc", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
train_curves_path = figures_dir / "fbmsnet_training_curves.png"
plt.savefig(train_curves_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n📊 Training curves saved to: {train_curves_path}")

# ===============================================================
# 6. Confusion Matrix
# ===============================================================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Test Set Confusion Matrix")
cm_path = figures_dir / "fbmsnet_confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"📊 Confusion matrix saved to: {cm_path}")

class_labels = [f"Class {i}" for i in range(num_classes)]
save_classification_outputs(metrics_dir, all_labels, all_preds, class_labels, "fbmsnet")
save_confusion_matrix_figure(
    figures_dir / "fbmsnet_confusion_matrix_clean.png",
    cm,
    class_labels,
    f"FBMSNet Confusion Matrix (Test Acc {test_acc:.2f}%)",
)
save_json(
    metrics_dir / "fbmsnet_training_summary.json",
    {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "num_classes": int(num_classes),
        "epochs": int(num_epochs),
    },
)

# ===============================================================
# 7. Save Model
# ===============================================================
model_path = checkpoints_dir / "fbmsnet_best_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'num_classes': num_classes
}, str(model_path))
print(f"💾 Model saved to: {model_path}")

save_json(
    metrics_dir / "fbmsnet_epoch_history.json",
    {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
    },
)

print("\n✅ Training complete!")

