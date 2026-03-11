# ===============================================================
# train_custom_fbmsnet.py — CORRECTED VERSION
# ===============================================================
import torch, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn, torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import random_split, DataLoader
from codes.centralRepo.networks import FBMSNet
from codes.centralRepo.eegDataset_custom import EEGMultiBandDataset, create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ===============================================================
# 1. Data
# ===============================================================
data_dir = "data"
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

# FIXED: Use NLLLoss because model outputs LogSoftmax
# criterion = nn.NLLLoss()  # Changed from CrossEntropyLoss

criterion = nn.CrossEntropyLoss()  # Changed back to CrossEntropyLoss

# Use a single fixed learning rate (no scheduler)
optimizer = optim.Adam(model.parameters(), lr=3e-3)

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
num_epochs = 250
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

best_val_acc = 0.0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X)  # Model returns (logits with LogSoftmax, features)
        loss = criterion(logits, y)  # NLLLoss works with LogSoftmax
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(logits, 1)  # Get predictions from LogSoftmax
        total += y.size(0)
        correct += (preds == y).sum().item()

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
          f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

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
plt.savefig("fbmsnet_training_curves.png", dpi=150, bbox_inches='tight')
print("\n📊 Training curves saved to: fbmsnet_training_curves.png")

# ===============================================================
# 6. Confusion Matrix
# ===============================================================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Test Set Confusion Matrix")
plt.savefig("fbmsnet_confusion_matrix.png", dpi=150, bbox_inches='tight')
print("📊 Confusion matrix saved to: fbmsnet_confusion_matrix.png")

# ===============================================================
# 7. Save Model
# ===============================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'num_classes': num_classes
}, 'fbmsnet_best_model.pth')
print("💾 Model saved to: fbmsnet_best_model.pth")

print("\n✅ Training complete!")

