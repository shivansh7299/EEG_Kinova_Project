# ===============================================================
# train_custom_fbmsnet.py — full version with validation + plots
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
val_size   = int(0.2 * len(full_train))
train_size = len(full_train) - val_size
train_ds, val_ds = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds,  batch_size=8, shuffle=False)
_, test_loader = create_dataloaders(data_dir, batch_size=8, shuffle=False)

y_train = np.load(f"{data_dir}/y_train_fbmsnet.npy")
num_classes = len(np.unique(y_train))
print(f"Detected number of classes: {num_classes}")

# ===============================================================
# 2. Model, loss, optimizer, scheduler
# ===============================================================
model = FBMSNet(
    nChan=8, nTime=248, nBands=9, nClass=num_classes,
    dropoutP=0.5, m=32, temporalLayer='LogVarLayer', doWeightNorm=True
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=50, verbose=True
)
print(model)

# ===============================================================
# 3. Training Loop
# ===============================================================
num_epochs = 300
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(logits, 1)
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
    scheduler.step(val_acc)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

# ===============================================================
# 4. Plot Training Curves
# ===============================================================
epochs = np.arange(1, num_epochs+1)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy Curve"); plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)
plt.show()

# ===============================================================
# 5. Test Evaluation + Confusion Matrix
# ===============================================================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model(X)
        _, preds = torch.max(logits, 1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
test_acc = (all_preds == all_labels).mean() * 100
print(f"\n✅ Final Test Accuracy: {test_acc:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', colorbar=False)
plt.title("FBMSNet Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ===============================================================
# 6. Save model and metrics
# ===============================================================
torch.save(model.state_dict(), "fbmsnet_trained.pth")
np.savez("training_metrics.npz",
         train_losses=train_losses, val_losses=val_losses,
         train_accs=train_accs, val_accs=val_accs)
print("✅ Model & metrics saved.")
