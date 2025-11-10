"""
Quick diagnostic to identify where the training is hanging
"""
import numpy as np
import torch
import torch.nn as nn
from ctnet_model import CTNetLite
import os

print("="*80)
print("🔍 DIAGNOSTIC TEST")
print("="*80)

# Load data
print("\n1. Loading data...")
X_train = np.load(os.path.join("", "X_train_ctnet.npy"))
y_train = np.load(os.path.join("", "y_train_ctnet.npy"))
print(f"   ✅ Data loaded: {X_train.shape}")

# Simple processing
print("\n2. Converting to tensors...")
X_train = torch.tensor(X_train[:100], dtype=torch.float32)  # Just use 100 samples
y_train = torch.tensor(y_train[:100], dtype=torch.long)
print(f"   ✅ Tensors created: {X_train.shape}")

# Create simple dataloader
print("\n3. Creating DataLoader...")
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True,
    num_workers=0
)
print(f"   ✅ DataLoader created")

# Create model
print("\n4. Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTNetLite(
    n_channels=X_train.shape[1],
    n_timepoints=X_train.shape[2],
    n_classes=len(np.unique(y_train.numpy())),
    dropout=0.5
).to(device)
print(f"   ✅ Model created on {device}")

# Test forward pass (no DataLoader)
print("\n5. Testing forward pass (single batch, no DataLoader)...")
test_batch = X_train[:4].to(device)
test_labels = y_train[:4].to(device)
print(f"   Input shape: {test_batch.shape}")

try:
    with torch.no_grad():
        output, features = model(test_batch)
    print(f"   ✅ Forward pass successful! Output: {output.shape}")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    exit(1)

# Test with DataLoader
print("\n6. Testing DataLoader iteration...")
try:
    for i, (xb, yb) in enumerate(train_loader):
        print(f"   Batch {i+1}: xb={xb.shape}, yb={yb.shape}")
        if i >= 2:  # Just test 3 batches
            break
    print(f"   ✅ DataLoader iteration successful!")
except Exception as e:
    print(f"   ❌ DataLoader iteration failed: {e}")
    exit(1)

# Test training step
print("\n7. Testing single training step...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

try:
    model.train()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    
    print(f"   Moving data to device...")
    optimizer.zero_grad()
    print(f"   Zeroed gradients...")
    
    preds, _ = model(xb)
    print(f"   Forward pass done: {preds.shape}")
    
    loss = criterion(preds, yb)
    print(f"   Loss computed: {loss.item():.4f}")
    
    loss.backward()
    print(f"   Backward pass done")
    
    optimizer.step()
    print(f"   ✅ Optimizer step successful!")
    
except Exception as e:
    print(f"   ❌ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Full epoch test
print("\n8. Testing full epoch...")
try:
    model.train()
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds, _ = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        print(f"   Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
    print(f"   ✅ Full epoch completed!")
except Exception as e:
    print(f"   ❌ Full epoch failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nIf you see this, the problem is NOT with:")
print("  - Data loading")
print("  - Model architecture")
print("  - DataLoader")
print("  - Training step")
print("\nThe issue might be:")
print("  - Something in the full training script")
print("  - Memory issues with full dataset")
print("  - Progress printing causing delays")
print("="*80)