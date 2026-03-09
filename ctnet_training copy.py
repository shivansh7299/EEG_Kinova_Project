# # # ============================================================================
# # # TRAINING SCRIPT FOR CTNet (Windows Compatible)
# # # ============================================================================
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from sklearn.metrics import confusion_matrix, classification_report
# # from sklearn.model_selection import train_test_split
# # import matplotlib.pyplot as plt
# # import os

# # # Import CTNet
# # from ctnet_model import CTNet, CTNetLite


# # def train_ctnet():
# #     """Main training function wrapped for Windows compatibility"""
    
# #     print("="*80)
# #     print("🚀 Training CTNet for Motor Imagery Classification")
# #     print("="*80)
    
# #     # ========================================================================
# #     # STEP 1: LOAD DATA (from .npy files)
# #     # ========================================================================
# #     print("\n📂 Loading preprocessed CTNet data...")
    
# #     # Adjust this path to wherever you saved them
# #     data_dir = ""
    
# #     X_train = np.load(os.path.join(data_dir, "X_train_ctnet.npy"))
# #     y_train = np.load(os.path.join(data_dir, "y_train_ctnet.npy"))
# #     X_test  = np.load(os.path.join(data_dir, "X_test_ctnet.npy"))
# #     y_test  = np.load(os.path.join(data_dir, "y_test_ctnet.npy"))
    
# #     print(f"✅ Data loaded successfully!")
# #     print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
# #     print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")
# #     print(f"  Unique train labels: {np.unique(y_train)}")
    
# #     num_classes = len(np.unique(y_train))
    
# #     # Device setup
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"Using device: {device}")
    
# #     # ========================================================================
# #     # STEP 2: RESHAPE DATA FOR CTNet
# #     # ========================================================================
# #     print("\n🔧 Reshaping data for CTNet...")
# #     if X_train.ndim == 4:  # If coming from FBMSNet (bands, channels, time)
# #         X_train_ctnet = X_train.mean(axis=1)
# #         X_test_ctnet = X_test.mean(axis=1)
# #     else:
# #         X_train_ctnet, X_test_ctnet = X_train, X_test
    
# #     print(f"After reshaping:")
# #     print(f"  X_train: {X_train_ctnet.shape}  (batch, channels, timepoints)")
# #     print(f"  X_test:  {X_test_ctnet.shape}")
    
# #     # ========================================================================
# #     # STEP 3: CREATE VALIDATION SPLIT
# #     # ========================================================================
# #     X_train_ctnet, X_val, y_train, y_val = train_test_split(
# #         X_train_ctnet, y_train,
# #         test_size=0.2,
# #         stratify=y_train,
# #         random_state=42
# #     )
    
# #     print(f"\nAfter validation split:")
# #     print(f"  Train: {X_train_ctnet.shape[0]} samples")
# #     print(f"  Val:   {X_val.shape[0]} samples")
# #     print(f"  Test:  {X_test_ctnet.shape[0]} samples")
    
# #     # Convert to tensors
# #     X_train_ctnet = torch.tensor(X_train_ctnet, dtype=torch.float32)
# #     y_train = torch.tensor(y_train, dtype=torch.long)
# #     X_val = torch.tensor(X_val, dtype=torch.float32)
# #     y_val = torch.tensor(y_val, dtype=torch.long)
# #     X_test_ctnet = torch.tensor(X_test_ctnet, dtype=torch.float32)
# #     y_test = torch.tensor(y_test, dtype=torch.long)
    
# #     # ========================================================================
# #     # STEP 4: CREATE DATALOADERS - WINDOWS FIX
# #     # ========================================================================
# #     batch_size = 64
    
# #     train_loader = torch.utils.data.DataLoader(
# #         torch.utils.data.TensorDataset(X_train_ctnet, y_train),
# #         batch_size=batch_size,
# #         shuffle=True,
# #         num_workers=0,      # ← CHANGED: Set to 0 for Windows
# #         pin_memory=False    # ← CHANGED: Disable for Windows
# #     )
    
# #     val_loader = torch.utils.data.DataLoader(
# #         torch.utils.data.TensorDataset(X_val, y_val),
# #         batch_size=batch_size,
# #         shuffle=False,
# #         num_workers=0,      # ← CHANGED: Set to 0 for Windows
# #         pin_memory=False    # ← CHANGED: Disable for Windows
# #     )
    
# #     test_loader = torch.utils.data.DataLoader(
# #         torch.utils.data.TensorDataset(X_test_ctnet, y_test),
# #         batch_size=batch_size,
# #         shuffle=False,
# #         num_workers=0       # ← CHANGED: Set to 0 for Windows
# #     )
    
# #     # ========================================================================
# #     # STEP 5: INITIALIZE CTNET MODEL
# #     # ========================================================================
# #     print("\n🏗️ Building CTNet model...")
    
# #     n_channels = X_train_ctnet.shape[1]
# #     n_timepoints = X_train_ctnet.shape[2]
    
# #     if X_train_ctnet.shape[0] > 3000:
# #         print("📉 Using CTNetLite (small dataset)")
# #         model = CTNetLite(
# #             n_channels=n_channels,
# #             n_timepoints=n_timepoints,
# #             n_classes=num_classes,
# #             dropout=0.6  # Increased for better regularization
# #         ).to(device)
# #     else:
# #         print("📊 Using full CTNet")
# #         model = CTNet(
# #             n_channels=n_channels,
# #             n_timepoints=n_timepoints,
# #             n_classes=num_classes,
# #             embed_dim=128,  # Increased from 64
# #             num_heads=4,
# #             num_layers=2,
# #             dropout=0.5
# #         ).to(device)
    
# #     total_params = sum(p.numel() for p in model.parameters())
# #     print(f"✅ Model created with {total_params:,} parameters")
# #     print(f"   Samples per parameter: {X_train_ctnet.shape[0] / total_params:.4f}")
    
# #     # ========================================================================
# #     # STEP 6: TRAINING SETUP
# #     # ========================================================================
# #     criterion = nn.CrossEntropyLoss()
# #     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# #     scheduler = optim.lr_scheduler.CosineAnnealingLR(
# #         optimizer, T_max=200, eta_min=1e-5
# #     )
    
# #     num_epochs = 200
# #     best_val_acc = 0
# #     patience = 200
# #     patience_counter = 0
    
# #     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
# #     print("\n" + "="*80)
# #     print("TRAINING CONFIGURATION")
# #     print("="*80)
# #     print(f"Model: {'CTNetLite' if X_train_ctnet.shape[0] < 3000 else 'CTNet'}")
# #     print(f"Epochs: {num_epochs}")
# #     print(f"Batch size: {batch_size}")
# #     print(f"Optimizer: AdamW, LR=1e-3, Scheduler=CosineAnnealing")
# #     print(f"Device: {device}")
# #     print(f"Early stopping patience: {patience}")
# #     print("="*80 + "\n")
    
# #     # ========================================================================
# #     # STEP 7: TRAINING LOOP
# #     # ========================================================================
# #     print("Starting training...\n")
    
# #     for epoch in range(num_epochs):
# #         # TRAINING
# #         model.train()
# #         train_loss = 0
# #         train_correct = 0
# #         train_total = 0
    
# #         for xb, yb in train_loader:
# #             xb, yb = xb.to(device), yb.to(device)
# #             optimizer.zero_grad()
# #             preds, _ = model(xb)
# #             loss = criterion(preds, yb)
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# #             optimizer.step()
    
# #             train_loss += loss.item()
# #             train_correct += (torch.argmax(preds, 1) == yb).sum().item()
# #             train_total += yb.size(0)
    
# #         avg_train_loss = train_loss / len(train_loader)
# #         train_acc = 100 * train_correct / train_total
    
# #         # VALIDATION
# #         model.eval()
# #         val_loss = 0
# #         val_correct = 0
# #         val_total = 0
        
# #         with torch.no_grad():
# #             for xb, yb in val_loader:
# #                 xb, yb = xb.to(device), yb.to(device)
# #                 preds, _ = model(xb)
# #                 loss = criterion(preds, yb)
# #                 val_loss += loss.item()
# #                 val_correct += (torch.argmax(preds, 1) == yb).sum().item()
# #                 val_total += yb.size(0)
    
# #         avg_val_loss = val_loss / len(val_loader)
# #         val_acc = 100 * val_correct / val_total
    
# #         scheduler.step()
    
# #         history['train_loss'].append(avg_train_loss)
# #         history['train_acc'].append(train_acc)
# #         history['val_loss'].append(avg_val_loss)
# #         history['val_acc'].append(val_acc)
    
# #         if (epoch + 1) % 10 == 0 or epoch == 0:
# #             current_lr = optimizer.param_groups[0]['lr']
# #             print(f"Epoch {epoch+1:03d}/{num_epochs} | "
# #                   f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
# #                   f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
# #                   f"LR: {current_lr:.6f}")
    
# #         # Early stopping
# #         if val_acc > best_val_acc:
# #             best_val_acc = val_acc
# #             patience_counter = 0
# #             torch.save(model.state_dict(), 'ctnet_best_model.pth')
# #             if (epoch + 1) % 10 == 0 or epoch == 0:
# #                 print(f"    ✅ New best: {best_val_acc:.2f}%")
# #         else:
# #             patience_counter += 1
# #             if patience_counter >= patience:
# #                 print(f"\n🛑 Early stopping at epoch {epoch+1}")
# #                 break
    
# #     print(f"\n" + "="*80)
# #     print("TRAINING COMPLETE")
# #     print("="*80)
# #     print(f"Best validation accuracy: {best_val_acc:.2f}%")
# #     print("="*80 + "\n")
    
# #     # ========================================================================
# #     # STEP 8: FINAL EVALUATION
# #     # ========================================================================
# #     print("Evaluating best model on test set...")
# #     model.load_state_dict(torch.load('ctnet_best_model.pth'))
# #     model.eval()
    
# #     test_correct, test_total = 0, 0
# #     all_preds, all_true = [], []
    
# #     with torch.no_grad():
# #         for xb, yb in test_loader:
# #             xb, yb = xb.to(device), yb.to(device)
# #             preds, _ = model(xb)
# #             pred_labels = torch.argmax(preds, 1)
# #             all_preds.extend(pred_labels.cpu().numpy())
# #             all_true.extend(yb.cpu().numpy())
# #             test_correct += (pred_labels == yb).sum().item()
# #             test_total += yb.size(0)
    
# #     test_acc = 100 * test_correct / test_total
    
# #     print("\n" + "="*80)
# #     print("FINAL TEST RESULTS")
# #     print("="*80)
# #     print(f"✅ Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
# #     print("="*80 + "\n")
    
# #     # ========================================================================
# #     # STEP 9: REPORTS & VISUALS
# #     # ========================================================================
# #     class_labels = [f"Class {i}" for i in range(num_classes)]
    
# #     print("Classification Report:")
# #     print(classification_report(all_true, all_preds, target_names=class_labels))
    
# #     # Plot Training Curves
# #     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
# #     axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
# #     axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
# #     axes[0].set_xlabel('Epoch')
# #     axes[0].set_ylabel('Loss')
# #     axes[0].set_title("CTNet: Loss over Epochs")
# #     axes[0].legend()
# #     axes[0].grid(True, alpha=0.3)
    
# #     axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
# #     axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
# #     axes[1].axhline(100/num_classes, color='r', linestyle='--', 
# #                     label=f'Random ({100/num_classes:.1f}%)', alpha=0.5)
# #     axes[1].set_xlabel('Epoch')
# #     axes[1].set_ylabel('Accuracy (%)')
# #     axes[1].set_title("CTNet: Accuracy over Epochs")
# #     axes[1].legend()
# #     axes[1].grid(True, alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig('../output/ctnet_training_history.png', dpi=300)
# #     plt.show()
    
# #     # Confusion Matrix
# #     cm = confusion_matrix(all_true, all_preds)
    
# #     plt.figure(figsize=(8, 6))
# #     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# #     plt.title(f"CTNet Confusion Matrix\nTest Accuracy: {test_acc:.2f}%", fontsize=14)
# #     plt.colorbar()
    
# #     tick_marks = np.arange(len(class_labels))
# #     plt.xticks(tick_marks, class_labels, rotation=45)
# #     plt.yticks(tick_marks, class_labels)
# #     plt.xlabel('Predicted Label')
# #     plt.ylabel('True Label')
    
# #     # Print values inside boxes
# #     for i in range(len(class_labels)):
# #         for j in range(len(class_labels)):
# #             plt.text(j, i, format(cm[i, j], 'd'),
# #                      ha="center", va="center",
# #                      color="white" if cm[i, j] > cm.max()/2 else "black")
    
# #     plt.tight_layout()
# #     plt.savefig('../output/ctnet_confusion_matrix.png', dpi=300)
# #     plt.show()
    
# #     print("\n✅ Visualizations saved to ../output/")
    
# #     return test_acc, best_val_acc


# # # ============================================================================
# # # MAIN ENTRY POINT - REQUIRED FOR WINDOWS
# # # ============================================================================
# # if __name__ == '__main__':
# #     # This wrapper is REQUIRED for Windows multiprocessing
# #     # Even with num_workers=0, it's good practice
# #     test_acc, val_acc = train_ctnet()
    
# #     print("\n" + "="*80)
# #     print("📊 FINAL SUMMARY")
# #     print("="*80)
# #     print(f"Best Validation Accuracy: {val_acc:.2f}%")
# #     print(f"Final Test Accuracy: {test_acc:.2f}%")
# #     print("="*80)


# # ============================================================================
# # CTNet FULL MODEL Training - No Augmentation, Optimized for Small Dataset
# # ============================================================================
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import os
# import sys

# sys.stdout.flush()

# from ctnet_model import CTNet, CTNetLite


# def train_ctnet():
#     """Training with full CTNet, no augmentation"""
    
#     print("="*80)
#     print("🚀 CTNet FULL MODEL Training (No Augmentation)")
#     print("="*80)
    
#     # ========================================================================
#     # STEP 1: LOAD DATA
#     # ========================================================================
#     print("\n📂 Loading data...")
#     sys.stdout.flush()
    
#     data_dir = ""
#     X_train = np.load(os.path.join(data_dir, "X_train_ctnet.npy"))
#     y_train = np.load(os.path.join(data_dir, "y_train_ctnet.npy"))
#     X_test = np.load(os.path.join(data_dir, "X_test_ctnet.npy"))
#     y_test = np.load(os.path.join(data_dir, "y_test_ctnet.npy"))
    
#     print(f"✅ Data loaded:")
#     print(f"   Train: {X_train.shape}")
#     print(f"   Test: {X_test.shape}")
#     print(f"   Classes: {np.unique(y_train)}")
#     sys.stdout.flush()
    
#     num_classes = len(np.unique(y_train))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"   Device: {device}")
    
#     # ========================================================================
#     # STEP 2: VALIDATION SPLIT (Larger validation set for better monitoring)
#     # ========================================================================
#     X_train_split, X_val, y_train_split, y_val = train_test_split(
#         X_train, y_train,
#         test_size=0.20,  # 20% for validation (more stable estimates)
#         stratify=y_train,
#         random_state=42
#     )
    
#     print(f"\n✅ Data split:")
#     print(f"   Train: {X_train_split.shape[0]} samples")
#     print(f"   Val: {X_val.shape[0]} samples")
#     print(f"   Test: {X_test.shape[0]} samples")
#     sys.stdout.flush()
    
#     # Convert to tensors
#     X_train_split = torch.FloatTensor(X_train_split)
#     y_train_split = torch.LongTensor(y_train_split)
#     X_val = torch.FloatTensor(X_val)
#     y_val = torch.LongTensor(y_val)
#     X_test = torch.FloatTensor(X_test)
#     y_test = torch.LongTensor(y_test)
    
#     # ========================================================================
#     # STEP 3: DATALOADERS (SMALLER BATCH SIZE!)
#     # ========================================================================
#     batch_size = 32  # CRITICAL: Smaller batches for small dataset!
    
#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(X_train_split, y_train_split),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,
#         drop_last=True  # Drop incomplete batches
#     )
    
#     val_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(X_val, y_val),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0
#     )
    
#     test_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(X_test, y_test),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0
#     )
    
#     # ========================================================================
#     # STEP 4: BUILD FULL CTNET MODEL
#     # ========================================================================
#     print("\n Building FULL CTNet...")
#     sys.stdout.flush()
    
#     n_channels = X_train_split.shape[1]
#     n_timepoints = X_train_split.shape[2]
    
#     # FULL CTNet with optimized hyperparameters for small dataset
#     # model = CTNetLite(
#     #         n_channels=n_channels,
#     #         n_timepoints=n_timepoints,
#     #         n_classes=num_classes,
#     #         dropout=0.7  # Increased for better regularization
#     #     ).to(device)
#     model = CTNet(
#         n_channels=n_channels,
#         n_timepoints=n_timepoints,
#         n_classes=num_classes,
#         embed_dim=64,      # Smaller embedding (less parameters)
#         num_heads=4,       # Keep 4 heads
#         num_layers=1,      # CRITICAL: Only 1 transformer layer (less overfitting)
#         dropout=0.7        # CRITICAL: High dropout for small dataset!
#     ).to(device)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"✅ Model created:")
#     print(f"   Total parameters: {total_params:,}")
#     print(f"   Trainable: {trainable_params:,}")
#     print(f"   Samples per parameter: {X_train_split.shape[0] / total_params:.4f}")
#     print(f"   ⚠️  WARNING: Low ratio means high overfitting risk!")
#     sys.stdout.flush()
    
#     # ========================================================================
#     # STEP 5: TRAINING SETUP (CRITICAL HYPERPARAMETERS!)
#     # ========================================================================
#     # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps!
#     criterion = nn.CrossEntropyLoss()

    
#     # MUCH lower learning rate for transformers with small data
#     optimizer = optim.AdamW(
#         model.parameters(), 
#         lr=1e-4,           # CRITICAL: Very low LR
#         weight_decay=5e-3  # CRITICAL: Strong weight decay
#     )
    
#     # Aggressive learning rate reduction
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 
#         mode='max',        # Maximize validation accuracy
#         factor=0.5,        # Halve LR
#         patience=15,       # Wait 15 epochs
#         min_lr=1e-6,
#         verbose=True
#     )
    
#     num_epochs = 500  # Many epochs with strong regularization
#     best_val_acc = 0
#     patience = 100  # Early stopping patience
#     patience_counter = 0
    
#     history = {
#         'train_loss': [], 'train_acc': [], 
#         'val_loss': [], 'val_acc': [],
#         'lr': []
#     }
    
#     print("\n" + "="*80)
#     print("TRAINING CONFIGURATION")
#     print("="*80)
#     print(f"Model: CTNet FULL (embed_dim=64, num_layers=1)")
#     print(f"Epochs: {num_epochs}")
#     print(f"Batch size: {batch_size}")
#     print(f"Initial LR: 1e-4")
#     print(f"Weight decay: 5e-3 (STRONG)")
#     print(f"Dropout: 0.7 (VERY HIGH)")
#     print(f"Label smoothing: 0.1")
#     print(f"Scheduler: ReduceLROnPlateau")
#     print(f"Early stopping patience: {patience}")
#     print("="*80 + "\n")
#     sys.stdout.flush()
    
#     # ========================================================================
#     # STEP 6: TRAINING LOOP WITH DETAILED MONITORING
#     # ========================================================================
#     print("Starting training...")
#     sys.stdout.flush()
    
#     for epoch in range(num_epochs):
#         # TRAINING
#         model.train()
#         train_loss = 0
#         train_correct = 0
#         train_total = 0
        
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
            
#             optimizer.zero_grad()
#             preds, _ = model(xb)
#             loss = criterion(preds, yb)
#             loss.backward()
            
#             # Gradient clipping (important for transformers)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
#             optimizer.step()
            
#             train_loss += loss.item()
#             train_correct += (preds.argmax(1) == yb).sum().item()
#             train_total += yb.size(0)
        
#         avg_train_loss = train_loss / len(train_loader)
#         train_acc = 100 * train_correct / train_total
        
#         # VALIDATION
#         model.eval()
#         val_loss = 0
#         val_correct = 0
#         val_total = 0
        
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 preds, _ = model(xb)
#                 loss = criterion(preds, yb)
                
#                 val_loss += loss.item()
#                 val_correct += (preds.argmax(1) == yb).sum().item()
#                 val_total += yb.size(0)
        
#         avg_val_loss = val_loss / len(val_loader)
#         val_acc = 100 * val_correct / val_total
        
#         # Update learning rate based on validation performance
#         scheduler.step(val_acc)
        
#         current_lr = optimizer.param_groups[0]['lr']
#         history['train_loss'].append(avg_train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(avg_val_loss)
#         history['val_acc'].append(val_acc)
#         history['lr'].append(current_lr)
        
#         # Print progress
#         if (epoch + 1) % 5 == 0 or epoch < 10:
#             print(f"Epoch {epoch+1:03d}/{num_epochs} | "
#                   f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
#                   f"T_Loss: {avg_train_loss:.4f} | V_Loss: {avg_val_loss:.4f} | "
#                   f"LR: {current_lr:.6f}")
#             sys.stdout.flush()
        
#         # Save best model
#         if val_acc > best_val_acc:
#             improvement = val_acc - best_val_acc
#             best_val_acc = val_acc
#             patience_counter = 0
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#             }, 'ctnet_best_model.pth')
            
#             if (epoch + 1) % 5 == 0 or epoch < 10:
#                 print(f"    ✅ New best: {best_val_acc:.2f}% (+{improvement:.2f}%)")
#                 sys.stdout.flush()
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"\n🛑 Early stopping at epoch {epoch+1}")
#                 print(f"   Best validation: {best_val_acc:.2f}%")
#                 sys.stdout.flush()
#                 break
    
#     print(f"\n" + "="*80)
#     print(f"✅ Training complete!")
#     print(f"   Best validation: {best_val_acc:.2f}%")
#     print(f"   Total epochs: {epoch+1}")
#     print("="*80)
#     sys.stdout.flush()
    
#     # ========================================================================
#     # STEP 7: LOAD BEST MODEL AND EVALUATE
#     # ========================================================================
#     print("\nLoading best model...")
#     sys.stdout.flush()
    
#     checkpoint = torch.load('ctnet_best_model.pth')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     # Test evaluation
#     test_correct = 0
#     test_total = 0
#     all_preds = []
#     all_true = []
    
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             preds, _ = model(xb)
#             pred_labels = preds.argmax(1)
#             all_preds.extend(pred_labels.cpu().numpy())
#             all_true.extend(yb.cpu().numpy())
#             test_correct += (pred_labels == yb).sum().item()
#             test_total += yb.size(0)
    
#     test_acc = 100 * test_correct / test_total
    
#     print("\n" + "="*80)
#     print("FINAL TEST RESULTS")
#     print("="*80)
#     print(f"✅ Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
#     print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
#     print(f"   Generalization Gap: {abs(best_val_acc - test_acc):.2f}%")
#     print("="*80 + "\n")
#     sys.stdout.flush()
    
#     # ========================================================================
#     # STEP 8: DETAILED ANALYSIS
#     # ========================================================================
#     class_labels = [f"Class {i}" for i in range(num_classes)]
    
#     print("Classification Report:")
#     print(classification_report(all_true, all_preds, target_names=class_labels))
    
#     # Plot training curves
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     # Loss
#     axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
#     axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
#     axes[0, 0].set_xlabel('Epoch')
#     axes[0, 0].set_ylabel('Loss')
#     axes[0, 0].set_title("Training & Validation Loss")
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # Accuracy
#     axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
#     axes[0, 1].plot(history['val_acc'], label='Val', linewidth=2)
#     axes[0, 1].axhline(20, color='r', linestyle='--', label='Random', alpha=0.5)
#     axes[0, 1].set_xlabel('Epoch')
#     axes[0, 1].set_ylabel('Accuracy (%)')
#     axes[0, 1].set_title("Training & Validation Accuracy")
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3)
    
#     # Learning rate
#     axes[1, 0].plot(history['lr'], linewidth=2, color='green')
#     axes[1, 0].set_xlabel('Epoch')
#     axes[1, 0].set_ylabel('Learning Rate')
#     axes[1, 0].set_title("Learning Rate Schedule")
#     axes[1, 0].set_yscale('log')
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # Overfitting gap
#     gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
#     axes[1, 1].plot(gap, linewidth=2, color='red')
#     axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
#     axes[1, 1].set_xlabel('Epoch')
#     axes[1, 1].set_ylabel('Train - Val Accuracy (%)')
#     axes[1, 1].set_title("Overfitting Monitor")
#     axes[1, 1].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     # Ensure output directory exists
#     os.makedirs('output', exist_ok=True)
#     plt.savefig('./output/ctnet_full_training.png', dpi=300)
#     plt.show()
    
#     # Confusion matrix
#     cm = confusion_matrix(all_true, all_preds)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(f"CTNet Full - Test Accuracy: {test_acc:.2f}%")
#     plt.colorbar()
#     plt.xticks(range(num_classes), class_labels, rotation=45)
#     plt.yticks(range(num_classes), class_labels)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
    
#     for i in range(num_classes):
#         for j in range(num_classes):
#             plt.text(j, i, cm[i, j], ha="center", va="center",
#                     color="white" if cm[i, j] > cm.max()/2 else "black")
    
#     plt.tight_layout()
#     os.makedirs('output', exist_ok=True)
#     plt.savefig('./output/ctnet_full_confusion.png', dpi=300)
#     plt.show()
    
#     print("\n✅ All visualizations saved!")
    
#     return test_acc, best_val_acc


# # ============================================================================
# # MAIN
# # ============================================================================
# if __name__ == '__main__':
#     test_acc, val_acc = train_ctnet()
    
#     print("\n" + "="*80)
#     print("📊 FINAL SUMMARY - CTNet Full Model")
#     print("="*80)
#     print(f"Best Validation: {val_acc:.2f}%")
#     print(f"Final Test: {test_acc:.2f}%")
#     print(f"")
#     print(f"Note: With only 2027 training samples and ~100K+ parameters,")
#     print(f"      this model is severely data-limited. Expected range:")
#     print(f"      Validation: 40-60%, Test: 35-55%")
#     print("="*80)


# train_ctnet.py
# ============================================================================
# CTNet FULL MODEL Training - Optimized for 8ch, 250Hz, 1s (8 x 250)
# No augmentation in this script.
# ============================================================================

# ctnet_training.py
# ============================================================================
# CTNet FULL MODEL Training
# Updated for: Python 3.7, Windows, new CTNet API (paper-aligned)
# Supports: 8-channel, 250 Hz, 1-second EEG trials (8 x 250)
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

# Import updated models
from ctnet_model import CTNet, CTNetLite


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_ctnet():

    print("=" * 80)
    print("🚀 CTNet Training (8 ch, 250 Hz, 1s Window)")
    print("=" * 80)

    # -----------------------------------------------
    # STEP 1 — LOAD DATA
    # -----------------------------------------------
    data_dir = ""  # put dataset folder path if needed

    print("\n📂 Loading preprocessed EEG data...")
    X_train = np.load(os.path.join(data_dir, "X_train_ctnet.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train_ctnet.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_ctnet.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test_ctnet.npy"))

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    print(f"Classes: {np.unique(y_train)}")

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
    # STEP 2 — TRAIN/VAL SPLIT
    # -----------------------------------------------
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
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
    batch_size = 32  # still small relative to dataset, but stabilizes gradients

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
    print("\n Building CTNet model...")

    n_channels = X_train_split.shape[1]
    n_timepoints = X_train_split.shape[2]

    # Auto-switch: if dataset extremely small, use CTNetLite
    use_lite = X_train_split.shape[0] < 1500

    if use_lite:
        print("Using CTNetLite (small dataset mode).")
        model = CTNetLite(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=num_classes,
            sampling_rate=250,
            dropout_conv=0.3,
            classifier_dropout=0.3
        ).to(device)
    else:
        print("Using full CTNet (paper-aligned).")
        model = CTNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=num_classes,
            sampling_rate=250,
            F1=8,
            D=2,
            F2=16,
            Kc1=None,      # uses Fs/4 rule
            Kc2=16,
            P1=8,
            P2=2,
            dropout_conv=0.5,
            n_heads=2,
            n_layers=4,    # You can increase to 6 if GPU allows
            ff_mult=4,
            dropout_transformer=0.1,
            use_positional_encoding=True,
            classifier_dropout=0.3
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # -----------------------------------------------
    # STEP 5 — OPTIMIZER + SCHEDULER
    # -----------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-3
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.7,
        patience=25,
        min_lr=5e-6,
        verbose=True
    )

    # -----------------------------------------------
    # STEP 6 — TRAINING LOOP
    # -----------------------------------------------
    num_epochs = 500
    patience = 200
    patience_counter = 0
    best_val_acc = 0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
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

            out = model(xb)
            preds = out[0] if isinstance(out, tuple) else out

            loss = criterion(preds, yb)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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

        scheduler.step(val_acc)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(
                f"Epoch {epoch+1:03d}/{num_epochs} | "
                f"Train {train_acc:5.1f}% | Val {val_acc:5.1f}% | "
                f"LR {optimizer.param_groups[0]['lr']:.6f}"
            )

        # Best model saving + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                {"model_state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch,
                 "val_acc": val_acc},
                "ctnet_best_model.pth"
            )
            print(f"   ✅ New best model saved! Val Acc = {best_val_acc:.2f}%")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\n🛑 Early Stopping Triggered!")
                break

    print("\nTraining Completed.")
    print(f"Best Validation Accuracy = {best_val_acc:.2f}%")

    # -----------------------------------------------
    # STEP 7 — TEST EVALUATION
    # -----------------------------------------------
    print("\nLoading best checkpoint...")
    ckpt = torch.load("ctnet_best_model.pth", map_location=device)
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

    print("\n==============================")
    print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")
    print(f"Best Val Accuracy:      {best_val_acc:.2f}%")
    print("==============================\n")

    # -----------------------------------------------
    # STEP 8 — PLOTS
    # -----------------------------------------------
    os.makedirs("output", exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["val_acc"], label="Val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(history["lr"])
    axes[1, 0].set_title("Learning Rate")

    gap = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    axes[1, 1].plot(gap)
    axes[1, 1].set_title("Overfitting Gap (Train - Val)")

    plt.tight_layout()
    plt.savefig("./output/ctnet_training_curves.png", dpi=300)
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"CTNet Confusion Matrix (Test Acc {test_acc:.2f}%)")
    plt.colorbar()

    class_labels = [f"Class {i}" for i in range(num_classes)]
    plt.xticks(range(num_classes), class_labels, rotation=45)
    plt.yticks(range(num_classes), class_labels)

    # values inside matrix
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    plt.savefig("./output/ctnet_confusion_matrix.png", dpi=300)
    plt.show()

    return test_acc, best_val_acc


# ============================================================================
# MAIN ENTRY (REQUIRED FOR WINDOWS)
# ============================================================================
if __name__ == "__main__":
    test_acc, val_acc = train_ctnet()

    print("\n==============================")
    print("📊 FINAL SUMMARY")
    print("==============================")
    print(f"Best Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Test Accuracy:      {test_acc:.2f}%")
    print("==============================")
