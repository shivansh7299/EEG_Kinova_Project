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

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from training_artifacts import build_run_paths, save_classification_outputs, save_confusion_matrix_figure, save_json

PROJECT_ROOT = Path(__file__).resolve().parent

# Force flush output (Windows fix)
sys.stdout.flush()

# Import updated models
from ctnet_model import CTNet, CTNetLite


# ============================================================================
# DATA AUGMENTATION FOR EEG (DISABLED - NOT USED FOR REAL-TIME INFERENCE)
# ============================================================================
# Augmentation code kept for reference but not used during training
# to ensure model works with raw, unaugmented data in real-time scenarios
#
# class EEGAugmentation:
#     """Data augmentation for EEG signals to increase dataset size."""
#     
#     @staticmethod
#     def add_gaussian_noise(x, noise_factor=0.05):
#         noise = torch.randn_like(x) * noise_factor * x.std()
#         return x + noise
#     
#     @staticmethod
#     def time_shift(x, shift_max=10):
#         shift = random.randint(-shift_max, shift_max)
#         if shift == 0:
#             return x
#         return torch.roll(x, shifts=shift, dims=-1)
#     
#     @staticmethod
#     def channel_dropout(x, p=0.1):
#         mask = torch.ones_like(x)
#         n_channels = x.shape[1]
#         n_drop = int(n_channels * p)
#         if n_drop > 0:
#             channels_to_drop = random.sample(range(n_channels), n_drop)
#             mask[:, channels_to_drop, :] = 0
#         return x * mask


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_ctnet(data_dir="", subject_id=None):

    print("=" * 80)
    print("🚀 CTNet Training (8 ch, 250 Hz, 1s Window)")
    print("=" * 80)

    # -----------------------------------------------
    # STEP 1 — LOAD DATA
    # -----------------------------------------------
    # NOTE: Data should be preprocessed using preprocessing_3_class_CTNet.ipynb
    # Expected format: (N, C, T) where N=samples, C=8 channels, T=250 timepoints
    # The preprocessing notebook supports two methods:
    #   - 'automatic': Uses data_curation.py pipeline (uniform_len -> filter -> zscore_normalize)
    #   - 'trial_index': Uses original notebook pipeline (filter -> baseline_norm -> StandardScaler)
    # Both methods produce the same output format: (N, 8, 250)

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
    results_paths = build_run_paths(
        PROJECT_ROOT,
        "CTNet",
        data_dir,
        num_classes=num_classes,
        subject_no=subject_id,
    )
    figures_dir = results_paths["figures_dir"]
    metrics_dir = results_paths["metrics_dir"]
    checkpoints_dir = results_paths["checkpoints_dir"]
    run_dir = results_paths["run_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Results directory: {run_dir}")

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
    batch_size = 16  # Smaller batch for more gradient updates per epoch

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

    # Use full CTNet with reduced layers for better capacity (even for small datasets)
    # CTNetLite is too simple and underfits
    print("Using Lite CTNet with optimized architecture for small dataset.")
    model = CTNet(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=num_classes,
        sampling_rate=250,
        F1=8,
        D=2,
        F2=16,
        Kc1=None,
        Kc2=16,
        P1=8,
        P2=2,
        dropout_conv=0.5,  # Moderate dropout
        n_heads=2,
        n_layers=2,  # Reduced from 6 to prevent overfitting, but more than Lite
        ff_mult=4,
        dropout_transformer=0.1,
        use_positional_encoding=True,
        classifier_dropout=0.4
    ).to(device)
    # model = CTNetLite(
    #         n_channels=n_channels,
    #         n_timepoints=n_timepoints,
    #         n_classes=num_classes,
    #         dropout_conv=0.7  # Validation Accuracy: 42.55% Test Accuracy:      29.17%
    #         #dropout_conv=0.5 #-  Validation Accuracy = 57.45% Test Accuracy =  47%
    #         # Final Test Accuracy: 50.00% Best Val Accuracy:      61.70%
    #     ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # -----------------------------------------------
    # STEP 5 — OPTIMIZER + SCHEDULER
    # -----------------------------------------------
    # Add class weights to handle any imbalance
    from collections import Counter
    train_counts = Counter(y_train_split.numpy())
    total = sum(train_counts.values())
    class_weights = torch.FloatTensor([
        total / (len(train_counts) * train_counts[i]) for i in sorted(train_counts.keys())
    ]).to(device)
    print(f"Class weights: {class_weights}")
    
    # Use Focal Loss to focus on hard examples and prevent class collapse
    # Focal loss helps when model ignores certain classes
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
    
    # Use focal loss to force learning of all classes
    # criterion = FocalLoss(class_weights, alpha=0.25, gamma=1.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -----------------------------------------------
    # STEP 5 — LEARNING RATE SEARCH
    # -----------------------------------------------
    # Learning rates to test
    # learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    learning_rates = [1e-3]
    num_epochs = 300  # Fixed 300 epochs per learning rate
    
    # Store results for each learning rate
    lr_results = {}  # {lr: {'best_val_acc': float, 'best_epoch': int, 'model_path': str}}
    
    print("\n" + "="*80)
    print("🔍 LEARNING RATE SEARCH")
    print("="*80)
    print(f"Testing {len(learning_rates)} learning rates: {learning_rates}")
    print(f"Epochs per LR: {num_epochs}")
    print("="*80 + "\n")

    # -----------------------------------------------
    # STEP 6 — TRAIN FOR EACH LEARNING RATE
    # -----------------------------------------------
    for lr_idx, lr in enumerate(learning_rates):
        print("\n" + "="*80)
        print(f"📊 Learning Rate {lr_idx+1}/{len(learning_rates)}: {lr:.6f}")
        print("="*80)
        
        # Reinitialize model for each LR
        model = CTNetLite(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=num_classes,
            dropout_conv=0.5
        ).to(device)
        
        # Optimizer with current learning rate
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=3e-4
        )
        
        # Learning rate scheduler (ReduceLROnPlateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',        # Maximize validation accuracy
            factor=0.5,        # Halve LR when plateau
            patience=30,       # Wait 30 epochs
            min_lr=1e-6,       # Minimum learning rate
            verbose=False      # Don't print scheduler messages
        )
        
        best_val_acc_lr = 0
        best_epoch_lr = 0
        patience_counter = 0
        patience = 100  # Early stopping patience
        
        print(f"🏁 Training for {num_epochs} epochs...\n")
        
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

            # Update scheduler
            scheduler.step(val_acc)

            # Track best model for this LR
            if val_acc > best_val_acc_lr:
                best_val_acc_lr = val_acc
                best_epoch_lr = epoch
                patience_counter = 0
                
                # Save model for this specific LR
                model_path = checkpoints_dir / f"ctnet_best_model_lr_{lr:.6f}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "lr": lr
                    },
                    str(model_path)
                )
                
                if (epoch + 1) % 25 == 0 or epoch < 5:
                    print(f"   Epoch {epoch+1:03d}/{num_epochs} | "
                          f"Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}% | "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                          f"✅ Best: {best_val_acc_lr:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n   ⛔ Early stopping at epoch {epoch+1}")
                    break

        # Store results for this LR
        lr_results[lr] = {
            'best_val_acc': best_val_acc_lr,
            'best_epoch': best_epoch_lr,
            'model_path': str(checkpoints_dir / f"ctnet_best_model_lr_{lr:.6f}.pth")
        }
        
        print(f"\n✅ LR {lr:.6f} Complete:")
        print(f"   Best Val Accuracy: {best_val_acc_lr:.2f}% @ Epoch {best_epoch_lr+1}")
        print(f"   Model saved: {lr_results[lr]['model_path']}")

    # -----------------------------------------------
    # STEP 7 — FIND BEST LEARNING RATE
    # -----------------------------------------------
    print("\n" + "="*80)
    print("📊 LEARNING RATE SEARCH RESULTS")
    print("="*80)
    
    best_lr = max(lr_results.keys(), key=lambda k: lr_results[k]['best_val_acc'])
    best_overall_acc = lr_results[best_lr]['best_val_acc']
    
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Learning Rate':<15} {'Best Val Acc':<15} {'Best Epoch':<12} {'Model Path':<40}")
    print("-" * 80)
    for lr in sorted(lr_results.keys()):
        result = lr_results[lr]
        marker = " ⭐ BEST" if lr == best_lr else ""
        print(f"{lr:<15.6f} {result['best_val_acc']:<15.2f}% {result['best_epoch']+1:<12} {result['model_path']:<40}{marker}")
    print("-" * 80)
    print(f"\n🏆 Best Learning Rate: {best_lr:.6f}")
    print(f"   Best Validation Accuracy: {best_overall_acc:.2f}%")
    print(f"   Best Model: {lr_results[best_lr]['model_path']}")
    print("="*80)

    # -----------------------------------------------
    # STEP 8 — TEST EVALUATION (ONLY BEST MODEL)
    # -----------------------------------------------
    print("\n" + "="*80)
    print("🧪 TESTING BEST MODEL")
    print("="*80)
    print(f"Loading best model: {lr_results[best_lr]['model_path']}")
    print(f"Learning Rate: {best_lr:.6f}")
    print(f"Validation Accuracy: {best_overall_acc:.2f}%")
    print("="*80)
    
    # Load best model
    ckpt = torch.load(lr_results[best_lr]['model_path'], map_location=device)
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

    print("\n" + "="*80)
    print("🎯 FINAL TEST RESULTS")
    print("="*80)
    print(f"Best Learning Rate:     {best_lr:.6f}")
    print(f"Best Val Accuracy:      {best_overall_acc:.2f}%")
    print(f"Final Test Accuracy:    {test_acc:.2f}%")
    print("="*80 + "\n")

    # -----------------------------------------------
    # STEP 9 — PLOTS
    # -----------------------------------------------
    save_json(metrics_dir / "ctnet_lr_search_results.json", {str(k): v for k, v in lr_results.items()})

    # Learning Rate Search Results Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Validation Accuracy vs Learning Rate
    lrs = sorted(lr_results.keys())
    val_accs = [lr_results[lr]['best_val_acc'] for lr in lrs]
    
    axes[0].plot(lrs, val_accs, 'o-', linewidth=2, markersize=8, label='Best Val Acc')
    axes[0].axvline(best_lr, color='r', linestyle='--', linewidth=2, label=f'Best LR: {best_lr:.6f}')
    axes[0].set_xlabel('Learning Rate', fontsize=12)
    axes[0].set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Learning Rate Search Results', fontsize=14, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add value labels on points
    for lr, acc in zip(lrs, val_accs):
        axes[0].annotate(f'{acc:.1f}%', (lr, acc), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Bar chart comparison
    colors = ['red' if lr == best_lr else 'steelblue' for lr in lrs]
    bars = axes[1].bar(range(len(lrs)), val_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_xticks(range(len(lrs)))
    axes[1].set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45, ha='right')
    axes[1].set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Learning Rate Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, val_accs)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / "ctnet_lr_search_results.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(figures_dir / "ctnet_confusion_matrix.png", dpi=300)
    plt.show()

    save_classification_outputs(metrics_dir, all_true, all_preds, class_labels, "ctnet")
    save_confusion_matrix_figure(
        figures_dir / "ctnet_confusion_matrix_clean.png",
        cm,
        class_labels,
        f"CTNet Confusion Matrix (Test Acc {test_acc:.2f}%)",
    )

    return test_acc, best_overall_acc, best_lr, lr_results


# ============================================================================
# MAIN ENTRY (REQUIRED FOR WINDOWS)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTNet and save subject-aware results.")
    parser.add_argument("--data-dir", default="", help="Directory containing X_train_ctnet.npy and related files")
    parser.add_argument("--subject-id", type=int, default=None, help="Optional subject id used for results folder naming")
    args = parser.parse_args()

    test_acc, val_acc, best_lr, lr_results = train_ctnet(data_dir=args.data_dir, subject_id=args.subject_id)

    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"Best Learning Rate:        {best_lr:.6f}")
    print(f"Best Validation Accuracy:  {val_acc:.2f}%")
    print(f"Final Test Accuracy:       {test_acc:.2f}%")
    print("="*80)
