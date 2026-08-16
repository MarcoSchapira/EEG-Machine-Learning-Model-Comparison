"""
train.py
Trains the dynamically selected EMG Model with granular timing, accuracy tracking, 
data recording, and confusion matrix visualization.
"""
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import get_dataloaders, CLASS_TO_IDX
from model import get_model

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training Mode: {args.task.upper()} | Model: {args.model_type.upper()}")
    
    apply_filter = not args.no_filter
    # Unpack the 3 variables returned by your updated get_dataloaders
    train_loader, val_loader, num_classes = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers, 
        electrodes=args.electrodes, scaling=args.scaling, feature_ext=args.feature_ext,
        exclude_rest=args.exclude_rest, three_classes=args.three_classes, 
        apply_filter=apply_filter, highpass_cutoff=args.highpass_cutoff
    )
    
    num_channels = len(args.electrodes)
    model = get_model(args.model_type, num_channels, args.feature_ext, num_classes=num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.HuberLoss()
    lamda = 100.0 
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    best_val_metric = float('inf')
    patience, epochs_no_improve = 4, 0
    
    history = {
        'train_total': [], 'train_cls': [], 'train_reg': [], 'train_acc': [],
        'val_total': [], 'val_cls': [], 'val_reg': [], 'val_acc': [],
        'epoch_times': []
    }
    
    best_cm_labels = []
    best_cm_preds = []
    
    total_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        train_start_time = time.time()
        model.train()
        
        run_loss_total, run_loss_cls, run_loss_reg = 0.0, 0.0, 0.0
        train_correct, train_total_samples = 0, 0
        
        for inputs, labels_cls, labels_reg in train_loader:
            inputs, labels_cls, labels_reg = inputs.to(device), labels_cls.to(device), labels_reg.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                out_cls, out_reg = model(inputs, apply_gating=False) 
                
                loss_cls = criterion_class(out_cls, labels_cls)
                loss_reg = criterion_reg(out_reg, labels_reg)
                loss_total = loss_cls + (lamda * loss_reg) if args.task == 'multi' else loss_cls
            
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            
            run_loss_total += loss_total.item()
            run_loss_cls += loss_cls.item()
            run_loss_reg += loss_reg.item()
            
            preds = torch.argmax(out_cls, dim=1)
            train_correct += (preds == labels_cls).sum().item()
            train_total_samples += labels_cls.size(0)
            
        train_duration = time.time() - train_start_time
        
        num_train_batches = len(train_loader)
        train_acc = (train_correct / train_total_samples) * 100
        
        history['train_total'].append(float(run_loss_total / num_train_batches))
        history['train_cls'].append(float(run_loss_cls / num_train_batches))
        history['train_reg'].append(float(run_loss_reg / num_train_batches))
        history['train_acc'].append(float(train_acc))
        
        # --- Validation Phase ---
        val_start_time = time.time()
        model.eval()
        
        val_loss_total, val_loss_cls, val_loss_reg = 0.0, 0.0, 0.0
        val_correct, val_total_samples = 0, 0
        
        val_epoch_labels = []
        val_epoch_preds = []
        
        with torch.no_grad():
            for inputs, labels_cls, labels_reg in val_loader:
                inputs, labels_cls, labels_reg = inputs.to(device), labels_cls.to(device), labels_reg.to(device)
                
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    out_cls, out_reg = model(inputs, apply_gating=True)
                    loss_cls = criterion_class(out_cls, labels_cls)
                    loss_reg = criterion_reg(out_reg, labels_reg)
                    batch_val_total = loss_cls + (lamda * loss_reg) if args.task == 'multi' else loss_cls
                        
                val_loss_total += batch_val_total.item()
                val_loss_cls += loss_cls.item()
                val_loss_reg += loss_reg.item()
                
                preds = torch.argmax(out_cls, dim=1)
                val_correct += (preds == labels_cls).sum().item()
                val_total_samples += labels_cls.size(0)
                
                val_epoch_labels.extend(labels_cls.cpu().numpy().tolist())
                val_epoch_preds.extend(preds.cpu().numpy().tolist())
                
        val_duration = time.time() - val_start_time
        
        num_val_batches = len(val_loader)
        avg_val_total = val_loss_total / num_val_batches
        avg_val_cls = val_loss_cls / num_val_batches
        avg_val_reg = val_loss_reg / num_val_batches
        val_acc = (val_correct / val_total_samples) * 100
        
        history['val_total'].append(float(avg_val_total))
        history['val_cls'].append(float(avg_val_cls))
        history['val_reg'].append(float(avg_val_reg))
        history['val_acc'].append(float(val_acc))
        
        epoch_duration = time.time() - epoch_start_time
        history['epoch_times'].append(float(epoch_duration))
        
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Time: {epoch_duration:.2f}s")
        print(f"  Train -> Total: {history['train_total'][-1]:.4f} | Cls: {history['train_cls'][-1]:.4f} | Reg: {history['train_reg'][-1]:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   -> Total: {avg_val_total:.4f} | Cls: {avg_val_cls:.4f} | Reg: {avg_val_reg:.4f} | Acc: {val_acc:.2f}%")
        
        # --- Save Best Model & Early Stopping ---
        current_metric = avg_val_cls if args.task == 'classification' else avg_val_total
        
        if current_metric < best_val_metric:
            best_val_metric = current_metric
            epochs_no_improve = 0
            best_cm_labels = val_epoch_labels
            best_cm_preds = val_epoch_preds
            save_name = args.model_path.replace('.pth', f'_{args.model_type}_{args.task}.pth')
            torch.save(model.state_dict(), save_name)
            print(f"  >>> Best model saved to {save_name}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("\nEarly stopping triggered. Halting training.")
                break
            
    total_time = time.time() - total_start_time
    
    # --- Generate Confusion Matrix ---
    cm = confusion_matrix(best_cm_labels, best_cm_preds)
    plt.figure(figsize=(12, 10))
    # Note: To dynamically match the active class names without Rest, you can extract it from your dataset module similar to test.py
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {args.model_type.upper()}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_plot_name = f"confusion_matrix_{args.model_type}_{args.task}.png"
    plt.savefig(cm_plot_name)
    
    # --- Record Data to JSON ---
    final_data = {
        "model_type": args.model_type,
        "task": args.task,
        "epochs_run": epoch + 1,
        "best_val_metric": float(best_val_metric),
        "total_training_time_s": float(total_time),
        "history": history,
        "confusion_matrix": cm.tolist()
    }
    
    json_name = f"training_data_{args.model_type}_{args.task}.json"
    with open(json_name, "w") as f:
        json.dump(final_data, f, indent=4)

    print("\n--- Training Statistics ---")
    print(f"Total Training Time: {total_time:.2f}s")
    print(f"All epoch data and confusion matrix saved to {json_name} and {cm_plot_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--task", type=str, choices=['multi', 'classification'], default='multi')
    parser.add_argument("--model_type", type=str, choices=['adaptive', 'multiscale_cnn', 'resnet_spectrogram', 'tcn_lstm'], default='adaptive')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--electrodes", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--scaling", type=str, choices=['none', 'minmax', 'zscore', 'constant'], default='constant')
    parser.add_argument("--feature_ext", type=str, choices=['none', 'pca', 'td'], default='none')
    parser.add_argument("--exclude_rest", action="store_true", help="Drops the 'Rest' state and tests on 11 classes.")
    parser.add_argument("--no_filter", action="store_true", help="Disable the high-pass butterworth filter")
    parser.add_argument("--highpass_cutoff", type=float, default=20.0, help="Cutoff frequency for the filter")
    parser.add_argument("--three_classes", action="store_true", help="Restrict dataset to Arm-reaching: Backward, Arm-reaching: Left, and Hand-grasping: Ball.")
    
    args = parser.parse_args()
    train_model(args)