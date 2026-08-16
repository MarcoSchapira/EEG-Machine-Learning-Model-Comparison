"""
visualize.py
Extracts overall R^2 metrics and produces detailed visualizations
for the best 4 and worst 4 prediction windows using batched inference for speed.
Supports dynamic class reduction via --exclude_rest.
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from dataset import get_dataloaders, get_class_to_idx
from model import get_model

def calculate_metrics(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[0]):
        try:
            scores.append(r2_score(y_true[i], y_pred[i]))
        except ValueError:
            scores.append(0.0)
    return np.array(scores)

def visualize_results(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualizing on device: {device} | Model: {args.model_type.upper()}")
    
    apply_filter = not args.no_filter
    try:
        _, test_loader, num_classes = get_dataloaders(
            args.data_root, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            electrodes=args.electrodes,
            scaling=args.scaling,
            feature_ext=args.feature_ext,
            exclude_rest=args.exclude_rest,
            three_classes=args.three_classes,
            apply_filter=apply_filter, 
            highpass_cutoff=args.highpass_cutoff
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    num_channels = len(args.electrodes)
    model = get_model(args.model_type, num_channels, args.feature_ext, num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(">>> Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)
        
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    all_signals = []
    all_classes = [] # New list to store class labels
    
    print(f"Running batched inference (Batch Size: {args.batch_size})...")
    with torch.no_grad():
        for inputs, labels_cls, labels_reg in test_loader: # Capture labels_cls here
            inputs = inputs.to(device)
            labels_reg = labels_reg.to(device)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                _, out_reg = model(inputs, apply_gating=True)
            
            all_signals.append(inputs.cpu().numpy())
            all_y_true.append(labels_reg.cpu().numpy())
            all_y_pred.append(out_reg.cpu().numpy())
            all_classes.append(labels_cls.cpu().numpy()) # Store the batch classes
            
    print("Concatenating batches...")
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)
    all_signals = np.concatenate(all_signals, axis=0)
    all_classes = np.concatenate(all_classes, axis=0) # Flatten the classes
    
    r2_scores = calculate_metrics(all_y_true, all_y_pred)
    
    sorted_indices = np.argsort(r2_scores)
    worst_4_idx = sorted_indices[:4]
    best_4_idx = sorted_indices[-4:][::-1]
    
    print("\n--- Performance Metrics (R^2) ---")
    print(f"All Results   - Min: {np.min(r2_scores):.4f}, Max: {np.max(r2_scores):.4f}, Mean: {np.mean(r2_scores):.4f}, StDev: {np.std(r2_scores):.4f}")

    # Map the integers back to their string names using your CLI flags
    active_class_map = get_class_to_idx(args.exclude_rest, args.three_classes)
    idx_to_class = {v: k for k, v in active_class_map.items()}

    def plot_overlays(indices, title_prefix):
        fig, axes = plt.subplots(4, 1, figsize=(10, 16))
        fig.suptitle(f"{title_prefix} Predictions: EMG Signal & Joint Space Overlay")
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            emg_ref = all_signals[idx][0, :] 
            time_axis = np.linspace(0, 0.2, len(emg_ref)) 
            
            ax.plot(time_axis, emg_ref, color='gray', alpha=0.5, label='EMG Window (Ch 1)')
            
            dof_axis = np.linspace(0.05, 0.15, 7)
            width = 0.005
            
            ax.bar(dof_axis - width/2, all_y_true[idx], width, label='True Joint State', color='blue')
            ax.bar(dof_axis + width/2, all_y_pred[idx], width, label='Pred Joint State', color='red')
            
            # Look up the string name for the current sample
            class_name = idx_to_class[all_classes[idx]]
            ax.set_title(f"Sample {idx} ({class_name}) | R^2 Score: {r2_scores[idx]:.4f}")
            
            ax.set_ylabel("Amplitude / Scaled Angle")
            ax.set_xlabel("Time (s) within Window")
            if i == 0:
                ax.legend(loc='upper right')
                
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        if not os.path.exists(args.image_root):
            os.makedirs(args.image_root)
            
        plt.savefig(f"{args.image_root}/{title_prefix.lower().replace(' ', '_')}_{args.model_type}_results.png")
        print(f"Saved {title_prefix} visualizations.")

    plot_overlays(best_4_idx, "Top 4 Best")
    plot_overlays(worst_4_idx, "Bottom 4 Worst")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=['adaptive', 'multiscale_cnn', 'resnet_spectrogram', 'tcn_lstm'], default='adaptive')
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--electrodes", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--scaling", type=str, choices=['none', 'minmax', 'zscore', 'constant'], default='constant')
    parser.add_argument("--feature_ext", type=str, choices=['none', 'pca', 'td'], default='none')
    parser.add_argument("--exclude_rest", action="store_true", help="Drops the 'Rest' state and tests on 11 classes.")
    parser.add_argument("--no_filter", action="store_true", help="Disable the high-pass butterworth filter")
    parser.add_argument("--highpass_cutoff", type=float, default=20.0, help="Cutoff frequency for the filter")
    parser.add_argument("--three_classes", action="store_true", help="Restrict dataset to Arm-reaching: Backward, Arm-reaching: Left, and Hand-grasping: Ball.")
    args = parser.parse_args()
    
    visualize_results(args)