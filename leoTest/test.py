"""
test.py
Evaluates the trained model on the test dataset partition,
measuring inference speed, calculating confusion matrices, and recording statistics.
Supports dynamic class reduction via --exclude_rest.
"""
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import get_dataloaders, get_class_to_idx
from model import get_model

def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device} | Model: {args.model_type.upper()}")
    
    apply_filter = not args.no_filter
    try:
        _, test_loader, num_classes = get_dataloaders(
            args.data_root, 
            args.batch_size, 
            args.num_workers, 
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
    
    inference_times = []
    all_labels = []
    all_preds = []
    
    total_start = time.time()
    
    print("Starting inference evaluation...")
    with torch.no_grad():
        for inputs, labels_cls, _ in test_loader:
            inputs = inputs.to(device)
            
            start_inf = time.time()
            out_cls, _ = model(inputs, apply_gating=True)
            end_inf = time.time()
            
            time_per_signal = (end_inf - start_inf) / inputs.size(0)
            inference_times.extend([time_per_signal] * inputs.size(0))
            
            preds = torch.argmax(out_cls, dim=1)
            all_labels.extend(labels_cls.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            
    total_end = time.time()
    total_time = total_end - total_start
    
    if not inference_times:
        print("Error: No inference data collected. Check the test dataset.")
        exit(1)
        
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save statistics and matrix to JSON
    test_results = {
        "model_type": args.model_type,
        "total_testing_time_s": float(total_time),
        "inference_min_ms": float(np.min(inference_times) * 1000),
        "inference_max_ms": float(np.max(inference_times) * 1000),
        "inference_mean_ms": float(np.mean(inference_times) * 1000),
        "inference_std_ms": float(np.std(inference_times) * 1000),
        "confusion_matrix": cm.tolist()
    }
    
    json_name = f"test_results_{args.model_type}.json"
    with open(json_name, "w") as f:
        json.dump(test_results, f, indent=4)
        
    # Plot Confusion Matrix
    active_class_map = get_class_to_idx(args.exclude_rest, args.three_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(active_class_map.keys()), 
                yticklabels=list(active_class_map.keys()))
    plt.title(f'Test Confusion Matrix - {args.model_type.upper()}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"test_confusion_matrix_{args.model_type}.png")

    print("\n--- Testing Statistics ---")
    print(f"Total Testing Time: {total_time:.4f}s")
    print(f"Inference Speed Mean: {np.mean(inference_times)*1000:.4f} ms/signal")
    print(f"Data saved to {json_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=['adaptive', 'multiscale_cnn', 'resnet_spectrogram', 'tcn_lstm'], default='adaptive')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--electrodes", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--scaling", type=str, choices=['none', 'minmax', 'zscore', 'constant'], default='constant')
    parser.add_argument("--feature_ext", type=str, choices=['none', 'pca', 'td'], default='none')
    parser.add_argument("--exclude_rest", action="store_true", help="Drops the 'Rest' state and tests on 11 classes.")
    parser.add_argument("--no_filter", action="store_true", help="Disable the high-pass butterworth filter")
    parser.add_argument("--highpass_cutoff", type=float, default=20.0, help="Cutoff frequency for the filter")
    parser.add_argument("--three_classes", action="store_true", help="Restrict dataset to Arm-reaching: Backward, Arm-reaching: Left, and Hand-grasping: Ball.")
    
    args = parser.parse_args()
    test_model(args)