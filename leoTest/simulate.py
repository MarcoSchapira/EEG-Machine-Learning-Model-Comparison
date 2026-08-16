"""
simulate.py
Simulates a full 4-second physical trial for a selected movement on the MyCobot280.
Randomly pulls a trial from the test set, runs the chronological sequence of sliding windows 
through the model, and plots the True vs Predicted 7-DOF continuous joint trajectories.
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import from your pipeline
from dataset import EMGDataset, get_active_classes, get_class_to_idx
from model import get_model

def simulate_trial(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Simulation Device: {device} | Model: {args.model_type.upper()}")
    
    # 1. Resolve Test Directory
    subdirs = [os.path.join(args.data_root, d) for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    test_dir = next((d for d in subdirs if 'test' in d.lower()), None)
    if not test_dir:
        raise FileNotFoundError(f"Could not find a 'test' subfolder inside {args.data_root}.")

    # 2. Initialize the Test Dataset directly (No DataLoader so we can manually index chronological windows)
    apply_filter = not args.no_filter
    test_dataset = EMGDataset(
        test_dir, 
        is_train=False, 
        electrodes=args.electrodes, 
        scaling=args.scaling, 
        feature_ext=args.feature_ext, 
        exclude_rest=args.exclude_rest, 
        apply_filter=apply_filter, 
        highpass_cutoff=args.highpass_cutoff
    )
    
    # 3. Validate and Map the Target Movement
    active_classes = get_active_classes(args.exclude_rest)
    active_class_map = get_class_to_idx(args.exclude_rest)
    
    if args.movement not in active_class_map:
        print(f"\nError: Movement '{args.movement}' is not valid or is excluded.")
        print(f"Available movements: {list(active_class_map.keys())}")
        exit(1)
        
    target_class_idx = active_class_map[args.movement]
    
    # 4. Find all trials matching the target movement and select one randomly
    matching_trials = [i for i, label in enumerate(test_dataset.labels_class) if label == target_class_idx]
    
    if not matching_trials:
        print(f"Error: No trials found for '{args.movement}' in the test dataset.")
        exit(1)
        
    selected_trial_idx = np.random.choice(matching_trials)
    print(f"Selected Trial Index {selected_trial_idx} out of {len(matching_trials)} available '{args.movement}' trials.")

    # 5. Extract chronological sequence of windows for the selected trial
    start_window = selected_trial_idx * test_dataset.num_windows_per_trial
    end_window = start_window + test_dataset.num_windows_per_trial
    
    inputs_list = []
    true_reg_list = []
    
    for i in range(start_window, end_window):
        x, _, y_reg = test_dataset[i]
        inputs_list.append(x)
        true_reg_list.append(y_reg)
        
    # Stack into a batch of shape (num_windows, ...)
    X_batch = torch.stack(inputs_list).to(device)
    Y_true = torch.stack(true_reg_list).numpy()
    
    # 6. Load Model and Weights
    num_classes = len(active_class_map)
    num_channels = len(args.electrodes)
    model = get_model(args.model_type, num_channels, args.feature_ext, num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)
        
    # 7. Run Inference
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # Run the entire sequence of windows through the model
            out_cls, out_reg = model(X_batch, apply_gating=True)
            Y_pred = out_reg.cpu().numpy()

    # 8. Plot the 7-DOF Continuous Trajectories
    # The V-shape targets start generating outputs at the end of the first sliding window (e.g., 0.2s)
    time_axis = np.linspace(0.2, 4.0, len(Y_true))
    
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    dof_names = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6", "Gripper"]
    
    for i in range(7):
        axes[i].plot(time_axis, Y_true[:, i], label='True Physical Requirement', color='blue', linewidth=2.5)
        axes[i].plot(time_axis, Y_pred[:, i], label='Model Prediction', color='red', linestyle='--', linewidth=2.5)
        axes[i].set_ylabel("Scaled Ang")
        axes[i].set_title(f"{dof_names[i]}")
        axes[i].grid(True, alpha=0.4)
        
        # Lock the Y-axis so visually comparing plots makes sense (-1.0 to 1.0 logic)
        axes[i].set_ylim(-1.2, 1.2)
        
        if i == 0:
            axes[i].legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))

    axes[-1].set_xlabel("Time (Seconds)")
    fig.suptitle(f"Chronological Robot Trajectory Simulation\nMovement: {args.movement} | Model: {args.model_type.upper()}", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the output image safely
    safe_name = args.movement.replace(' ', '_').replace(':', '')
    plot_filename = f"simulation_{safe_name}_{args.model_type}.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"\n>>> Simulation complete! Plot saved as '{plot_filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulates a single 4-second physical trial through the EMG-to-Kinematics pipeline.")
    
    # Core Arguments
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained .pth model weights")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root dataset folder")
    parser.add_argument("--movement", type=str, required=True, help="Exact string of the target movement (e.g., 'Wrist-twisting: Supination')")
    
    # Pipeline Arguments (Must match how the model was trained!)
    parser.add_argument("--model_type", type=str, choices=['adaptive', 'multiscale_cnn', 'resnet_spectrogram', 'tcn_lstm'], default='adaptive')
    parser.add_argument("--electrodes", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--scaling", type=str, choices=['none', 'minmax', 'zscore', 'constant'], default='constant')
    parser.add_argument("--feature_ext", type=str, choices=['none', 'pca', 'td'], default='none')
    parser.add_argument("--exclude_rest", action="store_true", help="Must be flagged if the model was trained with 11 classes.")
    parser.add_argument("--no_filter", action="store_true", help="Disable the high-pass butterworth filter")
    parser.add_argument("--highpass_cutoff", type=float, default=20.0, help="Cutoff frequency for the filter")
    
    args = parser.parse_args()
    simulate_trial(args)