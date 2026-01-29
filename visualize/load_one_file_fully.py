# ------------------------------------------------------------
# Legacy File - Backward Compatibility Wrapper
# This file now imports from the new modular structure
# For new code, use visualize.data_loader and visualize.visualize directly
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from pathlib import Path

# Import from new modular structure
from .data_loader import load_labels_and_trial_data, extract_node_epochs
from .visualize import visualize_single_file_node


# ------------------------------------------------------------
# Backward compatibility wrapper for extract_node_epochs_and_plot
# This function is kept for backward compatibility
# ------------------------------------------------------------
def extract_node_epochs_and_plot(labels: np.ndarray,
                                 trial_data: np.ndarray,
                                 trigger_code: int,
                                 node_number: int):
    """
    Legacy function - kept for backward compatibility.
    For new code, use visualize.visualize_single_file_node() instead.
    """
    # Extract node epochs using backend function
    node_epochs = extract_node_epochs(labels, trial_data, trigger_code, node_number)
    
    # Create plot manually (same as original)
    n_samples = trial_data.shape[2]
    t = np.arange(n_samples)
    ytr = np.arange(node_epochs.shape[0])
    X, Y = np.meshgrid(t, ytr, indexing="xy")
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    Z = node_epochs
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Amplitude (µV or raw units)')
    
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Selected trial index")
    ax.set_zlabel("Amplitude (µV or raw units)")
    ax.set_title(f"Node {node_number} | Trigger {trigger_code} | {node_epochs.shape[0]} trials")
    
    ax.view_init(elev=25, azim=-135)
    plt.tight_layout()
    
    return node_epochs, fig


# ------------------------------------------------------------
# Example usage (uncomment and edit the path to run locally):
# ------------------------------------------------------------
if __name__ == "__main__":
    path = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files/EEG_session1_sub1_multigrasp_realMove_compact.mat"
    labels, trial_data = load_labels_and_trial_data(path)
    
    # Extract node 5 for trigger 21 (e.g., "Ball" in Hand-Grasp task or "Backward" in Arm-Reach)
    node_epochs, fig = extract_node_epochs_and_plot(labels, trial_data, trigger_code=61, node_number=15)
    print("Extracted array shape:", node_epochs.shape)  # -> (n_selected, 1000)
    plt.show()
