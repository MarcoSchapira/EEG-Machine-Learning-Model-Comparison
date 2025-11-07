# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from typing import Union, Tuple

# ------------------------------------------------------------
# Helper: Load labels (1D) and trial_data (3D) from a MATLAB .mat file
# Expects keys: 'labels' -> (n_trials, 1) or (n_trials,)
#               'trial_data' -> (n_trials, n_channels, n_samples)
# Returns:
#   labels_1d: np.ndarray shape (n_trials,)
#   trial_data_3d: np.ndarray shape (n_trials, n_channels, n_samples)
# ------------------------------------------------------------
def load_labels_and_trial_data(
    mat_path: Union[str, Path],
    labels_key: str = "labels",
    trials_key: str = "trial_data"
) -> Tuple[np.ndarray, np.ndarray]:
    mat_path = Path(mat_path)
    md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if labels_key not in md or trials_key not in md:
        raise KeyError(
            "Expected keys '{}' and '{}' not both found in {}".format(
                labels_key, trials_key, mat_path.name
            )
        )

    labels_raw = np.array(md[labels_key])
    trial_data = np.array(md[trials_key], dtype=float)

    # Squeeze labels to 1D (e.g., 300x1 -> 300,)
    labels_1d = labels_raw.squeeze()

    # Ensure integer dtype for labels
    if np.issubdtype(labels_1d.dtype, np.floating):
        labels_1d = labels_1d.astype(int)
    elif labels_1d.dtype.kind in {"S", "U"}:
        labels_1d = labels_1d.astype(int)

    # Basic shape checks
    if trial_data.ndim != 3:
        raise ValueError("'{}' must be 3D (n_trials, n_channels, n_samples); got shape {}".format(
            trials_key, trial_data.shape
        ))
    if labels_1d.ndim != 1 or labels_1d.shape[0] != trial_data.shape[0]:
        raise ValueError("labels length must equal number of trials in trial_data")

    return labels_1d, trial_data


# ------------------------------------------------------------
# Core: Filter trials by trigger and extract a single node/channel
# Inputs:
#   labels: (n_trials,) integer trigger codes (e.g., 8, 11, 21, 61)
#   trial_data: (n_trials, n_channels, n_samples)
#   trigger_code: int (e.g., 8, 11, 21, 61)
#   node_number: 1-based index of the channel inside trial_data (1..n_channels)
# Outputs:
#   node_epochs: (n_selected_trials, n_samples) float array
#   fig: Matplotlib Figure with 3D surface plot (time × trial × value)
# Notes:
#   - node_number is 1-based (like MATLAB). Internally converted to 0-based.
# ------------------------------------------------------------
def extract_node_epochs_and_plot(labels: np.ndarray,
                                 trial_data: np.ndarray,
                                 trigger_code: int,
                                 node_number: int):
    # ----- Validate shapes and indices
    n_trials, n_channels, n_samples = trial_data.shape
    if not (1 <= node_number <= n_channels):
        raise ValueError(f"node_number must be in [1, {n_channels}] (1-based). Got {node_number}.")
    node_idx = node_number - 1  # convert to 0-based

    # ----- Select trials that match the trigger_code
    mask = (labels == trigger_code)
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        raise ValueError(f"No trials found for trigger code {trigger_code}.")

    # ----- Gather 1000-point (or n_samples) epochs for the requested node
    # Shape: (n_selected_trials, n_samples)
    node_epochs = trial_data[idxs, node_idx, :]

    # ------------------------------------------------------------
    # Plot: 3D surface (X=time, Y=trial_index_in_selection, Z=amplitude)
    # ------------------------------------------------------------
    # Build X (time axis) and Y (trial axis) grids
    t = np.arange(n_samples)  # 0..999 (if 1000 samples)
    ytr = np.arange(node_epochs.shape[0])  # 0..(n_selected_trials-1)
    X, Y = np.meshgrid(t, ytr, indexing="xy")

    # Create the figure and 3D axes
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Z is the node epochs (trial × time)
    Z = node_epochs

    # Surface plot with hot-to-cold colormap based on amplitude (Z values)
    # 'coolwarm' goes from blue (cold/low) to red (hot/high)
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)

    # Add colorbar to show the amplitude scale
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Amplitude (µV or raw units)')

    # Labels and title
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Selected trial index")
    ax.set_zlabel("Amplitude (µV or raw units)")
    ax.set_title(f"Node {node_number} | Trigger {trigger_code} | {node_epochs.shape[0]} trials")

    # A little viewing angle adjustment for clarity
    ax.view_init(elev=25, azim=-135)

    plt.tight_layout()

    return node_epochs, fig


# ------------------------------------------------------------
# Example usage (uncomment and edit the path to run locally):
# ------------------------------------------------------------
if __name__ == "__main__":
    path = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_Compact/EEG_session1_sub1_multigrasp_realMove_compact.mat"
    labels, trial_data = load_labels_and_trial_data(path)
    
    # Extract node 5 for trigger 21 (e.g., "Ball" in Hand-Grasp task or "Backward" in Arm-Reach)
    node_epochs, fig = extract_node_epochs_and_plot(labels, trial_data, trigger_code=61, node_number=15)
    print("Extracted array shape:", node_epochs.shape)  # -> (n_selected, 1000)
    plt.show()
