# ------------------------------------------------------------
# Backend Data Loading Utilities
# Functions for loading and reorganizing data for AI model training
# ------------------------------------------------------------
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional
import numpy as np
from scipy.io import loadmat


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
    """
    Load labels and trial data from a MATLAB .mat file.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    
    Returns:
    --------
    labels_1d : np.ndarray
        Shape (n_trials,) - integer labels
    trial_data_3d : np.ndarray
        Shape (n_trials, n_channels, n_samples) - trial data
    """
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
# Load a single file and return labels and trial data
# ------------------------------------------------------------
def filter_nodes(
    trial_data: np.ndarray,
    selected_nodes: Optional[List[int]]
) -> np.ndarray:
    """
    Filter trial data to keep only selected nodes/channels.
    
    Parameters:
    -----------
    trial_data : np.ndarray
        Shape (n_trials, n_channels, n_samples) - trial data
    selected_nodes : list of int, optional
        List of 1-based node indices to keep. If None, returns all nodes.
    
    Returns:
    --------
    filtered_trial_data : np.ndarray
        Shape (n_trials, n_selected_channels, n_samples) - filtered trial data
    """
    if selected_nodes is None:
        return trial_data
    
    # Convert to 0-based indices and validate
    n_trials, n_channels, n_samples = trial_data.shape
    node_indices = [node - 1 for node in selected_nodes]  # Convert to 0-based
    
    # Validate node indices
    for idx in node_indices:
        if not (0 <= idx < n_channels):
            raise ValueError(f"Node index {idx + 1} is out of range [1, {n_channels}]")
    
    # Filter nodes
    filtered_trial_data = trial_data[:, node_indices, :]
    
    return filtered_trial_data


def load_single_file(
    mat_path: Union[str, Path],
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    selected_nodes: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a single .mat file and return data with metadata.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    selected_nodes : list of int, optional
        List of 1-based node indices to keep. If None, keeps all nodes.
    
    Returns:
    --------
    labels : np.ndarray
        Shape (n_trials,) - integer labels
    trial_data : np.ndarray
        Shape (n_trials, n_selected_channels, n_samples) - filtered trial data
    metadata : dict
        Dictionary containing file metadata (n_trials, n_channels, n_samples, file_path, selected_nodes)
    """
    labels, trial_data = load_labels_and_trial_data(mat_path, labels_key, trials_key)
    
    # Filter nodes if specified
    original_n_channels = trial_data.shape[1]
    trial_data = filter_nodes(trial_data, selected_nodes)
    
    metadata = {
        'n_trials': trial_data.shape[0],
        'n_channels': trial_data.shape[1],
        'n_samples': trial_data.shape[2],
        'file_path': str(mat_path),
        'unique_labels': np.unique(labels).tolist(),
        'original_n_channels': original_n_channels,
        'selected_nodes': selected_nodes if selected_nodes is not None else list(range(1, original_n_channels + 1))
    }
    
    return labels, trial_data, metadata


# ------------------------------------------------------------
# Load multiple files and merge all trials together
# ------------------------------------------------------------
def load_multiple_files(
    mat_paths: List[Union[str, Path]],
    labels_key: str = "labels",
    trials_key: str = "trial_data"
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load multiple .mat files and merge all trials together.
    
    Parameters:
    -----------
    mat_paths : list of str or Path
        List of paths to .mat files
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    
    Returns:
    --------
    merged_labels : np.ndarray
        Shape (total_trials,) - concatenated labels from all files
    merged_trial_data : np.ndarray
        Shape (total_trials, n_channels, n_samples) - concatenated trial data
    file_metadata : list of dict
        List of metadata dictionaries for each file
    """
    all_labels = []
    all_trial_data = []
    file_metadata = []
    
    n_channels = None
    n_samples = None
    
    for mat_path in mat_paths:
        labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key)
        
        # Validate consistent channel and sample dimensions
        if n_channels is None:
            n_channels = metadata['n_channels']
            n_samples = metadata['n_samples']
        else:
            if metadata['n_channels'] != n_channels:
                raise ValueError(
                    f"Inconsistent n_channels: file {mat_path} has {metadata['n_channels']}, "
                    f"expected {n_channels}"
                )
            if metadata['n_samples'] != n_samples:
                raise ValueError(
                    f"Inconsistent n_samples: file {mat_path} has {metadata['n_samples']}, "
                    f"expected {n_samples}"
                )
        
        all_labels.append(labels)
        all_trial_data.append(trial_data)
        file_metadata.append(metadata)
    
    # Concatenate all data
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_trial_data = np.concatenate(all_trial_data, axis=0)
    
    return merged_labels, merged_trial_data, file_metadata


# ------------------------------------------------------------
# Extract trials by trigger code
# ------------------------------------------------------------
def extract_trials_by_trigger(
    labels: np.ndarray,
    trial_data: np.ndarray,
    trigger_code: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all trials matching a specific trigger code.
    
    Parameters:
    -----------
    labels : np.ndarray
        Shape (n_trials,) - integer labels/trigger codes
    trial_data : np.ndarray
        Shape (n_trials, n_channels, n_samples) - trial data
    trigger_code : int
        Trigger code to filter by (e.g., 8, 11, 21, 61)
    
    Returns:
    --------
    filtered_labels : np.ndarray
        Shape (n_selected_trials,) - labels matching trigger_code
    filtered_trial_data : np.ndarray
        Shape (n_selected_trials, n_channels, n_samples) - filtered trial data
    """
    mask = (labels == trigger_code)
    idxs = np.nonzero(mask)[0]
    
    if idxs.size == 0:
        raise ValueError(f"No trials found for trigger code {trigger_code}.")
    
    filtered_labels = labels[idxs]
    filtered_trial_data = trial_data[idxs, :, :]
    
    return filtered_labels, filtered_trial_data


# ------------------------------------------------------------
# Extract data for a specific node/channel
# ------------------------------------------------------------
def extract_node_data(
    trial_data: np.ndarray,
    node_number: int
) -> np.ndarray:
    """
    Extract data for a specific node/channel from trial data.
    
    Parameters:
    -----------
    trial_data : np.ndarray
        Shape (n_trials, n_channels, n_samples) - trial data
    node_number : int
        1-based index of the channel (1..n_channels)
    
    Returns:
    --------
    node_data : np.ndarray
        Shape (n_trials, n_samples) - data for the specified node
    """
    n_trials, n_channels, n_samples = trial_data.shape
    if not (1 <= node_number <= n_channels):
        raise ValueError(f"node_number must be in [1, {n_channels}] (1-based). Got {node_number}.")
    
    node_idx = node_number - 1  # convert to 0-based
    node_data = trial_data[:, node_idx, :]
    
    return node_data


# ------------------------------------------------------------
# Extract node epochs for a specific trigger and node
# ------------------------------------------------------------
def extract_node_epochs(
    labels: np.ndarray,
    trial_data: np.ndarray,
    trigger_code: int,
    node_number: int
) -> np.ndarray:
    """
    Extract epochs for a specific node and trigger code.
    
    Parameters:
    -----------
    labels : np.ndarray
        Shape (n_trials,) - integer labels/trigger codes
    trial_data : np.ndarray
        Shape (n_trials, n_channels, n_samples) - trial data
    trigger_code : int
        Trigger code to filter by
    node_number : int
        1-based index of the channel (1..n_channels)
    
    Returns:
    --------
    node_epochs : np.ndarray
        Shape (n_selected_trials, n_samples) - epochs for the specified node and trigger
    """
    filtered_labels, filtered_trial_data = extract_trials_by_trigger(
        labels, trial_data, trigger_code
    )
    node_epochs = extract_node_data(filtered_trial_data, node_number)
    
    return node_epochs


# ------------------------------------------------------------
# Get all unique trigger codes from labels
# ------------------------------------------------------------
def get_unique_triggers(labels: np.ndarray) -> np.ndarray:
    """
    Get all unique trigger codes from labels.
    
    Parameters:
    -----------
    labels : np.ndarray
        Shape (n_trials,) - integer labels/trigger codes
    
    Returns:
    --------
    unique_triggers : np.ndarray
        Sorted array of unique trigger codes
    """
    return np.unique(labels)


# ------------------------------------------------------------
# Get statistics about the data
# ------------------------------------------------------------
def get_data_statistics(
    labels: np.ndarray,
    trial_data: np.ndarray
) -> Dict:
    """
    Get statistics about the loaded data.
    
    Parameters:
    -----------
    labels : np.ndarray
        Shape (n_trials,) - integer labels/trigger codes
    trial_data : np.ndarray
        Shape (n_trials, n_channels, n_samples) - trial data
    
    Returns:
    --------
    stats : dict
        Dictionary containing data statistics
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    stats = {
        'n_trials': trial_data.shape[0],
        'n_channels': trial_data.shape[1],
        'n_samples': trial_data.shape[2],
        'unique_triggers': unique_labels.tolist(),
        'trigger_counts': dict(zip(unique_labels.tolist(), counts.tolist())),
        'data_shape': trial_data.shape,
        'data_dtype': str(trial_data.dtype),
        'data_min': float(np.min(trial_data)),
        'data_max': float(np.max(trial_data)),
        'data_mean': float(np.mean(trial_data)),
        'data_std': float(np.std(trial_data))
    }
    
    return stats
