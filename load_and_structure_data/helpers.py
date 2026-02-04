import numpy as np
from typing import Optional, List

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


def rename_labels(
    trial_data: np.ndarray,
    labels: np.ndarray,
    CurrentAction: str,
    AllActions: List[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rename labels to count up from 0 and delete rest trials.
    """

    hand_grasping_triggers = [
        11,  # Cup        -> index 6
        21,  # Ball       -> index 7
        61,  # Card       -> index 8
        8,   # Rest (DELETE)
    ]
    
    arm_reaching_triggers = [
        11,  # Forward    -> index 0
        21,  # Backward   -> index 1
        31,  # Left       -> index 2
        41,  # Right      -> index 3
        51,  # Up         -> index 4
        61,  # Down       -> index 5
        8,   # Rest (DELETE)
    ]

    wrist_twisting_triggers = [
        91,   # Pronation   -> index 9
        101,  # Supination  -> index 10
        8,    # Rest (DELETE)
    ]

    if CurrentAction == "reaching":
        start_index = 0
        raw_labels = arm_reaching_triggers
    elif CurrentAction == "multigrasp":
        start_index = 6
        raw_labels = hand_grasping_triggers
    elif CurrentAction == "twist":
        start_index = 9
        raw_labels = wrist_twisting_triggers
    else:
        raise ValueError(f"Invalid action: {CurrentAction}")

    new_labels = []
    new_trial_data = []

    for raw_label in raw_labels: #! Iterate over the raw labels (like 61, 91, 11, 21, 31, 41, 51, 61)
        
        for i, current_label in enumerate(labels):
            if current_label == 8:
                continue

            if current_label == raw_label:
                new_labels.append(start_index)
                new_trial_data.append(trial_data[i,:,:])
        start_index += 1

    return  new_trial_data, new_labels
