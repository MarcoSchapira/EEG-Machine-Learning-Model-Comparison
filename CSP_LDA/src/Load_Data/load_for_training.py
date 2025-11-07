"""
Data Loading for AI Model Training
===================================
This module provides functions to load EEG data organized by action labels,
making it easy to feed into machine learning models during training.

Each trial is a 2D array (channels × time_samples) containing all nodes and timestamps.
Trials are grouped by action label in a dictionary for easy access.
"""

from pathlib import Path
import numpy as np
from scipy.io import loadmat
from typing import Union, Dict, List, Optional
from collections import defaultdict

# Import the existing loader function
from load_one_file_fully import load_labels_and_trial_data


def load_trials_by_action(
    mat_path: Union[str, Path],
    labels_key: str = "labels",
    trials_key: str = "trial_data"
) -> Dict[int, List[np.ndarray]]:
    """
    Load all trials from a MATLAB file and organize them by action label.
    
    Each trial is extracted as a 2D array (n_channels, n_samples) containing
    all nodes and all timestamps. Trials are grouped by their action/trigger code.
    
    Parameters
    ----------
    mat_path : str or Path
        Path to the .mat file to load
    labels_key : str, default="labels"
        Key name for labels in the .mat file
    trials_key : str, default="trial_data"
        Key name for trial data in the .mat file
    
    Returns
    -------
    dict
        Dictionary where:
        - Keys are action/trigger codes (int, e.g., 8, 11, 21, 61)
        - Values are lists of 2D numpy arrays, one per trial
        - Each 2D array has shape (n_channels, n_samples)
        
    Example
    -------
    >>> data_by_action = load_trials_by_action("path/to/file.mat")
    >>> # Access all "Rest" trials (trigger code 8)
    >>> rest_trials = data_by_action[8]  # List of 2D arrays
    >>> # Access first Rest trial: shape (n_channels, n_samples)
    >>> first_rest_trial = rest_trials[0]
    >>> # Access all "Forward" trials (trigger code 11)
    >>> forward_trials = data_by_action[11]
    """
    # Load labels and trial data using existing function
    labels, trial_data = load_labels_and_trial_data(mat_path, labels_key, trials_key)
    
    n_trials, n_channels, n_samples = trial_data.shape
    
    # Organize trials by action label
    trials_by_action = defaultdict(list)
    
    for trial_idx in range(n_trials):
        action_label = int(labels[trial_idx])
        # Extract single trial: shape (n_channels, n_samples)
        single_trial = trial_data[trial_idx, :, :]  # 2D array
        trials_by_action[action_label].append(single_trial)
    
    # Convert defaultdict to regular dict for cleaner output
    return dict(trials_by_action)


def load_trials_by_action_batch(
    mat_paths: List[Union[str, Path]],
    labels_key: str = "labels",
    trials_key: str = "trial_data"
) -> Dict[int, List[np.ndarray]]:
    """
    Load multiple MATLAB files and combine trials by action label.
    
    This is useful when you want to combine data from multiple files
    (e.g., multiple sessions or subjects) into a single training dataset.
    
    Parameters
    ----------
    mat_paths : list of str or Path
        List of paths to .mat files to load
    labels_key : str, default="labels"
        Key name for labels in the .mat files
    trials_key : str, default="trial_data"
        Key name for trial data in the .mat files
    
    Returns
    -------
    dict
        Dictionary where:
        - Keys are action/trigger codes (int)
        - Values are lists of 2D numpy arrays, one per trial across all files
        - Each 2D array has shape (60 channels, 1000 timesamples)
    """
    all_trials_by_action = defaultdict(list)
    
    for mat_path in mat_paths:
        mat_path = Path(mat_path)
        if not mat_path.exists():
            print(f"Warning: File not found, skipping: {mat_path}")
            continue
        
        print(f"Loading: {mat_path.name}")
        file_trials = load_trials_by_action(mat_path, labels_key, trials_key)
        
        # Merge trials from this file into the combined dictionary
        for action_label, trials in file_trials.items():
            all_trials_by_action[action_label].extend(trials)
    
    return dict(all_trials_by_action)


def get_action_statistics(trials_by_action: Dict[int, List[np.ndarray]]) -> Dict[int, Dict]:
    """
    Get statistics about trials organized by action.
    
    Parameters
    ----------
    trials_by_action : dict
        Dictionary from load_trials_by_action() or load_trials_by_action_batch()
    
    Returns
    -------
    dict
        Dictionary mapping action codes to statistics:
        - 'count': number of trials
        - 'shape': shape of each trial (n_channels, n_samples)
    """
    stats = {}
    for action_label, trials in trials_by_action.items():
        if len(trials) > 0:
            stats[action_label] = {
                'count': len(trials),
                'shape': trials[0].shape,  # All trials should have same shape
                'n_channels': trials[0].shape[0],
                'n_samples': trials[0].shape[1]
            }
    return stats


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Load a single file
    print("=" * 80)
    print("Example 1: Loading a single file")
    print("=" * 80)
    
    file_path = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_Compact/EEG_session1_sub1_multigrasp_realMove_compact.mat"
    
    data_by_action = load_trials_by_action(file_path)
    
    print(f"\nLoaded {len(data_by_action)} different action types")
    print(f"Action labels found: {sorted(data_by_action.keys())}")
    
    # Display statistics
    stats = get_action_statistics(data_by_action)
    print("\nStatistics by action:")
    for action, info in sorted(stats.items()):
        print(f"  Action {action}: {info['count']} trials, shape {info['shape']}")
    
    # Example 2: Access specific action trials
    print("\n" + "=" * 80)
    print("Example 2: Accessing specific action trials")
    print("=" * 80)
    
    # Get all "Rest" trials (trigger code 8)
    if 8 in data_by_action:
        rest_trials = data_by_action[8]
        print(f"\nRest trials (action 8): {len(rest_trials)} trials")
        print(f"First Rest trial shape: {rest_trials[0].shape}")
        print(f"  - Channels: {rest_trials[0].shape[0]}")
        print(f"  - Time samples: {rest_trials[0].shape[1]}")
    
    # Get all "Forward" trials (trigger code 11)
    if 11 in data_by_action:
        forward_trials = data_by_action[11]
        print(f"\nForward trials (action 11): {len(forward_trials)} trials")
        print(f"First Forward trial shape: {forward_trials[0].shape}")
    
    # Example 3: How to use in training loop
    print("\n" + "=" * 80)
    print("Example 3: Usage in training loop")
    print("=" * 80)
    print("""
# In your training script, you would use it like this:

from load_for_training import load_trials_by_action

# Load data organized by action
data_by_action = load_trials_by_action("path/to/file.mat")

# Iterate through actions and trials
for action_label, trials in data_by_action.items():
    for trial_idx, trial_data in enumerate(trials):
        # trial_data is a 2D array (n_channels, n_samples)
        # Feed this into your model
        # X = trial_data  # shape: (n_channels, n_samples)
        # y = action_label  # the label for this trial
        
        # Example: Reshape if needed for your model
        # X_reshaped = trial_data.T  # (n_samples, n_channels) if needed
        # or
        # X_reshaped = trial_data[np.newaxis, :, :]  # (1, n_channels, n_samples)
        
        pass  # Your training code here
    """)
