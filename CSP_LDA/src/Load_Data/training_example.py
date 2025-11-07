"""
Example: How to Use load_for_training.py in Your AI Model Training
==================================================================

This file demonstrates the best practices for loading and using the data
during model training. Copy and adapt these patterns for your training script.
"""

import numpy as np
from pathlib import Path
from load_for_training import load_trials_by_action, load_trials_by_action_batch


# ============================================================================
# Method 1: Load Single File and Iterate Through Trials
# ============================================================================

def train_single_file_example():
    """
    Example: Load one file and process trials one at a time.
    """
    # Load data organized by action
    file_path = "local_data/EEG_Compact/EEG_session1_sub1_multigrasp_realMove_compact.mat"
    data_by_action = load_trials_by_action(file_path)
    
    # Iterate through each action and its trials
    for action_label, trials in data_by_action.items():
        print(f"\nProcessing action {action_label} ({len(trials)} trials)")
        
        for trial_idx, trial_data in enumerate(trials):
            # trial_data is a 2D array: (n_channels, n_samples)
            # e.g., (60, 1000) = 60 channels × 1000 time samples
            
            # Your model expects different shapes depending on architecture:
            
            # Option A: CNN expects (1, n_channels, n_samples) or (n_channels, n_samples, 1)
            X_cnn = trial_data[np.newaxis, :, :]  # Shape: (1, n_channels, n_samples)
            # or
            # X_cnn = trial_data[:, :, np.newaxis]  # Shape: (n_channels, n_samples, 1)
            
            # Option B: RNN/LSTM expects (n_samples, n_channels)
            X_rnn = trial_data.T  # Shape: (n_samples, n_channels)
            
            # Option C: Transformer expects (n_samples, n_channels)
            X_transformer = trial_data.T  # Shape: (n_samples, n_channels)
            
            # The label for this trial
            y = action_label
            
            # Feed to your model
            # prediction = model(X_cnn)  # or X_rnn, X_transformer, etc.
            # loss = criterion(prediction, y)
            # ... training step ...
            
            pass  # Replace with your actual training code


# ============================================================================
# Method 2: Prepare Batches for Training
# ============================================================================

def prepare_batches(data_by_action, batch_size=32, shuffle=True):
    """
    Convert data_by_action dictionary into batches suitable for training.
    
    Parameters
    ----------
    data_by_action : dict
        Dictionary from load_trials_by_action()
    batch_size : int
        Number of trials per batch
    shuffle : bool
        Whether to shuffle the data
    
    Returns
    -------
    batches : list of tuples
        Each tuple is (X_batch, y_batch) where:
        - X_batch: numpy array of shape (batch_size, n_channels, n_samples)
        - y_batch: numpy array of shape (batch_size,) with action labels
    """
    # Collect all trials and labels
    all_trials = []
    all_labels = []
    
    for action_label, trials in data_by_action.items():
        for trial_data in trials:
            all_trials.append(trial_data)
            all_labels.append(action_label)
    
    # Convert to numpy arrays
    all_trials = np.array(all_trials)  # Shape: (n_total_trials, n_channels, n_samples)
    all_labels = np.array(all_labels)  # Shape: (n_total_trials,)
    
    # Shuffle if requested
    if shuffle:
        indices = np.random.permutation(len(all_trials))
        all_trials = all_trials[indices]
        all_labels = all_labels[indices]
    
    # Create batches
    batches = []
    n_trials = len(all_trials)
    
    for i in range(0, n_trials, batch_size):
        end_idx = min(i + batch_size, n_trials)
        X_batch = all_trials[i:end_idx]
        y_batch = all_labels[i:end_idx]
        batches.append((X_batch, y_batch))
    
    return batches


def train_with_batches_example():
    """
    Example: Load data and train using batches.
    """
    # Load data
    file_path = "local_data/EEG_Compact/EEG_session1_sub1_multigrasp_realMove_compact.mat"
    data_by_action = load_trials_by_action(file_path)
    
    # Prepare batches
    batches = prepare_batches(data_by_action, batch_size=32, shuffle=True)
    
    print(f"Created {len(batches)} batches")
    print(f"First batch X shape: {batches[0][0].shape}")  # (batch_size, n_channels, n_samples)
    print(f"First batch y shape: {batches[0][1].shape}")  # (batch_size,)
    
    # Training loop
    for epoch in range(10):  # Example: 10 epochs
        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            # X_batch shape: (batch_size, n_channels, n_samples)
            # y_batch shape: (batch_size,)
            
            # Reshape if needed for your model
            # For CNN: might need (batch_size, 1, n_channels, n_samples)
            # X_batch_cnn = X_batch[:, np.newaxis, :, :]
            
            # For RNN: might need (batch_size, n_samples, n_channels)
            # X_batch_rnn = np.transpose(X_batch, (0, 2, 1))
            
            # Feed to model
            # predictions = model(X_batch)
            # loss = criterion(predictions, y_batch)
            # ... backprop and update ...
            
            pass  # Replace with your actual training code


# ============================================================================
# Method 3: Load Multiple Files (Combined Dataset)
# ============================================================================

def train_multiple_files_example():
    """
    Example: Load multiple files and combine them for training.
    """
    # List of files to load
    data_dir = Path("local_data/EEG_Compact")
    file_paths = list(data_dir.glob("EEG_session1_sub*_multigrasp_realMove_compact.mat"))
    
    # Load all files and combine by action
    data_by_action = load_trials_by_action_batch(file_paths)
    
    print(f"Loaded {len(file_paths)} files")
    print(f"Total action types: {len(data_by_action)}")
    
    # Count total trials per action
    for action_label, trials in sorted(data_by_action.items()):
        print(f"Action {action_label}: {len(trials)} trials")
    
    # Now use prepare_batches or iterate as in previous examples
    batches = prepare_batches(data_by_action, batch_size=32, shuffle=True)
    
    # Training loop...
    pass


# ============================================================================
# Method 4: Train/Val Split by Action
# ============================================================================

def split_train_val(data_by_action, val_ratio=0.2):
    """
    Split data into training and validation sets, maintaining action balance.
    
    Parameters
    ----------
    data_by_action : dict
        Dictionary from load_trials_by_action()
    val_ratio : float
        Ratio of data to use for validation (0.0 to 1.0)
    
    Returns
    -------
    train_data : dict
        Same structure as data_by_action, but with training trials
    val_data : dict
        Same structure as data_by_action, but with validation trials
    """
    train_data = {}
    val_data = {}
    
    for action_label, trials in data_by_action.items():
        n_trials = len(trials)
        n_val = int(n_trials * val_ratio)
        
        # Shuffle trials for this action
        indices = np.random.permutation(n_trials)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_data[action_label] = [trials[i] for i in train_indices]
        val_data[action_label] = [trials[i] for i in val_indices]
    
    return train_data, val_data


def train_with_validation_example():
    """
    Example: Load data, split into train/val, and train with validation.
    """
    # Load data
    file_path = "local_data/EEG_Compact/EEG_session1_sub1_multigrasp_realMove_compact.mat"
    data_by_action = load_trials_by_action(file_path)
    
    # Split into train and validation
    train_data, val_data = split_train_val(data_by_action, val_ratio=0.2)
    
    # Prepare batches
    train_batches = prepare_batches(train_data, batch_size=32, shuffle=True)
    val_batches = prepare_batches(val_data, batch_size=32, shuffle=False)
    
    print(f"Training batches: {len(train_batches)}")
    print(f"Validation batches: {len(val_batches)}")
    
    # Training loop with validation
    for epoch in range(10):
        # Training phase
        for X_batch, y_batch in train_batches:
            # train_step(model, X_batch, y_batch)
            pass
        
        # Validation phase
        val_loss = 0.0
        for X_batch, y_batch in val_batches:
            # val_loss += validate_step(model, X_batch, y_batch)
            pass
        
        # print(f"Epoch {epoch}: val_loss = {val_loss / len(val_batches)}")
        pass


# ============================================================================
# Method 5: PyTorch DataLoader Example
# ============================================================================

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class EEGDataset(Dataset):
        """PyTorch Dataset for EEG trials organized by action."""
        
        def __init__(self, data_by_action):
            """
            Initialize dataset from data_by_action dictionary.
            
            Parameters
            ----------
            data_by_action : dict
                Dictionary from load_trials_by_action()
            """
            self.trials = []
            self.labels = []
            
            for action_label, trials in data_by_action.items():
                for trial_data in trials:
                    self.trials.append(trial_data)
                    self.labels.append(action_label)
        
        def __len__(self):
            return len(self.trials)
        
        def __getitem__(self, idx):
            trial = self.trials[idx]
            label = self.labels[idx]
            
            # Convert to torch tensor
            # Shape: (n_channels, n_samples)
            trial_tensor = torch.FloatTensor(trial)
            
            # Reshape if needed for your model
            # For CNN: add channel dimension -> (1, n_channels, n_samples)
            # trial_tensor = trial_tensor.unsqueeze(0)
            
            # For RNN: transpose -> (n_samples, n_channels)
            # trial_tensor = trial_tensor.T
            
            return trial_tensor, label
    
    def pytorch_training_example():
        """
        Example: Using PyTorch DataLoader with the loaded data.
        """
        # Load data
        file_path = "local_data/EEG_Compact/EEG_session1_sub1_multigrasp_realMove_compact.mat"
        data_by_action = load_trials_by_action(file_path)
        
        # Create dataset and split
        train_data, val_data = split_train_val(data_by_action, val_ratio=0.2)
        
        train_dataset = EEGDataset(train_data)
        val_dataset = EEGDataset(val_data)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Training loop
        # for epoch in range(10):
        #     for X_batch, y_batch in train_loader:
        #         # X_batch shape: (batch_size, n_channels, n_samples)
        #         # y_batch shape: (batch_size,)
        #         # ... your training code ...
        #         pass
        
        print("PyTorch DataLoader created successfully")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

except ImportError:
    print("PyTorch not available, skipping PyTorch example")


# ============================================================================
# Main: Run Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TRAINING EXAMPLES")
    print("=" * 80)
    
    # Uncomment the example you want to run:
    
    # train_single_file_example()
    # train_with_batches_example()
    # train_multiple_files_example()
    # train_with_validation_example()
    
    # if 'pytorch_training_example' in globals():
    #     pytorch_training_example()
    
    print("\nSee function docstrings for detailed usage examples.")
