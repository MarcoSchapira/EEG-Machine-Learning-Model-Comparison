"""
EEG Data Loading Script
=======================
This script loads EEG data from MATLAB .mat files and organizes it for neural network processing.

Data Structure:
- Files are located in: local_data/EEG_Compact/
- Naming pattern: EEG_session{1-3}_sub{1-25}_{movement_type}_realMove_compact.mat
- Movement types: multigrasp, reaching, twist
- Each file contains trial data with shape (trials, channels, time_samples)
"""

import os
import numpy as np
import scipy.io
from pathlib import Path
from collections import defaultdict
import re


# ============================================================================
# Configuration
# ============================================================================

# Base directory containing the data
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent  # Go up to project root (capstone/)
DATA_DIR = BASE_DIR / "local_data" / "EEG_Compact"

# Channel information (from dataset description)
# EEG electrodes: channels 1-31 and 36-64 (59 channels total)
# EOG electrodes: channels 32-35 (4 channels)
# EMG electrodes: channels 65-71 (7 channels)
# Total: 71 channels, but files contain 60 channels (likely 59 EEG + 1 reference)

# Trigger codes mapping (from dataset description)
TRIGGER_CODES = {
    # Arm reaching
    '11': 'Forward',
    '21': 'Backward',
    '31': 'Left',
    '41': 'Right',
    '51': 'Up',
    '61': 'Down',
    # Hand grasping (multigrasp)
    # '11': 'Cup',  # Same code as Forward
    # '21': 'Ball',  # Same code as Backward
    # '61': 'Card',  # Same code as Down
    # Wrist twisting
    '91': 'Pronation',
    '101': 'Supination',
    # Rest (common to all)
    '8': 'Rest',
    # Experiment control
    '13': 'Start',
    '14': 'End'
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_matlab_file(file_path):
    """
    Load a single MATLAB .mat file and extract relevant data.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the .mat file
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'trial_data': numpy array of shape (trials, channels, time_samples)
        - 'labels': numpy array of shape (trials, 1) with class labels
        - 'downsample_factor': downsampling factor used
        - 'initial_sample_rate': original sampling rate in Hz
        - 'effective_sample_rate': final sampling rate after downsampling
        - 'file_info': dictionary with file metadata (session, subject, movement_type)
    """
    # Load the MATLAB file
    mat_data = scipy.io.loadmat(str(file_path))
    
    # Extract data arrays
    trial_data = mat_data['trial_data']  # Shape: (trials, channels, time_samples)
    labels = mat_data['labels']  # Shape: (trials, 1)
    downsample_factor = int(mat_data['downsample_factor'][0, 0])
    initial_sample_rate = float(mat_data['initial_sample_rate'][0, 0])
    effective_sample_rate = initial_sample_rate / downsample_factor
    
    # Parse file name to extract metadata
    file_name = Path(file_path).stem
    match = re.match(r'EEG_session(\d+)_sub(\d+)_(\w+)_realMove_compact', file_name)
    
    if match:
        session = int(match.group(1))
        subject = int(match.group(2))
        movement_type = match.group(3)
    else:
        session = None
        subject = None
        movement_type = None
    
    return {
        'trial_data': trial_data,
        'labels': labels.flatten(),  # Flatten to 1D array for easier handling
        'downsample_factor': downsample_factor,
        'initial_sample_rate': initial_sample_rate,
        'effective_sample_rate': effective_sample_rate,
        'file_info': {
            'session': session,
            'subject': subject,
            'movement_type': movement_type,
            'file_name': file_name
        }
    }


def load_all_data(data_dir=None, max_files=None):
    """
    Load all MATLAB files from the data directory and organize them.
    
    Parameters:
    -----------
    data_dir : str or Path, optional
        Directory containing .mat files. If None, uses default DATA_DIR.
    max_files : int, optional
        Maximum number of files to load. If None, loads all files.
        Useful for testing or when memory is limited.
        
    Returns:
    --------
    dict : Organized data structure:
        - 'by_file': list of all loaded data dictionaries
        - 'by_session': dict organized by session number
        - 'by_subject': dict organized by subject number
        - 'by_movement': dict organized by movement type
        - 'summary': dictionary with overall statistics
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)
    
    # Verify directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all .mat files
    mat_files = sorted(data_dir.glob("*.mat"))
    
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    
    # Limit number of files if specified
    total_files = len(mat_files)
    if max_files is not None:
        mat_files = mat_files[:max_files]
        print(f"Found {total_files} total .mat files, loading first {max_files}")
    else:
        print(f"Found {total_files} .mat files")
    
    # Initialize data structures
    all_data = []
    by_session = defaultdict(list)
    by_subject = defaultdict(list)
    by_movement = defaultdict(list)
    
    # Load each file
    for file_path in mat_files:
        print(f"Loading: {file_path.name}")
        try:
            data = load_matlab_file(file_path)
            all_data.append(data)
            
            # Organize by different dimensions
            session = data['file_info']['session']
            subject = data['file_info']['subject']
            movement = data['file_info']['movement_type']
            
            if session is not None:
                by_session[session].append(data)
            if subject is not None:
                by_subject[subject].append(data)
            if movement is not None:
                by_movement[movement].append(data)
                
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    # Calculate summary statistics
    total_trials = sum(len(d['labels']) for d in all_data)
    total_samples = sum(d['trial_data'].size for d in all_data)
    
    # Get unique sessions, subjects, and movement types
    sessions = sorted(set(d['file_info']['session'] for d in all_data if d['file_info']['session'] is not None))
    subjects = sorted(set(d['file_info']['subject'] for d in all_data if d['file_info']['subject'] is not None))
    movements = sorted(set(d['file_info']['movement_type'] for d in all_data if d['file_info']['movement_type'] is not None))
    
    summary = {
        'num_files': len(all_data),
        'total_trials': total_trials,
        'total_samples': total_samples,
        'sessions': sessions,
        'subjects': subjects,
        'movements': movements,
        'data_shape_example': all_data[0]['trial_data'].shape if all_data else None,
        'sample_rate': all_data[0]['effective_sample_rate'] if all_data else None
    }
    
    return {
        'by_file': all_data,
        'by_session': dict(by_session),
        'by_subject': dict(by_subject),
        'by_movement': dict(by_movement),
        'summary': summary
    }


# ============================================================================
# Data Display and Analysis Functions
# ============================================================================

def display_data_summary(data_dict):
    """
    Display a comprehensive summary of the loaded data.
    
    Parameters:
    -----------
    data_dict : dict
        Output from load_all_data() function
    """
    summary = data_dict['summary']
    
    print("\n" + "="*80)
    print("EEG DATA SUMMARY")
    print("="*80)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total files loaded: {summary['num_files']}")
    print(f"  Total trials: {summary['total_trials']:,}")
    print(f"  Total data samples: {summary['total_samples']:,}")
    
    if summary['data_shape_example']:
        n_trials, n_channels, n_timepoints = summary['data_shape_example']
        print(f"\n📐 Data Shape (per file):")
        print(f"  Trials per file: ~{n_trials}")
        print(f"  Channels: {n_channels}")
        print(f"  Time points per trial: {n_timepoints}")
        print(f"  Trial duration: ~{n_timepoints / summary['sample_rate']:.2f} seconds" if summary['sample_rate'] else "")
    
    if summary['sample_rate']:
        print(f"\n⏱️  Sampling Information:")
        print(f"  Effective sampling rate: {summary['sample_rate']:.1f} Hz")
    
    print(f"\n📁 Data Organization:")
    print(f"  Sessions: {summary['sessions']}")
    print(f"  Subjects: {len(summary['subjects'])} subjects (IDs: {summary['subjects'][:5]}{'...' if len(summary['subjects']) > 5 else ''})")
    print(f"  Movement types: {summary['movements']}")
    
    # Display breakdown by session
    print(f"\n📂 Breakdown by Session:")
    for session in summary['sessions']:
        session_data = data_dict['by_session'][session]
        num_files = len(session_data)
        num_trials = sum(len(d['labels']) for d in session_data)
        print(f"  Session {session}: {num_files} files, {num_trials:,} trials")
    
    # Display breakdown by movement type
    print(f"\n🏃 Breakdown by Movement Type:")
    for movement in summary['movements']:
        movement_data = data_dict['by_movement'][movement]
        num_files = len(movement_data)
        num_trials = sum(len(d['labels']) for d in movement_data)
        print(f"  {movement}: {num_files} files, {num_trials:,} trials")
    
    # Display label distribution for a sample file
    if data_dict['by_file']:
        print(f"\n🏷️  Label Distribution (sample from first file):")
        sample_data = data_dict['by_file'][0]
        unique_labels, counts = np.unique(sample_data['labels'], return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = TRIGGER_CODES.get(str(label), f'Unknown ({label})')
            print(f"  Label {label} ({label_name}): {count} trials")
    
    print("\n" + "="*80)


def display_file_details(data_dict, session=None, subject=None, movement_type=None):
    """
    Display detailed information about specific files.
    
    Parameters:
    -----------
    data_dict : dict
        Output from load_all_data() function
    session : int, optional
        Filter by session number
    subject : int, optional
        Filter by subject number
    movement_type : str, optional
        Filter by movement type
    """
    files_to_display = data_dict['by_file']
    
    # Apply filters
    if session is not None:
        files_to_display = [d for d in files_to_display if d['file_info']['session'] == session]
    if subject is not None:
        files_to_display = [d for d in files_to_display if d['file_info']['subject'] == subject]
    if movement_type is not None:
        files_to_display = [d for d in files_to_display if d['file_info']['movement_type'] == movement_type]
    
    print(f"\n📋 Detailed File Information ({len(files_to_display)} files):")
    print("-" * 80)
    
    for data in files_to_display[:10]:  # Show first 10 files
        info = data['file_info']
        print(f"\nFile: {info['file_name']}")
        print(f"  Session: {info['session']}, Subject: {info['subject']}, Movement: {info['movement_type']}")
        print(f"  Data shape: {data['trial_data'].shape}")
        print(f"  Number of trials: {len(data['labels'])}")
        print(f"  Unique labels: {np.unique(data['labels'])}")
        print(f"  Sample rate: {data['effective_sample_rate']:.1f} Hz")
    
    if len(files_to_display) > 10:
        print(f"\n... and {len(files_to_display) - 10} more files")


# ============================================================================
# Data Preparation for Neural Networks
# ============================================================================

def prepare_for_neural_network(data_dict, normalize=True):
    """
    Prepare the loaded data in a format suitable for neural network input.
    
    Parameters:
    -----------
    data_dict : dict
        Output from load_all_data() function
    normalize : bool, default=True
        Whether to normalize the data (z-score normalization)
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'X': numpy array of shape (total_trials, channels, time_samples) - features
        - 'y': numpy array of shape (total_trials,) - labels
        - 'metadata': list of dictionaries with metadata for each trial
        - 'label_mapping': dictionary mapping label codes to class names
    """
    all_trials = []
    all_labels = []
    all_metadata = []
    
    # Concatenate all trials from all files
    for data in data_dict['by_file']:
        n_trials = len(data['labels'])
        for i in range(n_trials):
            all_trials.append(data['trial_data'][i])
            all_labels.append(data['labels'][i])
            all_metadata.append({
                'session': data['file_info']['session'],
                'subject': data['file_info']['subject'],
                'movement_type': data['file_info']['movement_type'],
                'file_name': data['file_info']['file_name']
            })
    
    # Convert to numpy arrays
    X = np.array(all_trials)  # Shape: (total_trials, channels, time_samples)
    y = np.array(all_labels)  # Shape: (total_trials,)
    
    # Normalize if requested
    if normalize:
        # Reshape for normalization: (trials, channels, time) -> (trials, channels*time)
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        
        # Z-score normalization
        mean = np.mean(X_flat, axis=1, keepdims=True)
        std = np.std(X_flat, axis=1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        
        X_normalized = (X_flat - mean) / std
        X = X_normalized.reshape(original_shape)
    
    # Create label mapping
    unique_labels = np.unique(y)
    label_mapping = {int(label): TRIGGER_CODES.get(str(int(label)), f'Class_{int(label)}') for label in unique_labels}
    
    print(f"\n✅ Data prepared for neural network:")
    print(f"  Input shape (X): {X.shape}")
    print(f"  Labels shape (y): {y.shape}")
    print(f"  Number of classes: {len(unique_labels)}")
    print(f"  Classes: {label_mapping}")
    if normalize:
        print(f"  Data normalized: Yes (z-score)")
    else:
        print(f"  Data normalized: No")
    
    return {
        'X': X,
        'y': y,
        'metadata': all_metadata,
        'label_mapping': label_mapping
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Starting EEG Data Loading...")
    print(f"Data directory: {DATA_DIR}")
    
    # Load all data
    # Set max_files=None to load all 225 files, or specify a number for testing/memory-limited scenarios
    try:
        # For testing: load only first 10 files
        # For production: use load_all_data() or load_all_data(max_files=None) to load all files
        data = load_all_data(max_files=10)  # Change to max_files=None to load all 225 files
        
        # Display summary
        display_data_summary(data)
        
        # Display detailed information for first session
        print("\n" + "="*80)
        display_file_details(data, session=1)
        
        # Prepare data for neural network
        print("\n" + "="*80)
        print("PREPARING DATA FOR NEURAL NETWORK")
        print("="*80)
        nn_data = prepare_for_neural_network(data, normalize=True)
        
        # Example: Access prepared data
        print(f"\n📦 Prepared Data Available:")
        print(f"  - nn_data['X']: Feature array with shape {nn_data['X'].shape}")
        print(f"  - nn_data['y']: Label array with shape {nn_data['y'].shape}")
        print(f"  - nn_data['metadata']: List of {len(nn_data['metadata'])} metadata dictionaries")
        print(f"  - nn_data['label_mapping']: {nn_data['label_mapping']}")
        
        print("\n✅ Data loading complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

