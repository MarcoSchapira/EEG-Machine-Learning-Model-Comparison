"""
Data Loading Module for CSP-LDA Pipeline
=========================================
Loads EEG data from .mat files, aggregates by paradigm, and returns (X, y) tensors.
"""

import sys
from pathlib import Path
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Add parent directory to path to import load_trials_by_action
sys.path.insert(0, str(Path(__file__).parent.parent / "Load_Data"))


# ============================================================================
# Paradigm Trigger Code Mappings
# ============================================================================

# Trigger codes for each paradigm
TRIGGER_TO_CLASS = {
    'reach': {
        11: 'Forward',
        21: 'Backward',
        31: 'Left',
        41: 'Right',
        51: 'Up',
        61: 'Down',
        8: 'Rest'
    },
    'grasp': {
        11: 'Cup',      # Cylindrical
        21: 'Ball',     # Spherical
        61: 'Card',     # Lateral/Lumbrical
        8: 'Rest'
    },
    'twist': {
        91: 'Pronation',
        101: 'Supination',
        8: 'Rest'
    }
}

# Paradigm to file name pattern mapping
PARADIGM_TO_PATTERN = {
    'reach': 'reaching',
    'grasp': 'multigrasp',
    'twist': 'twist'
}


# ============================================================================
# Core Loading Function
# ============================================================================

def get_X_y(
    paradigm: str,
    include_rest: bool = False,
    data_dir: Optional[Union[str, Path]] = None,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load and aggregate EEG data for a given paradigm.
    
    Parameters
    ----------
    paradigm : str
        Paradigm to load: 'reach', 'grasp', or 'twist'
    include_rest : bool, default=False
        If True, include Rest (trigger 8) as a class. If False, filter out Rest trials.
    data_dir : str or Path, optional
        Directory containing .mat files. If None, uses default relative to project root.
    random_seed : int, default=42
        Random seed for reproducibility (used if shuffling is needed)
    
    Returns
    -------
    X : np.ndarray
        Shape (n_trials, n_channels=60, n_times)
        EEG trial data. Only includes 60 EEG channels (excludes EOG/EMG if present).
    y : np.ndarray
        Shape (n_trials,)
        Integer class IDs starting at 0.
    meta : dict
        Metadata dictionary containing:
        - 'trigger_to_class': mapping from trigger codes to class names
        - 'class_to_id': mapping from class names to class IDs
        - 'id_to_class': mapping from class IDs to class names
        - 'file_list': list of files loaded
        - 'file_metadata': list of dicts with subject/session info per file
        - 'class_counts': dict of class name -> count
        - 'n_channels': number of channels (60)
        - 'n_times': number of time samples
        - 'n_trials': total number of trials
        - 'paradigm': paradigm name
    """
    if paradigm not in TRIGGER_TO_CLASS:
        raise ValueError(f"Unknown paradigm: {paradigm}. Must be one of {list(TRIGGER_TO_CLASS.keys())}")
    
    # Set up data directory
    if data_dir is None:
        # Default: relative to project root (5 levels up from this file)
        # This file: git_repo/CSP_LDA/src/data/loading.py
        # Go up 5 levels: capstone/
        data_dir = Path(__file__).parent.parent.parent.parent.parent / "local_data" / "EEG_Compact"
    else:
        data_dir = Path(data_dir)
        # Resolve relative paths
        if not data_dir.is_absolute():
            data_dir = data_dir.resolve()
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please check that the data directory exists, or set 'data_dir' in the config file."
        )
    
    # Find all files matching the paradigm
    file_pattern = PARADIGM_TO_PATTERN[paradigm]
    mat_files = sorted(data_dir.glob(f"EEG_session*_sub*_{file_pattern}_realMove_compact.mat"))
    
    if len(mat_files) == 0:
        raise FileNotFoundError(
            f"No files found for paradigm '{paradigm}' (pattern: {file_pattern}) in {data_dir}"
        )
    
    print(f"Found {len(mat_files)} files for paradigm '{paradigm}'")
    
    # Get trigger-to-class mapping for this paradigm
    trigger_map = TRIGGER_TO_CLASS[paradigm].copy()
    
    # Filter out Rest if not included
    if not include_rest and 8 in trigger_map:
        trigger_map.pop(8)
    
    # Build class ID mapping (excluding control triggers 13, 14)
    valid_triggers = sorted([t for t in trigger_map.keys() if t not in [13, 14]])
    class_names = [trigger_map[t] for t in valid_triggers]
    
    # Create mappings
    trigger_to_class_id = {t: idx for idx, t in enumerate(valid_triggers)}
    class_to_id = {trigger_map[t]: idx for idx, t in enumerate(valid_triggers)}
    id_to_class = {idx: trigger_map[t] for idx, t in enumerate(valid_triggers)}
    
    # Aggregate trials into X and y
    # Load files individually to track metadata per trial
    X_list = []
    y_list = []
    file_metadata_list = []  # Track which file each trial came from
    file_metadata_summary = []  # Summary of all files
    
    # Import load_trials_by_action for individual file loading
    from load_for_training import load_trials_by_action
    
    for mat_file in mat_files:
        # Parse file name to extract metadata
        match = re.match(r'EEG_session(\d+)_sub(\d+)_(\w+)_realMove_compact\.mat', mat_file.name)
        if match:
            session = int(match.group(1))
            subject = int(match.group(2))
            movement_type = match.group(3)
        else:
            session = None
            subject = None
            movement_type = None
        
        file_info = {
            'file_name': mat_file.name,
            'file_path': str(mat_file),
            'session': session,
            'subject': subject,
            'movement_type': movement_type
        }
        file_metadata_summary.append(file_info)
        
        # Load trials from this file
        file_trials_by_action = load_trials_by_action(
            mat_path=str(mat_file),
            labels_key="labels",
            trials_key="trial_data"
        )
        
        # Count trials per class in this file
        file_trial_counts = {}
        
        # Aggregate trials from this file
        for trigger, trials in file_trials_by_action.items():
            if trigger not in valid_triggers:
                continue  # Skip control triggers
            
            class_id = trigger_to_class_id[trigger]
            class_name = id_to_class[class_id]
            
            if class_name not in file_trial_counts:
                file_trial_counts[class_name] = 0
            
            for trial in trials:
                # Ensure trial is (n_channels, n_times)
                if trial.ndim != 2:
                    raise ValueError(f"Expected 2D trial array, got shape {trial.shape}")
                
                # Verify we have 60 channels (or adjust if needed)
                # Files should contain 60 EEG channels
                n_channels, n_times = trial.shape
                
                # Add to lists
                X_list.append(trial)
                y_list.append(class_id)
                file_metadata_list.append(file_info.copy())  # Track which file this trial came from
                file_trial_counts[class_name] += 1
        
        # Add trial counts to file info
        file_info['trial_counts'] = file_trial_counts
        file_info['total_trials'] = sum(file_trial_counts.values())
    
    if len(X_list) == 0:
        raise ValueError(f"No valid trials found for paradigm '{paradigm}'")
    
    # Stack into arrays
    X = np.stack(X_list, axis=0)  # (n_trials, n_channels, n_times)
    y = np.array(y_list, dtype=np.int32)  # (n_trials,)
    
    # Verify shape
    n_trials, n_channels, n_times = X.shape
    assert n_channels == 60, f"Expected 60 EEG channels, got {n_channels}"
    assert len(y) == n_trials, "X and y must have same number of trials"
    
    # Count classes
    class_counts = {}
    for class_id, class_name in id_to_class.items():
        class_counts[class_name] = int(np.sum(y == class_id))
    
    # Build metadata
    meta = {
        'trigger_to_class': {t: trigger_map[t] for t in valid_triggers},
        'class_to_id': class_to_id,
        'id_to_class': id_to_class,
        'file_list': [str(f) for f in mat_files],
        'file_metadata': file_metadata_summary,  # Summary of all files with trial counts
        'trial_file_mapping': file_metadata_list,  # Per-trial file metadata (same length as y)
        'class_counts': class_counts,
        'n_channels': n_channels,
        'n_times': n_times,
        'n_trials': n_trials,
        'paradigm': paradigm,
        'include_rest': include_rest
    }
    
    print(f"\nLoaded {n_trials} trials from {len(mat_files)} files")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {class_counts}")
    
    return X, y, meta


def load_from_config(config_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load data using settings from a YAML configuration file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file
    
    Returns
    -------
    X, y, meta : tuple
        Same as get_X_y()
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required to load from config files. "
            "Install it with: pip install pyyaml"
        )
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    paradigm = config.get('paradigm', 'reach')
    include_rest = config.get('include_rest', False)
    random_seed = config.get('random_seed', 42)
    data_dir = config.get('data_dir', None)
    
    # Resolve relative data_dir path if provided
    if data_dir is not None and not Path(data_dir).is_absolute():
        # Try multiple resolution strategies
        # 1. Relative to config file location
        config_resolved = (config_path.parent / data_dir).resolve()
        if config_resolved.exists():
            data_dir = config_resolved
        else:
            # 2. Relative to git_repo directory (config_path.parent.parent)
            repo_resolved = (config_path.parent.parent / data_dir).resolve()
            if repo_resolved.exists():
                data_dir = repo_resolved
            else:
                # 3. Try as-is (might be relative to current working directory)
                data_dir = Path(data_dir).resolve()
    
    return get_X_y(
        paradigm=paradigm,
        include_rest=include_rest,
        data_dir=data_dir,
        random_seed=random_seed
    )


def save_manifest(meta: Dict, output_path: Union[str, Path]):
    """
    Save metadata manifest to JSON file.
    
    Parameters
    ----------
    meta : dict
        Metadata dictionary from get_X_y()
    output_path : str or Path
        Path to save JSON manifest
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    meta_serializable = {}
    for key, value in meta.items():
        if isinstance(value, np.integer):
            meta_serializable[key] = int(value)
        elif isinstance(value, np.floating):
            meta_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            meta_serializable[key] = value.tolist()
        elif isinstance(value, dict):
            # Recursively convert dict values
            meta_serializable[key] = {
                k: int(v) if isinstance(v, np.integer) else v
                for k, v in value.items()
            }
        else:
            meta_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(meta_serializable, f, indent=2)
    
    print(f"Saved manifest to {output_path}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Load from config
    config_path = Path(__file__).parent.parent.parent / "configs" / "csp_lda.yaml"
    
    if config_path.exists():
        print("=" * 80)
        print("Example 1: Loading from config file")
        print("=" * 80)
        X, y, meta = load_from_config(config_path)
        
        print(f"\nX shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Unique classes: {np.unique(y)}")
        print(f"Class names: {[meta['id_to_class'][cid] for cid in np.unique(y)]}")
        
        # Save manifest
        manifest_path = Path(__file__).parent.parent.parent / "data_manifests" / f"{meta['paradigm']}_manifest.json"
        save_manifest(meta, manifest_path)
    
    # Example 2: Load directly
    print("\n" + "=" * 80)
    print("Example 2: Loading directly with parameters")
    print("=" * 80)
    
    X, y, meta = get_X_y(
        paradigm='reach',
        include_rest=False,
        random_seed=42
    )
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution:")
    for class_name, count in meta['class_counts'].items():
        print(f"  {class_name}: {count}")

