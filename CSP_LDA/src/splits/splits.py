"""
Train/Validation Splits Module
===============================
Creates reproducible train/validation splits for CSP-LDA pipeline.

Supports:
- Stratified 5-fold CV (primary)
- Leave-One-File-Out (LOFO) for cross-session robustness
"""

import sys
from pathlib import Path
import numpy as np
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import warnings

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    StratifiedKFold = None
    StratifiedGroupKFold = None

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Default split configuration
DEFAULT_N_SPLITS = 5  # For stratified 5-fold CV
MIN_SAMPLES_PER_CLASS = 5  # Minimum samples per class for stratification

# Output directory for split artifacts
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"


# ============================================================================
# Helper Functions
# ============================================================================

def create_file_id(trial_file_info: Dict) -> str:
    """
    Create a unique file identifier from trial file metadata.
    
    Parameters
    ----------
    trial_file_info : dict
        Dictionary with keys: file_name, file_path, session, subject, movement_type
    
    Returns
    -------
    str
        File identifier: "sub{subject}_session{session}_{movement_type}"
        or file_name stem if subject/session not available
    """
    if trial_file_info.get('subject') is not None and trial_file_info.get('session') is not None:
        subject = trial_file_info['subject']
        session = trial_file_info['session']
        movement_type = trial_file_info.get('movement_type', 'unknown')
        return f"sub{subject}_session{session}_{movement_type}"
    else:
        # Fallback to file name stem
        file_name = trial_file_info.get('file_name', 'unknown')
        return Path(file_name).stem


def extract_file_ids(meta: Dict) -> np.ndarray:
    """
    Extract file_id array from metadata.
    
    Parameters
    ----------
    meta : dict
        Metadata dictionary from get_X_y()
    
    Returns
    -------
    np.ndarray
        Array of file_id strings, one per trial
    """
    if 'trial_file_mapping' not in meta:
        raise ValueError("meta must contain 'trial_file_mapping'")
    
    file_ids = [create_file_id(trial_info) for trial_info in meta['trial_file_mapping']]
    return np.array(file_ids)


def compute_y_hash(y: np.ndarray) -> str:
    """
    Compute hash of y array for reproducibility.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels array
    
    Returns
    -------
    str
        SHA256 hash of y array
    """
    y_bytes = y.tobytes()
    return hashlib.sha256(y_bytes).hexdigest()


def validate_inputs(X: np.ndarray, y: np.ndarray, meta: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Validate inputs and extract file_ids.
    
    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_trials, n_channels, n_times)
    y : np.ndarray
        Class labels, shape (n_trials,)
    meta : dict
        Metadata dictionary
    
    Returns
    -------
    file_ids : np.ndarray
        File identifiers, shape (n_trials,)
    validated_meta : dict
        Updated metadata with file_ids
    """
    # Check lengths
    n_trials = len(X)
    if len(y) != n_trials:
        raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
    
    # Extract file_ids
    file_ids = extract_file_ids(meta)
    
    if len(file_ids) != n_trials:
        raise ValueError(
            f"file_ids length ({len(file_ids)}) must match number of trials ({n_trials})"
        )
    
    # Check class coverage
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = min(class_counts)
    
    if min_count < MIN_SAMPLES_PER_CLASS:
        warnings.warn(
            f"Some classes have fewer than {MIN_SAMPLES_PER_CLASS} samples. "
            f"Minimum class count: {min_count}. "
            f"Consider reducing n_splits or handling rare classes."
        )
    
    # Add file_ids to meta
    validated_meta = meta.copy()
    validated_meta['file_ids'] = file_ids.tolist()
    validated_meta['unique_file_ids'] = sorted(np.unique(file_ids).tolist())
    validated_meta['n_files'] = len(validated_meta['unique_file_ids'])
    
    return file_ids, validated_meta


# ============================================================================
# Stratified 5-Fold CV
# ============================================================================

def create_stratified_splits(
    y: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = RANDOM_SEED,
    shuffle: bool = True
) -> List[Dict]:
    """
    Create stratified k-fold splits.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels, shape (n_trials,)
    n_splits : int
        Number of folds (default: 5)
    random_state : int
        Random seed
    shuffle : bool
        Whether to shuffle before splitting
    
    Returns
    -------
    list of dict
        List of dictionaries with 'train_idx' and 'val_idx' keys
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for stratified splits. "
            "Install with: pip install scikit-learn"
        )
    
    # Check class coverage
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = min(class_counts)
    
    if min_count < n_splits:
        raise ValueError(
            f"Not enough samples per class for {n_splits}-fold CV. "
            f"Minimum class count: {min_count}, required: {min(n_splits, min_count)}. "
            f"Consider reducing n_splits or handling rare classes."
        )
    
    # Set random seed
    np.random.seed(random_state)
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    splits = []
    indices = np.arange(len(y))
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, y)):
        # Check class distribution
        train_classes = np.unique(y[train_idx])
        val_classes = np.unique(y[val_idx])
        
        # Verify all classes present in validation (should be with stratification)
        if len(val_classes) < len(unique_classes):
            warnings.warn(
                f"Fold {fold_idx}: Not all classes present in validation set. "
                f"Train: {len(train_classes)}, Val: {len(val_classes)}, Total: {len(unique_classes)}"
            )
        
        # Compute class distributions
        train_class_counts = Counter(y[train_idx])
        val_class_counts = Counter(y[val_idx])
        
        split_dict = {
            'fold': fold_idx,
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
            'train_class_counts': {int(k): int(v) for k, v in train_class_counts.items()},
            'val_class_counts': {int(k): int(v) for k, v in val_class_counts.items()},
            'n_train': len(train_idx),
            'n_val': len(val_idx)
        }
        
        splits.append(split_dict)
    
    return splits


# ============================================================================
# Leave-One-File-Out (LOFO)
# ============================================================================

def create_lofo_splits(
    y: np.ndarray,
    file_ids: np.ndarray,
    meta: Dict
) -> List[Dict]:
    """
    Create Leave-One-File-Out splits.
    
    For each unique file_id, use that file as validation and all others as training.
    If a file's validation set is missing any class, merge with adjacent files.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels, shape (n_trials,)
    file_ids : np.ndarray
        File identifiers, shape (n_trials,)
    meta : dict
        Metadata dictionary (for accessing class names)
    
    Returns
    -------
    list of dict
        List of dictionaries with 'file_id', 'train_idx', 'val_idx', etc.
    """
    unique_file_ids = sorted(np.unique(file_ids).tolist())
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    splits = []
    
    for file_id in unique_file_ids:
        # Initial validation set: all trials from this file
        val_mask = (file_ids == file_id)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        
        # Check class coverage in validation
        val_classes = np.unique(y[val_idx])
        
        # If missing classes, try to merge with next file(s)
        if len(val_classes) < n_classes:
            missing_classes = set(unique_classes) - set(val_classes)
            warnings.warn(
                f"File '{file_id}' missing classes {missing_classes}. "
                f"Attempting to merge with adjacent files..."
            )
            
            # Try to find adjacent file with missing classes
            file_idx = unique_file_ids.index(file_id)
            merged = False
            
            # Try next file
            for next_file_idx in range(file_idx + 1, len(unique_file_ids)):
                next_file_id = unique_file_ids[next_file_idx]
                next_mask = (file_ids == next_file_id)
                next_val_idx = np.where(next_mask)[0]
                combined_classes = np.unique(y[np.concatenate([val_idx, next_val_idx])])
                
                if len(combined_classes) == n_classes:
                    # Merge successful
                    val_idx = np.concatenate([val_idx, next_val_idx])
                    train_mask = ~np.isin(np.arange(len(y)), val_idx)
                    train_idx = np.where(train_mask)[0]
                    merged = True
                    warnings.warn(f"  Merged with '{next_file_id}' to achieve full class coverage")
                    break
            
            if not merged:
                warnings.warn(
                    f"  Could not merge to achieve full class coverage. "
                    f"Validation set will have {len(val_classes)}/{n_classes} classes."
                )
        
        # Compute class distributions
        train_class_counts = Counter(y[train_idx])
        val_class_counts = Counter(y[val_idx])
        
        split_dict = {
            'file_id': file_id,
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
            'train_class_counts': {int(k): int(v) for k, v in train_class_counts.items()},
            'val_class_counts': {int(k): int(v) for k, v in val_class_counts.items()},
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_classes_in_val': len(np.unique(y[val_idx]))
        }
        
        splits.append(split_dict)
    
    return splits


# ============================================================================
# Group-Aware Stratified Splits (Optional)
# ============================================================================

def create_stratified_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = RANDOM_SEED,
    shuffle: bool = True
) -> List[Dict]:
    """
    Create stratified group k-fold splits.
    
    Ensures that groups (e.g., subjects) don't mix across train/val.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels, shape (n_trials,)
    groups : np.ndarray
        Group identifiers (e.g., subject_id), shape (n_trials,)
    n_splits : int
        Number of folds
    random_state : int
        Random seed
    shuffle : bool
        Whether to shuffle before splitting
    
    Returns
    -------
    list of dict
        List of dictionaries with 'train_idx' and 'val_idx' keys
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for stratified group splits. "
            "Install with: pip install scikit-learn"
        )
    
    np.random.seed(random_state)
    
    # Create stratified group k-fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    splits = []
    indices = np.arange(len(y))
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(indices, y, groups)):
        train_class_counts = Counter(y[train_idx])
        val_class_counts = Counter(y[val_idx])
        
        split_dict = {
            'fold': fold_idx,
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
            'train_class_counts': {int(k): int(v) for k, v in train_class_counts.items()},
            'val_class_counts': {int(k): int(v) for k, v in val_class_counts.items()},
            'train_groups': sorted(np.unique(groups[train_idx]).tolist()),
            'val_groups': sorted(np.unique(groups[val_idx]).tolist()),
            'n_train': len(train_idx),
            'n_val': len(val_idx)
        }
        
        splits.append(split_dict)
    
    return splits


# ============================================================================
# Main Function
# ============================================================================

def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    meta: Dict,
    schemes: List[str] = ['stratified5', 'lofo'],
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = RANDOM_SEED,
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Dict:
    """
    Create train/validation splits for CSP-LDA pipeline.
    
    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_trials, n_channels, n_times)
    y : np.ndarray
        Class labels, shape (n_trials,)
    meta : dict
        Metadata dictionary from get_X_y()
    schemes : list of str
        Split schemes to create: 'stratified5', 'lofo', 'stratified_group'
    n_splits : int
        Number of folds for stratified splits
    random_state : int
        Random seed for reproducibility
    output_dir : str or Path, optional
        Directory to save split artifacts (default: artifacts/)
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    dict
        Dictionary with split information and saved paths
    """
    if output_dir is None:
        output_dir = ARTIFACTS_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(random_state)
    
    # Validate inputs
    if verbose:
        print("Validating inputs...")
    file_ids, validated_meta = validate_inputs(X, y, meta)
    
    # Compute y hash for reproducibility
    y_hash = compute_y_hash(y)
    
    # Create splits
    all_splits = {}
    
    if 'stratified5' in schemes:
        if verbose:
            print(f"\nCreating stratified {n_splits}-fold CV splits...")
        splits_stratified = create_stratified_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        all_splits['stratified5'] = splits_stratified
        
        # Save stratified splits
        stratified_path = output_dir / "splits_stratified5.json"
        with open(stratified_path, 'w') as f:
            json.dump(splits_stratified, f, indent=2)
        if verbose:
            print(f"  Saved to {stratified_path}")
    
    if 'lofo' in schemes:
        if verbose:
            print(f"\nCreating Leave-One-File-Out splits...")
        splits_lofo = create_lofo_splits(y, file_ids, validated_meta)
        all_splits['lofo'] = splits_lofo
        
        # Save LOFO splits
        lofo_path = output_dir / "splits_lofo.json"
        with open(lofo_path, 'w') as f:
            json.dump(splits_lofo, f, indent=2)
        if verbose:
            print(f"  Saved to {lofo_path}")
    
    if 'stratified_group' in schemes:
        if verbose:
            print(f"\nCreating stratified group splits...")
        # Extract subject_id from meta
        subject_ids = np.array([
            trial_info.get('subject', i) 
            for i, trial_info in enumerate(meta['trial_file_mapping'])
        ])
        
        splits_group = create_stratified_group_splits(
            y, subject_ids, n_splits=n_splits, random_state=random_state
        )
        all_splits['stratified_group'] = splits_group
        
        # Save group splits
        group_path = output_dir / "splits_stratified_group.json"
        with open(group_path, 'w') as f:
            json.dump(splits_group, f, indent=2)
        if verbose:
            print(f"  Saved to {group_path}")
    
    # Create manifest
    manifest = {
        'random_seed': random_state,
        'n_splits': n_splits,
        'schemes': schemes,
        'y_hash': y_hash,
        'n_trials': len(y),
        'n_classes': len(np.unique(y)),
        'class_counts': validated_meta['class_counts'],
        'file_ids': validated_meta['unique_file_ids'],
        'n_files': validated_meta['n_files'],
        'paradigm': validated_meta.get('paradigm', 'unknown'),
        'split_info': {}
    }
    
    # Add split summaries
    for scheme, splits in all_splits.items():
        manifest['split_info'][scheme] = {
            'n_folds': len(splits),
            'fold_summary': []
        }
        
        for split in splits:
            fold_summary = {
                'n_train': split['n_train'],
                'n_val': split['n_val'],
                'train_classes': len(split['train_class_counts']),
                'val_classes': len(split['val_class_counts'])
            }
            if 'file_id' in split:
                fold_summary['file_id'] = split['file_id']
            manifest['split_info'][scheme]['fold_summary'].append(fold_summary)
    
    # Save manifest
    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"\nSaved manifest to {manifest_path}")
    
    if verbose:
        print(f"\n✓ Created {len(schemes)} split scheme(s)")
        print(f"  Total trials: {len(y)}")
        print(f"  Files: {validated_meta['n_files']}")
        print(f"  Classes: {len(np.unique(y))}")
    
    return {
        'splits': all_splits,
        'manifest': manifest,
        'file_ids': file_ids,
        'validated_meta': validated_meta,
        'paths': {
            'manifest': manifest_path,
            'stratified5': output_dir / "splits_stratified5.json" if 'stratified5' in schemes else None,
            'lofo': output_dir / "splits_lofo.json" if 'lofo' in schemes else None,
            'stratified_group': output_dir / "splits_stratified_group.json" if 'stratified_group' in schemes else None
        }
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Load data and create splits
    print("=" * 80)
    print("EEG Data Splits Module - Example")
    print("=" * 80)
    
    try:
        from src.data.loading import get_X_y
        
        # Load data
        print("\nLoading data...")
        X, y, meta = get_X_y(paradigm='reach', include_rest=False, random_seed=42)
        
        # Create splits
        print("\nCreating splits...")
        result = create_splits(
            X, y, meta,
            schemes=['stratified5', 'lofo'],
            n_splits=5,
            random_state=42,
            verbose=True
        )
        
        print("\n✓ Split creation complete!")
        print(f"  Created {len(result['splits'])} split scheme(s)")
        
    except ImportError as e:
        print(f"⚠ Could not import loading module: {e}")
        print("  This is expected if running standalone")

