# Train/Validation Splits Module

## Overview

This module creates reproducible train/validation splits for the CSP-LDA pipeline, ensuring no data leakage and proper stratification.

## Features

- **Stratified 5-Fold CV**: Primary baseline split scheme
- **Leave-One-File-Out (LOFO)**: Cross-session robustness check
- **Stratified Group K-Fold**: Optional group-aware splits (keeps subjects/sessions separate)
- **Reproducibility**: Fixed random seeds, y-hash for verification
- **Validation**: Comprehensive checks for class coverage and data integrity

## Quick Start

```python
from src.data.loading import get_X_y
from src.data.splits import create_splits

# Load data
X, y, meta = get_X_y(paradigm='reach', include_rest=False)

# Create splits
result = create_splits(
    X, y, meta,
    schemes=['stratified5', 'lofo'],
    n_splits=5,
    random_state=42
)

# Access splits
stratified_splits = result['splits']['stratified5']
lofo_splits = result['splits']['lofo']

# Use splits for training
for fold in stratified_splits:
    train_idx = fold['train_idx']
    val_idx = fold['val_idx']
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # ... train your model
```

## Output Files

All splits are saved to `artifacts/` directory:

- `splits_stratified5.json`: Stratified 5-fold CV splits
- `splits_lofo.json`: Leave-One-File-Out splits
- `splits_stratified_group.json`: Group-aware splits (if created)
- `split_manifest.json`: Complete manifest with metadata

## Split Schemes

### Stratified 5-Fold CV (Primary)

- Uses `sklearn.model_selection.StratifiedKFold`
- Ensures balanced class distribution in each fold
- Shuffled with fixed random seed
- Validates that all classes appear in validation sets

### Leave-One-File-Out (LOFO)

- Uses each unique file as validation set
- All other files used for training
- Automatically merges adjacent files if validation set misses classes
- Ideal for cross-session/subject robustness testing

### Stratified Group K-Fold (Optional)

- Uses `sklearn.model_selection.StratifiedGroupKFold`
- Keeps groups (subjects/sessions) separate across train/val
- Prevents data leakage from same subject/session
- More strict evaluation

## Function Reference

### `create_splits(X, y, meta, schemes=['stratified5', 'lofo'], ...)`

Main function to create all splits.

**Parameters:**
- `X`: Input data, shape `(n_trials, n_channels, n_times)`
- `y`: Class labels, shape `(n_trials,)`
- `meta`: Metadata dictionary from `get_X_y()`
- `schemes`: List of split schemes to create
- `n_splits`: Number of folds for stratified splits (default: 5)
- `random_state`: Random seed (default: 42)
- `output_dir`: Directory to save artifacts (default: `artifacts/`)
- `verbose`: Print progress (default: True)

**Returns:**
- Dictionary with:
  - `splits`: Dictionary of split schemes
  - `manifest`: Complete manifest
  - `file_ids`: File identifiers array
  - `validated_meta`: Updated metadata
  - `paths`: Paths to saved files

### Helper Functions

- `create_file_id(trial_file_info)`: Create file identifier
- `extract_file_ids(meta)`: Extract file IDs from metadata
- `validate_inputs(X, y, meta)`: Validate and prepare inputs
- `create_stratified_splits(y, ...)`: Create stratified splits
- `create_lofo_splits(y, file_ids, meta)`: Create LOFO splits
- `compute_y_hash(y)`: Compute hash for reproducibility

## Validation Checks

The module performs several validation checks:

1. **Length Validation**: Ensures X, y, and file_ids have matching lengths
2. **Class Coverage**: Checks minimum samples per class for stratification
3. **Class Distribution**: Verifies all classes appear in validation sets
4. **File Coverage**: For LOFO, ensures validation sets have all classes (with merging if needed)

## Reproducibility

- Fixed random seed (default: 42)
- Y-hash stored in manifest for verification
- All splits saved as JSON for later loading
- Manifest includes all parameters and metadata

## Dependencies

- **Required**: `numpy`, `scikit-learn`
- Install with: `pip install numpy scikit-learn`

## Testing

Run the test suite:

```bash
cd git_repo
python3 src/data/test_splits.py
```

## Integration with Pipeline

Splits are designed to work with the full pipeline:

```python
# Step 1: Load data
X, y, meta = get_X_y(paradigm='reach', include_rest=False)

# Step 2: Preprocess (optional)
from src.preprocessing.preprocess import preprocess_epochs
X_processed, _ = preprocess_epochs(X, fs=250.0)

# Step 3: Create splits
result = create_splits(X_processed, y, meta, schemes=['stratified5'])

# Step 4: Use splits for training (Step 6 - CSP-LDA)
for fold in result['splits']['stratified5']:
    train_idx = fold['train_idx']
    val_idx = fold['val_idx']
    # ... train CSP-LDA on train_idx, evaluate on val_idx
```

## Notes

- **No Data Leakage**: All splits ensure train/val separation
- **Class Balance**: Stratification maintains class distribution
- **File-based Splitting**: LOFO ensures no file appears in both train and val
- **Flexible**: Can create multiple split schemes simultaneously
- **Reproducible**: Fixed seeds and hash verification ensure reproducibility

