# Pipeline Summary and Model Input Format

## How `run_all_steps.py` Works

### Overview
This script runs the complete EEG data pipeline from raw .mat files to train-ready data, showing detailed information at each step.

### Pipeline Flow

```
Raw .mat Files → Load → Preprocess → Split → Train-Ready Data
```

### Step-by-Step Process

#### Step 1: Load Data (`get_X_y()`)
- **Input**: Paradigm name ('reach', 'grasp', or 'twist')
- **Process**:
  - Finds all matching .mat files in `local_data/EEG_Compact/`
  - Loads trials from each file
  - Maps trigger codes to class IDs (0, 1, 2, ...)
  - Aggregates all trials into unified arrays
- **Output**:
  - `X`: Raw EEG data, shape `(n_trials, 60, n_times)`
  - `y`: Class labels, shape `(n_trials,)`
  - `meta`: Metadata dictionary with class mappings, file info, etc.

#### Step 2: Preprocess (`preprocess_epochs()`)
- **Input**: Raw `X` from Step 1
- **Process** (applied in order):
  1. **Notch Filter**: Removes 60 Hz line noise
  2. **Band-Pass Filter**: 8-30 Hz, zero-phase Butterworth (4th order)
  3. **CAR (Common Average Reference)**: Re-reference to channel mean
  4. **Baseline Correction**: Subtract pre-cue (-0.5s to 0.0s) mean
  5. **ICA** (optional): Artifact removal (off by default)
- **Output**:
  - `X_processed`: Preprocessed EEG data, shape `(n_trials, 60, n_times)` (same as input)
  - `ica_obj`: ICA object (None if not used)

#### Step 3: Create Splits (`create_splits()`)
- **Input**: Preprocessed `X_processed`, `y`, and `meta`
- **Process**:
  - Creates stratified 5-fold CV splits (primary)
  - Creates Leave-One-File-Out (LOFO) splits (robustness)
  - Validates class coverage and data integrity
  - Saves splits to JSON files in `artifacts/`
- **Output**:
  - `result`: Dictionary containing:
    - `splits['stratified5']`: List of 5 fold dictionaries
    - `splits['lofo']`: List of file-based splits
    - `manifest`: Complete metadata
    - `paths`: Paths to saved JSON files

#### Step 4: Extract Train-Ready Data
- **Input**: Split results from Step 3
- **Process**:
  - Extracts first fold from stratified splits
  - Creates train/validation arrays using indices
- **Output**:
  - `X_train`: Training data, shape `(n_train, 60, n_times)`
  - `y_train`: Training labels, shape `(n_train,)`
  - `X_val`: Validation data, shape `(n_val, 60, n_times)`
  - `y_val`: Validation labels, shape `(n_val,)`

### Helper Function: `print_array_info()`
Prints comprehensive information about numpy arrays:
- Shape and dimensions
- Memory usage
- Statistics (min, max, mean, std)
- Sample values (formatted for 1D/2D/3D arrays)

---

## Model Input Format

### Your CSP-LDA Model Should Expect:

#### Training Data (`X_train`, `y_train`)

```python
X_train: np.ndarray
    Shape: (n_train, 60, n_times)
    - n_train: Number of training trials
    - 60: Number of EEG channels (fixed)
    - n_times: Number of time samples (typically 1000 = 4s at 250 Hz)
    Data type: float64
    Values: Preprocessed EEG signals (filtered, CAR, baseline-corrected)
    Range: Typically between -100 and 100 µV (after preprocessing)

y_train: np.ndarray
    Shape: (n_train,)
    Data type: int32
    Values: Integer class IDs (0, 1, 2, ..., K-1)
    - 0, 1, 2, 3, 4, 5 for 'reach' paradigm (6 classes)
    - 0, 1, 2 for 'grasp' paradigm (3 classes)
    - 0, 1 for 'twist' paradigm (2 classes)
```

#### Validation Data (`X_val`, `y_val`)

```python
X_val: np.ndarray
    Shape: (n_val, 60, n_times)
    - Same structure as X_train
    - n_val: Number of validation trials
    - Typically ~20% of total trials (for 5-fold CV)

y_val: np.ndarray
    Shape: (n_val,)
    - Same structure as y_train
    - Class IDs for validation set
```

### Expected Data Characteristics

1. **Shape Consistency**:
   - All trials have same shape: `(60, n_times)`
   - Channels dimension is always 60 (EEG channels only)
   - Time dimension is consistent across all trials

2. **Preprocessing**:
   - Already filtered (8-30 Hz band-pass)
   - Already CAR-referenced
   - Already baseline-corrected
   - No NaN or Inf values

3. **Class Labels**:
   - Start at 0 (0-indexed)
   - Sequential (0, 1, 2, ..., K-1)
   - Balanced across train/val (stratified split)

4. **Memory Considerations**:
   - Typical size: ~100-1000 trials
   - Each trial: 60 × 1000 = 60,000 floats = ~480 KB
   - Full dataset: ~50-500 MB depending on trial count

### Model Usage Pattern

```python
# Example: How to use the data in your CSP-LDA model

# 1. Get splits (already created by run_all_steps.py)
result = create_splits(X_processed, y, meta, schemes=['stratified5'])

# 2. For each fold
for fold_idx, fold in enumerate(result['splits']['stratified5']):
    # Extract indices
    train_idx = np.array(fold['train_idx'])
    val_idx = np.array(fold['val_idx'])
    
    # Get train/val data
    X_train = X_processed[train_idx]  # Shape: (n_train, 60, n_times)
    X_val = X_processed[val_idx]       # Shape: (n_val, 60, n_times)
    y_train = y[train_idx]             # Shape: (n_train,)
    y_val = y[val_idx]                 # Shape: (n_val,)
    
    # 3. Train your model
    # Your CSP-LDA pipeline should accept:
    #   - X_train: (n_train, 60, n_times)
    #   - y_train: (n_train,)
    
    # 4. Evaluate on validation
    # Your model should output:
    #   - Predictions: (n_val,)
    #   - Probabilities: (n_val, n_classes) [optional]
    
    # 5. Compute metrics
    # balanced_accuracy, macro_f1, cohen_kappa, confusion_matrix
```

### Data Format Summary

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `X_train` | `(n_train, 60, n_times)` | float64 | Preprocessed EEG trials for training |
| `y_train` | `(n_train,)` | int32 | Class labels for training (0 to K-1) |
| `X_val` | `(n_val, 60, n_times)` | float64 | Preprocessed EEG trials for validation |
| `y_val` | `(n_val,)` | int32 | Class labels for validation (0 to K-1) |

**Key Points**:
- All data is **ready to use** - no further preprocessing needed
- Shape is **consistent** - all trials have same dimensions
- Classes are **balanced** - stratified splits ensure equal distribution
- Data is **validated** - no missing values, no data loss

### Example: Accessing Data from Script Output

If you run `run_all_steps.py`, the final variables are:
- `X_train`: Ready for training
- `y_train`: Training labels
- `X_val`: Ready for validation
- `y_val`: Validation labels
- `meta`: Contains `id_to_class` mapping for interpreting predictions

### Next Steps for Model Training

1. **Window Selection** (optional): Extract 0.5-2.5s post-cue window if needed
2. **CSP Feature Extraction**: Apply CSP to X_train
3. **LDA Classification**: Train LDA on CSP features
4. **Evaluate**: Use X_val, y_val for validation

The data format is **ready for direct use** with scikit-learn's CSP and LDA implementations.

