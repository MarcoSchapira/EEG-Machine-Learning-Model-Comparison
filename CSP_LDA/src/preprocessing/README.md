# EEG Preprocessing Module

## Overview

This module provides preprocessing functions for EEG motor decoding. All preprocessing steps are applied per-epoch in a vectorized manner.

## Features

- **Notch Filter**: Removes 60 Hz line noise (optional)
- **Band-Pass Filter**: 8-30 Hz zero-phase 4th-order Butterworth (target μ/β motor rhythms)
- **Common Average Reference (CAR)**: Re-reference to channel-wise mean
- **Baseline Correction**: Subtracts mean of pre-cue 0.5s period
- **Optional ICA**: Infomax ICA for artifact removal (requires MNE-Python)

## Quick Start

```python
from src.preprocessing.preprocess import preprocess_epochs
import numpy as np

# Load your data (from loading.py)
# X shape: (n_trials, 60, n_times)
X, y, meta = get_X_y(paradigm='reach', include_rest=False)

# Preprocess
X_processed, ica_obj = preprocess_epochs(
    X,
    fs=250.0,
    do_notch=True,
    do_ica=False  # Set to True for ICA (requires MNE)
)
```

## Hyperparameters

All hyperparameters are defined at the top of `preprocess.py` for easy access:

```python
FS = 250.0              # Sampling rate (Hz)
NOTCH_FREQ = 60.0       # Notch frequency (Hz)
APPLY_NOTCH = True      # Whether to apply notch filter

BANDPASS_LOW = 8.0      # Lower cutoff (Hz)
BANDPASS_HIGH = 30.0    # Upper cutoff (Hz)
BANDPASS_ORDER = 4      # Filter order

BASELINE_START = -0.5   # Baseline start (seconds)
BASELINE_END = 0.0      # Baseline end (seconds)

ICA_METHOD = 'infomax'  # ICA method
ICA_N_COMPONENTS = None # Number of components (None = all)
ICA_MAX_ITER = 1000     # Max iterations
```

## Testing

Run the test suite:

```bash
cd git_repo
python3 preprocessing/test_preprocessing.py
```

**Note**: Requires `numpy` and `scipy`. For ICA tests, also requires `mne`.

## Dependencies

- **Required**: `numpy`, `scipy`
- **Optional**: `mne` (for ICA artifact removal)

Install with:
```bash
pip install numpy scipy
pip install mne  # Optional, for ICA
```

## File Structure

```
preprocessing/
├── __init__.py           # Package initialization
├── preprocess.py         # Main preprocessing module
├── test_preprocessing.py  # Comprehensive test suite
└── README.md            # This file
```

## Function Reference

### `preprocess_epochs(X, fs=250, do_notch=True, do_ica=False, ...)`

Main preprocessing function. Applies all preprocessing steps in order:
1. Notch filter (if enabled)
2. Band-pass filter
3. CAR
4. Baseline correction
5. ICA (if enabled)

**Parameters:**
- `X`: Input epochs, shape `(n_trials, n_channels, n_times)`
- `fs`: Sampling frequency (default: 250.0 Hz)
- `do_notch`: Apply notch filter (default: True)
- `do_ica`: Apply ICA (default: False)
- `verbose`: Print progress (default: True)

**Returns:**
- `X_processed`: Preprocessed epochs, same shape as input
- `ica_object`: Fitted ICA object (None if ICA not applied)

### Individual Functions

- `apply_notch_filter(data, fs, notch_freq)`: Apply notch filter
- `apply_bandpass_filter(data, fs, low, high)`: Apply band-pass filter
- `apply_car(data)`: Apply Common Average Reference
- `apply_baseline_correction(data, fs, baseline_start, baseline_end)`: Baseline correction
- `apply_ica(data, ...)`: Apply ICA (requires MNE)

## Pipeline Details

### 1. Notch Filter (60 Hz)
- Uses `scipy.signal.iirnotch`
- Zero-phase filtering (forward-backward)
- Optional: set `APPLY_NOTCH = False` if already applied

### 2. Band-Pass Filter (8-30 Hz)
- 4th-order Butterworth filter
- Zero-phase filtering (filtfilt)
- Targets μ (8-13 Hz) and β (13-30 Hz) motor rhythms

### 3. Common Average Reference (CAR)
- Subtracts mean across channels for each time sample
- Reduces common noise and artifacts

### 4. Baseline Correction
- Subtracts mean of pre-cue period (-0.5s to 0.0s)
- Removes DC offset and slow drifts
- Assumes epochs include -0.5s pre-cue data

### 5. ICA (Optional)
- Infomax ICA via MNE-Python
- Can identify and remove artifact components
- Off by default for speed

## Integration with Loading Module

The preprocessing module is designed to work seamlessly with the data loading module:

```python
from src.data.loading import get_X_y
from src.preprocessing.preprocess import preprocess_epochs

# Load data
X, y, meta = get_X_y(paradigm='reach', include_rest=False)

# Preprocess
X_processed, _ = preprocess_epochs(X, fs=250.0, verbose=True)

# X_processed is ready for CSP-LDA pipeline
```

## Notes

- All filters use zero-phase filtering (filtfilt) to avoid phase distortion
- Processing is vectorized and efficient for large datasets
- ICA is optional and can be slow for large datasets
- Baseline correction assumes epochs start at -0.5s relative to cue

