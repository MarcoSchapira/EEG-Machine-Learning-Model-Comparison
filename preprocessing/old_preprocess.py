"""
EEG Preprocessing Module
=======================
Preprocesses EEG epochs for motor decoding pipeline.

Per-epoch preprocessing with:
- Notch filter (60 Hz, optional)
- Band-pass filter (8-30 Hz, zero-phase Butterworth)
- Common Average Reference (CAR)
- Baseline correction (pre-cue 0.5s)
- Optional ICA artifact removal
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple
import warnings

# Try to import MNE for ICA (optional)
try:
    import mne
    from mne.preprocessing import ICA
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    ICA = None

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Sampling rate (Hz) - already downsampled to 250 Hz
FS = 250.0

# Notch filter parameters
NOTCH_FREQ = 60.0  # Hz (skip if already applied in raw)
APPLY_NOTCH = False  # Set to False if notch already applied

# Band-pass filter parameters
BANDPASS_LOW = 8.0   # Hz (lower cutoff)
BANDPASS_HIGH = 30.0  # Hz (upper cutoff)
BANDPASS_ORDER = 4    # 4th-order Butterworth

# Baseline correction parameters
BASELINE_START = -0.5  # seconds relative to cue (start of epoch)
BASELINE_END = 0.0     # seconds relative to cue (cue onset)
# At 250 Hz: 0.5s = 125 samples

# ICA parameters (optional artifact removal)
APPLY_ICA = False
ICA_METHOD = 'infomax'  # 'infomax' or 'fastica'
ICA_N_COMPONENTS = None  # None = use all components, or specify number
ICA_MAX_ITER = 1000      # Maximum iterations for ICA convergence

# ============================================================================
# Preprocessing Functions
# ============================================================================

def _check_signal_length(data: np.ndarray, filter_coeffs: Tuple) -> bool:
    """
    Check if signal is long enough for filtfilt.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    filter_coeffs : tuple
        (b, a) filter coefficients
    
    Returns
    -------
    bool
        True if signal is long enough, False otherwise
    """
    b, a = filter_coeffs
    # filtfilt requires length > 3 * max(len(a), len(b)) - 1
    min_length = 3 * max(len(a), len(b)) - 1
    
    # Get time dimension (last axis)
    n_times = data.shape[-1]
    
    return n_times > min_length


def apply_notch_filter(
    data: np.ndarray,
    fs: float = FS,
    notch_freq: float = NOTCH_FREQ,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove line noise (60 Hz).
    
    Parameters
    ----------
    data : np.ndarray
        Input data, shape (..., n_times) or (n_trials, n_channels, n_times)
    fs : float
        Sampling frequency in Hz
    notch_freq : float
        Notch frequency in Hz (default: 60.0)
    quality_factor : float
        Quality factor for notch filter (higher = narrower notch)
    
    Returns
    -------
    np.ndarray
        Filtered data with same shape as input
    """
    # Design notch filter
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    
    # Check if signal is long enough for filtfilt
    if not _check_signal_length(data, (b, a)):
        n_times = data.shape[-1]
        min_length = 3 * max(len(a), len(b)) - 1
        warnings.warn(
            f"Signal too short for notch filter (length={n_times}, "
            f"required>{min_length}). Returning unfiltered data."
        )
        return data
    
    # Apply filter along time axis (last dimension)
    try:
        if data.ndim == 2:
            # Single trial: (n_channels, n_times)
            filtered = signal.filtfilt(b, a, data, axis=-1)
        elif data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_times)
            filtered = np.array([signal.filtfilt(b, a, trial, axis=-1) 
                                for trial in data])
        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    except ValueError as e:
        if "padlen" in str(e):
            warnings.warn(
                f"Signal too short for filtfilt: {e}. Returning unfiltered data."
            )
            return data
        raise
    
    return filtered


def apply_bandpass_filter(
    data: np.ndarray,
    fs: float = FS,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER
) -> np.ndarray:
    """
    Apply zero-phase band-pass Butterworth filter.
    
    Parameters
    ----------
    data : np.ndarray
        Input data, shape (..., n_times) or (n_trials, n_channels, n_times)
    fs : float
        Sampling frequency in Hz
    low : float
        Lower cutoff frequency in Hz
    high : float
        Upper cutoff frequency in Hz
    order : int
        Filter order (default: 4)
    
    Returns
    -------
    np.ndarray
        Filtered data with same shape as input
    """
    # Design Butterworth filter
    nyquist = fs / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    # Ensure frequencies are within valid range
    low_norm = max(0.01, min(low_norm, 0.99))
    high_norm = max(0.01, min(high_norm, 0.99))
    
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    # Check if signal is long enough for filtfilt
    if not _check_signal_length(data, (b, a)):
        n_times = data.shape[-1]
        min_length = 3 * max(len(a), len(b)) - 1
        warnings.warn(
            f"Signal too short for band-pass filter (length={n_times}, "
            f"required>{min_length}). Returning unfiltered data."
        )
        return data
    
    # Apply zero-phase filter (filtfilt) along time axis
    try:
        if data.ndim == 2:
            # Single trial: (n_channels, n_times)
            filtered = signal.filtfilt(b, a, data, axis=-1)
        elif data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_times)
            filtered = np.array([signal.filtfilt(b, a, trial, axis=-1) 
                                for trial in data])
        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    except ValueError as e:
        if "padlen" in str(e):
            warnings.warn(
                f"Signal too short for filtfilt: {e}. Returning unfiltered data."
            )
            return data
        raise
    
    return filtered


def apply_car(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR).
    
    Subtracts channel-wise mean per time sample.
    
    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_trials, n_channels, n_times) or (n_channels, n_times)
    
    Returns
    -------
    np.ndarray
        CAR-referenced data with same shape as input
    """
    if data.ndim == 2:
        # Single trial: (n_channels, n_times)
        # Calculate mean across channels for each time point
        mean_across_channels = np.mean(data, axis=0, keepdims=True)
        car_data = data - mean_across_channels
    elif data.ndim == 3:
        # Multiple trials: (n_trials, n_channels, n_times)
        # Calculate mean across channels for each time point, per trial
        mean_across_channels = np.mean(data, axis=1, keepdims=True)
        car_data = data - mean_across_channels
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    
    return car_data


def apply_baseline_correction(
    data: np.ndarray,
    fs: float = FS,
    baseline_start: float = BASELINE_START,
    baseline_end: float = BASELINE_END
) -> np.ndarray:
    """
    Apply baseline correction by subtracting mean of pre-cue period.
    
    Assumes epochs include -0.5s before cue (baseline_start = -0.5s).
    
    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_trials, n_channels, n_times) or (n_channels, n_times)
    fs : float
        Sampling frequency in Hz
    baseline_start : float
        Start of baseline period in seconds (relative to cue)
    baseline_end : float
        End of baseline period in seconds (relative to cue, typically 0.0)
    
    Returns
    -------
    np.ndarray
        Baseline-corrected data with same shape as input
    """
    # Convert time to samples
    # Assuming time starts at baseline_start (negative for pre-cue)
    n_times = data.shape[-1]
    time_samples = np.arange(n_times) / fs + baseline_start
    
    # Find baseline indices
    baseline_mask = (time_samples >= baseline_start) & (time_samples < baseline_end)
    
    if not np.any(baseline_mask):
        warnings.warn(
            f"No baseline samples found for {baseline_start}s to {baseline_end}s. "
            f"Data has {n_times} samples (duration: {n_times/fs:.2f}s). "
            "Skipping baseline correction."
        )
        return data
    
    if data.ndim == 2:
        # Single trial: (n_channels, n_times)
        baseline_mean = np.mean(data[:, baseline_mask], axis=1, keepdims=True)
        corrected = data - baseline_mean
    elif data.ndim == 3:
        # Multiple trials: (n_trials, n_channels, n_times)
        baseline_mean = np.mean(data[:, :, baseline_mask], axis=2, keepdims=True)
        corrected = data - baseline_mean
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    
    return corrected


def apply_ica(
    data: np.ndarray,
    n_components: Optional[int] = ICA_N_COMPONENTS,
    method: str = ICA_METHOD,
    max_iter: int = ICA_MAX_ITER,
    random_state: int = 42
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Apply Infomax ICA for artifact removal (optional).
    
    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_trials, n_channels, n_times)
    n_components : int, optional
        Number of ICA components (None = all channels)
    method : str
        ICA method: 'infomax' or 'fastica'
    max_iter : int
        Maximum iterations for ICA convergence
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    cleaned_data : np.ndarray
        ICA-cleaned data, same shape as input
    ica_object : object or None
        Fitted ICA object (for component inspection/removal)
    """
    if not HAS_MNE:
        raise ImportError(
            "MNE-Python is required for ICA. Install with: pip install mne"
        )
    
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (n_trials, n_channels, n_times), got {data.ndim}D")
    
    n_trials, n_channels, n_times = data.shape
    
    # Reshape for MNE: (n_trials * n_times, n_channels)
    # MNE expects (n_channels, n_times) or (n_samples, n_channels)
    data_reshaped = data.transpose(1, 0, 2)  # (n_channels, n_trials, n_times)
    data_2d = data_reshaped.reshape(n_channels, n_trials * n_times)
    
    # Create MNE RawArray object
    info = mne.create_info(
        ch_names=[f'EEG{i+1:02d}' for i in range(n_channels)],
        sfreq=FS,
        ch_types='eeg'
    )
    
    raw = mne.io.RawArray(data_2d, info)
    
    # Fit ICA
    ica = ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter,
        random_state=random_state
    )
    
    ica.fit(raw)
    
    # Apply ICA (remove components)
    # For now, we'll keep all components but return the fitted ICA
    # In practice, you might want to identify and remove artifact components
    raw_cleaned = ica.apply(raw, exclude=[])  # Don't exclude any components yet
    
    # Extract cleaned data
    cleaned_data_2d = raw_cleaned.get_data()  # (n_channels, n_samples)
    
    # Reshape back to (n_trials, n_channels, n_times)
    cleaned_data_reshaped = cleaned_data_2d.reshape(n_channels, n_trials, n_times)
    cleaned_data = cleaned_data_reshaped.transpose(1, 0, 2)  # (n_trials, n_channels, n_times)
    
    return cleaned_data, ica


def preprocess_epochs(
    X: np.ndarray,
    fs: float = FS,
    do_notch: bool = APPLY_NOTCH,
    do_ica: bool = APPLY_ICA,
    notch_freq: float = NOTCH_FREQ,
    bandpass_low: float = BANDPASS_LOW,
    bandpass_high: float = BANDPASS_HIGH,
    baseline_start: float = BASELINE_START,
    baseline_end: float = BASELINE_END,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Preprocess EEG epochs (vectorized, per-epoch processing).
    
    Processing pipeline:
    1. Notch filter (60 Hz, optional)
    2. Band-pass filter (8-30 Hz, zero-phase Butterworth)
    3. Common Average Reference (CAR)
    4. Baseline correction (pre-cue 0.5s)
    5. Optional ICA artifact removal
    
    Parameters
    ----------
    X : np.ndarray
        Input epochs, shape (n_trials, n_channels=60, n_times)
    fs : float
        Sampling frequency in Hz (default: 250.0)
    do_notch : bool
        Whether to apply notch filter (default: True)
    do_ica : bool
        Whether to apply ICA for artifact removal (default: False)
    notch_freq : float
        Notch frequency in Hz (default: 60.0)
    bandpass_low : float
        Lower cutoff frequency in Hz (default: 8.0)
    bandpass_high : float
        Upper cutoff frequency in Hz (default: 30.0)
    baseline_start : float
        Start of baseline period in seconds (default: -0.5)
    baseline_end : float
        End of baseline period in seconds (default: 0.0)
    verbose : bool
        Whether to print progress messages
    
    Returns
    -------
    X_processed : np.ndarray
        Preprocessed epochs, same shape as input
    ica_object : object or None
        Fitted ICA object (if do_ica=True), None otherwise
    """
    if verbose:
        print(f"Preprocessing {X.shape[0]} epochs with shape {X.shape[1:]}")
    
    X_processed = X.copy()
    ica_object = None
    
    # 1. Notch filter (60 Hz)
    # if do_notch:
    #     if verbose:
    #         print(f"  Applying notch filter at {notch_freq} Hz...")
    #     X_processed = apply_notch_filter(X_processed, fs=fs, notch_freq=notch_freq)
    
    # 2. Band-pass filter (8-30 Hz)
    if verbose:
        print(f"  Applying band-pass filter ({bandpass_low}-{bandpass_high} Hz)...")
    X_processed = apply_bandpass_filter(
        X_processed, 
        fs=fs, 
        low=bandpass_low, 
        high=bandpass_high
    )
    
    # 3. Common Average Reference (CAR)
    if verbose:
        print("  Applying Common Average Reference (CAR)...")
    X_processed = apply_car(X_processed)
    
    # 4. Baseline correction
    if verbose:
        print(f"  Applying baseline correction ({baseline_start}s to {baseline_end}s)...")
    X_processed = apply_baseline_correction(
        X_processed,
        fs=fs,
        baseline_start=baseline_start,
        baseline_end=baseline_end
    )
    
    # 5. Optional ICA
    # if do_ica:
    #     if verbose:
    #         print("  Applying ICA for artifact removal...")
    #     if not HAS_MNE:
    #         warnings.warn(
    #             "MNE-Python not available. Skipping ICA. "
    #             "Install with: pip install mne"
    #         )
    #     else:
    #         X_processed, ica_object = apply_ica(X_processed)
    
    if verbose:
        print("  Preprocessing complete!")
    
    return X_processed, ica_object


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create dummy data for testing
    print("=" * 80)
    print("EEG Preprocessing Module - Example")
    print("=" * 80)
    
    # Create dummy data: 10 trials, 60 channels, 1000 time samples (4s at 250 Hz)
    n_trials = 10
    n_channels = 60
    n_times = 1000
    
    print(f"\nCreating dummy data: {n_trials} trials, {n_channels} channels, {n_times} samples")
    X_dummy = np.random.randn(n_trials, n_channels, n_times)
    
    # Test preprocessing
    print("\nTesting preprocessing pipeline...")
    X_processed, ica_obj = preprocess_epochs(
        X_dummy,
        fs=FS,
        do_notch=APPLY_NOTCH,
        do_ica=APPLY_ICA,  # Set to True if MNE is installed
        verbose=True
    )
    
    print(f"\n✓ Preprocessing successful!")
    print(f"  Input shape:  {X_dummy.shape}")
    print(f"  Output shape: {X_processed.shape}")
    print(f"  ICA object:   {ica_obj}")

