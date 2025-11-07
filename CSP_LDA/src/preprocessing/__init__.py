"""
Preprocessing package for EEG motor decoding pipeline.
"""

from .preprocess import (
    preprocess_epochs,
    apply_notch_filter,
    apply_bandpass_filter,
    apply_car,
    apply_baseline_correction,
    apply_ica,
    FS,
    NOTCH_FREQ,
    APPLY_NOTCH,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    BANDPASS_ORDER,
    BASELINE_START,
    BASELINE_END,
    ICA_METHOD,
    ICA_N_COMPONENTS,
    ICA_MAX_ITER
)

__all__ = [
    'preprocess_epochs',
    'apply_notch_filter',
    'apply_bandpass_filter',
    'apply_car',
    'apply_baseline_correction',
    'apply_ica',
    'FS',
    'NOTCH_FREQ',
    'APPLY_NOTCH',
    'BANDPASS_LOW',
    'BANDPASS_HIGH',
    'BANDPASS_ORDER',
    'BASELINE_START',
    'BASELINE_END',
    'ICA_METHOD',
    'ICA_N_COMPONENTS',
    'ICA_MAX_ITER',
]

