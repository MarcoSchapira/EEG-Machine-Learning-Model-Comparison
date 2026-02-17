from pathlib import Path
from typing import Union, Tuple
import numpy as np
from scipy.io import loadmat



def load_raw_file(
    mat_path: Union[str, Path],

) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load labels and trial data from a MATLAB .mat file.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    
    Returns:
    --------
    labels_1d : np.ndarray
        Shape (n_trials,) - integer labels
    trial_data_3d : np.ndarray
        Shape (n_trials, n_channels, n_samples) - trial data
    """
    labels_key = "labels"
    trials_key = "trial_data"

    mat_path = Path(mat_path)
    md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if labels_key not in md or trials_key not in md:
        raise KeyError(
            "Expected keys '{}' and '{}' not both found in {}".format(
                labels_key, trials_key, mat_path.name
            )
        )

    labels_raw = np.array(md[labels_key])
    trial_data = np.array(md[trials_key], dtype=float)

    # Squeeze labels to 1D (e.g., 300x1 -> 300,)
    labels_1d = labels_raw.squeeze()

    # Ensure integer dtype for labels
    if np.issubdtype(labels_1d.dtype, np.floating):
        labels_1d = labels_1d.astype(int)
    elif labels_1d.dtype.kind in {"S", "U"}:
        labels_1d = labels_1d.astype(int)

    # Basic shape checks
    if trial_data.ndim != 3:
        raise ValueError("'{}' must be 3D (n_trials, n_channels, n_samples); got shape {}".format(
            trials_key, trial_data.shape
        ))
    if labels_1d.ndim != 1 or labels_1d.shape[0] != trial_data.shape[0]:
        raise ValueError("labels length must equal number of trials in trial_data")

    return labels_1d, trial_data