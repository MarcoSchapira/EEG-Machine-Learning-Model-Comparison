# ------------------------------------------------------------
# New load_data_evaluate for EEG realMove_compact .mat folder
# Replaces the evaluation loader in utils.py without modifying utils.
# Uses visualize.data_loader as base; matches current API (4 returns + nodes_kept).
# ------------------------------------------------------------
from pathlib import Path
import re
from typing import Tuple, List, Optional
import numpy as np


from visualize_matlab.data_loader import (
    load_labels_and_trial_data,
    filter_nodes,
    load_single_file,
)


from scipy.io import loadmat

# Only keep nodes 1--8 (1-based); 60 nodes total in files
NODES_KEPT = list(range(1, 9))  # [1, 2, 3, 4, 5, 6, 7, 8]

# File pattern: EEG_session<N>_sub<M>_<action>_realMove_compact.mat
FILENAME_PATTERN = re.compile(
    r"EEG_session(\d+)_sub(\d+)_(\w+)_realMove_compact\.mat",
    re.IGNORECASE,
)


def _mat_keys_for_file(mat_path: Path) -> Tuple[str, str]:
    """Detect label and trial data keys in a .mat file. Tries visualize convention then utils."""
    md = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    keys = set(k for k in md if not k.startswith("__"))
    if "labels" in keys and "trial_data" in keys:
        return "labels", "trial_data"
    if "label" in keys and "data" in keys:
        return "label", "data"
    raise KeyError(
        f"Expected keys 'labels'/'trial_data' or 'label'/'data' in {mat_path.name}; found {keys}"
    )


def _load_one_mat(
    mat_path: Path,
    labels_key: str,
    trials_key: str,
    selected_nodes: Optional[List[int]] = NODES_KEPT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one .mat file and return (labels_1d, trial_data_3d) with optional node filtering."""
    labels, trial_data = load_labels_and_trial_data(
        mat_path, labels_key=labels_key, trials_key=trials_key
    )
    if selected_nodes is not None:
        trial_data = filter_nodes(trial_data, selected_nodes)
    return labels, trial_data


def _collect_realmove_files(dir_path: str) -> List[Tuple[Path, int, int, str]]:
    """List all *realMove*compact*.mat in dir_path; return (path, session, subject, action)."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"dir_path is not a directory: {dir_path}")
    out = []
    for p in dir_path.glob("*realMove*compact*.mat"):
        m = FILENAME_PATTERN.match(p.name)
        if m:
            session, sub, action = int(m.group(1)), int(m.group(2)), m.group(3)
            out.append((p, session, sub, action))
    return out


def load_data_evaluate(
    dir_path: str,
    dataset_type: str,
    n_sub: int,
    trainOnEveryone: Optional[bool] = None,
    mode_evaluate: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load EEG realMove_compact .mat data for evaluation.

    Matches the API of utils.load_data_evaluate (same 4 return arrays) and adds
    a 5th return: array of node indices that were kept (1-based).

    Parameters
    ----------
    dir_path : str
        Directory containing EEG_session*_sub*_*_realMove_compact.mat files.
    dataset_type : str
        Kept for API compatibility with utils.load_data_evaluate; not used.
    n_sub : int
        Subject id (1..25). If trainOnEveryone True: this subject is used as
        test set and all others as train. If trainOnEveryone False: only this
        subject is loaded and split by session (train = sessions 1,2; test = session 3).
    trainOnEveryone : bool, optional
        If True (LOSO-style): train on all subjects except n_sub, test on n_sub.
        If False: load only n_sub, use sessions 1 and 2 for train and session 3 for test.
    mode_evaluate : str, optional
        Backward compatibility: "LOSO" -> trainOnEveryone=True; any other value
        (e.g. "subject-dependent") -> trainOnEveryone=False. Ignored if trainOnEveryone is set.

    Returns
    -------
    X_train : np.ndarray
        Shape (n_train_trials, n_nodes, n_samples) with n_nodes = 8.
    y_train : np.ndarray
        Shape (n_train_trials, 1) integer labels.
    X_test : np.ndarray
        Shape (n_test_trials, n_nodes, n_samples).
    y_test : np.ndarray
        Shape (n_test_trials, 1) integer labels.
    nodes_kept : np.ndarray
        1-based indices of channels kept, shape (8,) -> [1,2,3,4,5,6,7,8].
    """
    if trainOnEveryone is None:
        trainOnEveryone = (mode_evaluate == "LOSO") if mode_evaluate is not None else True
    files_with_meta = _collect_realmove_files(dir_path)
    if not files_with_meta:
        raise FileNotFoundError(
            f"No *realMove*compact*.mat files found in {dir_path}"
        )

    # Detect .mat keys once from first file
    first_path = files_with_meta[0][0]
    labels_key, trials_key = _mat_keys_for_file(first_path)

    def load_files(file_list: List[Tuple[Path, int, int, str]]) -> Tuple[np.ndarray, np.ndarray]:
        all_labels, all_data = [], []
        for (p, _s, _sub, _a) in file_list:
            lab, dat = _load_one_mat(p, labels_key, trials_key, NODES_KEPT)
            all_labels.append(lab)
            all_data.append(dat)
        if not all_labels:
            return np.array([]).reshape(0, 1), np.array([]).reshape(0, len(NODES_KEPT), 0)
        labels_cat = np.concatenate(all_labels, axis=0)
        data_cat = np.concatenate(all_data, axis=0)
        # Downstream expects labels (n_trials, 1) for transpose + [0]
        if labels_cat.ndim == 1:
            labels_cat = labels_cat[:, np.newaxis]
        return labels_cat, data_cat

    if trainOnEveryone:
        train_files = [(p, s, sub, a) for (p, s, sub, a) in files_with_meta if sub != n_sub]
        test_files = [(p, s, sub, a) for (p, s, sub, a) in files_with_meta if sub == n_sub]
    else:
        sub_files = [(p, s, sub, a) for (p, s, sub, a) in files_with_meta if sub == n_sub]
        train_files = [(p, s, sub, a) for (p, s, sub, a) in sub_files if s in (1, 2)]
        test_files = [(p, s, sub, a) for (p, s, sub, a) in sub_files if s == 3]

    y_train, X_train = load_files(train_files)
    y_test, X_test = load_files(test_files)

    nodes_kept = np.array(NODES_KEPT, dtype=np.int64)
    return X_train, y_train, X_test, y_test
