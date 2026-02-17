import os
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt

# ----------------------------
# Configuration (EDIT PATHS)
# ----------------------------
MAT_DATA_DIR = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files"
PREPROCESSED_DIR = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_PT_files"

ALL_SESSIONS = [1, 2, 3] #! order here must match precessMatToTensor.py
ALL_ACTIONS = ["multigrasp", "reaching", "twist"] #! order here must match precessMatToTensor.py

LABEL_REMAPPING = {
    "reaching": {11: 0, 21: 1, 31: 2, 41: 3, 51: 4, 61: 5, 8: 11},
    "multigrasp": {11: 6, 21: 7, 61: 8, 8: 11},
    "twist": {91: 9, 101: 10, 8: 11},
}

# ----------------------------
# Helpers
# ----------------------------
def get_mat_path(subject: int, session: int, action: str) -> str:
    filename = f"EEG_session{session}_sub{subject}_{action}_realMove_compact.mat"
    return os.path.join(MAT_DATA_DIR, filename)

def get_pt_path(subject: int, trial_number: int) -> str:
    return os.path.join(PREPROCESSED_DIR, f"sub_{subject}", f"trial_{trial_number:05d}.pt")

def remap_labels(labels_raw: np.ndarray, action: str) -> np.ndarray:
    mapping = LABEL_REMAPPING.get(action, {})
    remapped = np.copy(labels_raw)
    for old_label, new_label in mapping.items():
        remapped[labels_raw == old_label] = new_label
    return remapped

def find_source_mat_for_subject_trial(subject: int, trial_number: int):
    count = 0
    for sess in ALL_SESSIONS:
        for action in ALL_ACTIONS:
            mat_path = get_mat_path(subject, sess, action)
            if not os.path.exists(mat_path):
                continue

            mat = sio.loadmat(mat_path)
            trial_data = mat["trial_data"]
            labels_raw = np.asarray(mat["labels"]).squeeze()

            n_trials = trial_data.shape[0]

            if count <= trial_number < count + n_trials:
                local_i = trial_number - count
                remapped_labels = remap_labels(labels_raw, action)
                return (
                    mat_path,
                    sess,
                    action,
                    local_i,
                    trial_data[local_i],
                    labels_raw[local_i],
                    int(remapped_labels[local_i]),
                )

            count += n_trials

    raise FileNotFoundError("Could not locate matching trial in MAT files.")


# ----------------------------
# Main
# ----------------------------
def main():
    SUBJECT = 1
    TRIAL_NUMBER = 1

    # ---- Load PT ----
    pt_path = get_pt_path(SUBJECT, TRIAL_NUMBER)
    pt_obj = torch.load(pt_path, map_location="cpu")
    pt_data = pt_obj["data"].numpy()
    pt_label = int(pt_obj["label"].item())

    # ---- Load MAT equivalent ----
    mat_path, sess, action, local_i, mat_trial, mat_label_raw, mat_label_remapped = \
        find_source_mat_for_subject_trial(SUBJECT, TRIAL_NUMBER)

    print("\n===== Verification =====")
    print(f"Subject: {SUBJECT}, Trial: {TRIAL_NUMBER}")
    print(f"Session: {sess}, Action: {action}")
    print(f"PT label: {pt_label}")
    print(f"MAT raw label: {mat_label_raw}")
    print(f"MAT remapped label: {mat_label_remapped}")
    print(f"Label match? {pt_label == mat_label_remapped}")

    # ---- Compare data numerically ----
    max_abs_diff = np.max(np.abs(pt_data - mat_trial))
    print(f"Max absolute difference in data: {max_abs_diff:.6g}")

    # ---- Plot side-by-side heatmaps ----
    vmin = min(mat_trial.min(), pt_data.min())
    vmax = max(mat_trial.max(), pt_data.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    im1 = axes[0].imshow(
        mat_trial,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("MAT File Trial")
    axes[0].set_xlabel("Time Samples")
    axes[0].set_ylabel("Channels")

    im2 = axes[1].imshow(
        pt_data,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("PT File Trial")
    axes[1].set_xlabel("Time Samples")

    fig.colorbar(im2, ax=axes, shrink=0.8, label="Amplitude")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
