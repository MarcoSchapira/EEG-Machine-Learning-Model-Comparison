import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Config: subject, trial range, and label to show ---
DATA_DIR = Path("/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_PT_files")
SUBJECT = "sub_1"
TRIAL_END = 500   # inclusive: trial_00000.pt through trial_03300.pt
TARGET_LABEL = 1   # only show trials with this label

# Grid layout: number of columns for the 2D plots
N_COLS = 5


def plot_single_trial_surface(
    all_trial_data: np.ndarray,
    all_labels: np.ndarray,
    trial_index: int = 0,
    title_suffix: str = "",
    twoDor3d: str = "3d",
    ax=None
):
    trialToShow = all_trial_data[trial_index]
    labelToShow = all_labels[trial_index]

    n_nodes, n_samples = trialToShow.shape
    t = np.arange(n_samples)
    y_nodes = np.arange(n_nodes)
    X, Y = np.meshgrid(t, y_nodes, indexing="xy")
    Z = trialToShow

    # Create figure ONLY if ax not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        if twoDor3d == "3d":
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if twoDor3d == "3d":
        surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        ax.set_zlabel("Amplitude")

    else:  # 2D
        im = ax.pcolormesh(X, Y, Z, cmap="coolwarm")
        fig.colorbar(im, ax=ax, shrink=0.5, aspect=20)

    ax.set_xlabel("Time Sample")
    ax.set_ylabel("Node Index")
    ax.set_title(f"Trial {trial_index} | Label {labelToShow}")

    return fig


# Load all trials for sub_1 from trial_00000.pt through trial_03300.pt, keep only TARGET_LABEL
sub_dir = DATA_DIR / SUBJECT
all_data_list = []
all_labels_list = []
trial_ids_list = []   # original trial file index for titles

for trial_id in range(0, TRIAL_END + 1):
    file_path = sub_dir / f"trial_{trial_id:05d}.pt"
    if not file_path.exists():
        continue
    state = torch.load(file_path, weights_only=False)
    data = state["data"].numpy()   # (n_nodes, n_samples)
    label = state["label"].numpy()
    if label.ndim == 0:
        label_val = int(label.item())
    else:
        label_val = int(np.atleast_1d(label).flat[0])
    if label_val != TARGET_LABEL:
        continue
    all_data_list.append(data)
    all_labels_list.append(label_val)
    trial_ids_list.append(trial_id)

if not all_data_list:
    print(f"No trials with label {TARGET_LABEL} found for {SUBJECT} (trial_00000.pt .. trial_{TRIAL_END:05d}.pt)")
    raise SystemExit(1)

all_trial_data = np.stack(all_data_list, axis=0)   # (n_trials, n_nodes, n_samples)
all_labels = np.array(all_labels_list)
n_plots = len(trial_ids_list)

print(f"Found {n_plots} trials with label {TARGET_LABEL} for {SUBJECT}.")

# 2D grid of 2D graphs
n_cols = min(N_COLS, n_plots)
n_rows = math.ceil(n_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
if n_plots == 1:
    axes = np.array([[axes]])
elif not isinstance(axes, np.ndarray) or axes.ndim == 1:
    axes = np.atleast_2d(axes)

for idx in range(n_plots):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col]
    plot_single_trial_surface(
        all_trial_data,
        all_labels,
        trial_index=idx,
        twoDor3d="2d",
        ax=ax
    )
    ax.set_title(f"Trial file {trial_ids_list[idx]} | Label {all_labels[idx]}")

# Hide unused subplots
for idx in range(n_plots, n_rows * n_cols):
    row, col = idx // n_cols, idx % n_cols
    axes[row, col].set_visible(False)

plt.tight_layout()
plt.show()