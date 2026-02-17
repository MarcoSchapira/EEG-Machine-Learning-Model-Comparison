from load_raw_file import load_raw_file
from load_all_trials import load_all_trials
import numpy as np
import matplotlib.pyplot as plt

run_load_raw_file = False
run_load_all_trials = True

subjects = [1, 2, 3, 4, 5]
sessions = [1, 2, 3]
AllActions = ["reaching", "multigrasp", "twist"]
selected_nodes = [
    16,  # C3
    17,  # C1
    15,  # C5
    21,  # CP3
    22,  # CP1
    12,  # FC3
    13,  # FC1
    11,  # FC5
]
target_label = 1 
n_trials = 20
rows, cols = 4, 5

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


if run_load_raw_file == True:
    labels, trial_data = load_raw_file(
        mat_path="/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files/EEG_session1_sub1_multigrasp_realMove_compact.mat",
    )

    print("Labels shape: ", labels.shape)
    print("Trial data shape: ", trial_data.shape)

    plot_single_trial_surface(trial_data, labels, trial_index=0, twoDor3d="2d")
    plt.show()


if run_load_all_trials == True:

    all_trial_data, all_labels = load_all_trials(subjects, sessions, AllActions, selected_nodes)
    print("All labels shape: ", all_labels.shape)
    print("All trial data shape: ", all_trial_data.shape)

    np.set_printoptions(threshold=np.inf)
    print("All labels: ", all_labels)


    # 1) Get all indices that match the label you want
    matching_idx = np.where(all_labels == target_label)[0]

    # 2) Take up to n_trials of them
    plot_idx = matching_idx[:n_trials]

    print(f"Found {len(matching_idx)} trials with label == {target_label}. Plotting {len(plot_idx)}.")

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4, rows * 3),
        constrained_layout=True
    )

    axes = axes.flatten()

    # 3) Plot contiguously (no gaps)
    for ax_i, trial_index in enumerate(plot_idx):
        plot_single_trial_surface(
            all_trial_data,
            all_labels,
            trial_index=trial_index,
            twoDor3d="2d",
            ax=axes[ax_i]
        )

    # 4) Hide unused axes
    for ax in axes[len(plot_idx):]:
        ax.axis("off")

    plt.show()
