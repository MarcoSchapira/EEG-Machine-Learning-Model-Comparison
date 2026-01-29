# ------------------------------------------------------------
# Visualization Functions
# Functions for visualizing EEG data using matplotlib
# All functions call backend data_loader functions
# ------------------------------------------------------------
from pathlib import Path
from typing import Union, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib.colors import Normalize
import matplotlib.cm as cm

try:
    from .data_loader import (
        load_single_file,
        load_multiple_files,
        extract_node_epochs,
        extract_trials_by_trigger,
        get_unique_triggers,
        get_data_statistics
    )
except ImportError:
    # Handle direct execution
    from data_loader import (
        load_single_file,
        load_multiple_files,
        extract_node_epochs,
        extract_trials_by_trigger,
        get_unique_triggers,
        get_data_statistics
    )


# ------------------------------------------------------------
# 1. Load one file and show data on a graph (existing functionality)
# ------------------------------------------------------------
def visualize_single_file_node(
    mat_path: Union[str, Path],
    trigger_code: int,
    node_number: int,
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    selected_nodes: Optional[List[int]] = None
):
    """
    Load one file and visualize a specific node for a specific trigger code.
    This is the existing functionality - kept for compatibility.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    trigger_code : int
        Trigger code to filter by (e.g., 8, 11, 21, 61)
    node_number : int
        1-based index of the channel (1..n_channels)
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    
    Returns:
    --------
    node_epochs : np.ndarray
        Shape (n_selected_trials, n_samples) - extracted epochs
    fig : matplotlib.figure.Figure
        Figure object with the 3D surface plot
    """
    # Load data using backend function
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    
    # Extract node epochs
    node_epochs = extract_node_epochs(labels, trial_data, trigger_code, node_number)
    
    # Create 3D surface plot
    n_samples = metadata['n_samples']
    t = np.arange(n_samples)  # Time axis
    ytr = np.arange(node_epochs.shape[0])  # Trial axis
    X, Y = np.meshgrid(t, ytr, indexing="xy")
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    Z = node_epochs
    
    # Surface plot with coolwarm colormap
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Amplitude (µV or raw units)')
    
    # Labels and title
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Selected trial index")
    ax.set_zlabel("Amplitude (µV or raw units)")
    ax.set_title(f"Node {node_number} | Trigger {trigger_code} | {node_epochs.shape[0]} trials")
    
    # Viewing angle
    ax.view_init(elev=25, azim=-135)
    
    plt.tight_layout()
    
    return node_epochs, fig


# ------------------------------------------------------------
# 2. Load a full action showing all nodes
# ------------------------------------------------------------
def visualize_full_action(
    mat_path: Union[str, Path],
    trigger_code: int,
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    max_nodes: Optional[int] = None,
    figsize: tuple = (16, 10),
    selected_nodes: Optional[List[int]] = None
):
    """
    Visualize a full action showing all nodes/channels for a specific trigger code.
    Creates a grid of subplots, one for each node.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    trigger_code : int
        Trigger code to filter by (e.g., 8, 11, 21, 61)
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    max_nodes : int, optional
        Maximum number of nodes to display. If None, displays all nodes.
    figsize : tuple, default=(16, 10)
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with subplots for each node
    """
    # Load data
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    
    # Extract trials for the trigger code
    filtered_labels, filtered_trial_data = extract_trials_by_trigger(
        labels, trial_data, trigger_code
    )
    
    n_channels = metadata['n_channels']
    n_samples = metadata['n_samples']
    n_trials = filtered_trial_data.shape[0]
    
    # Limit number of nodes if specified
    if max_nodes is not None:
        n_channels = min(n_channels, max_nodes)
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                             subplot_kw={'projection': '3d'})
    
    # Flatten axes array for easier indexing
    if n_channels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Time and trial axes
    t = np.arange(n_samples)
    ytr = np.arange(n_trials)
    X, Y = np.meshgrid(t, ytr, indexing="xy")
    
    # Plot each node
    for node_idx in range(n_channels):
        ax = axes[node_idx]
        node_number = node_idx + 1  # 1-based
        
        # Extract node data
        node_data = filtered_trial_data[:, node_idx, :]
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, node_data, cmap='coolwarm', 
                              linewidth=0, antialiased=True)
        
        # Labels and title
        ax.set_xlabel("Time sample\n(0 to {} samples)".format(n_samples - 1))
        ax.set_ylabel("Trial index\n(0 to {} trials)".format(n_trials - 1))
        ax.set_zlabel("Amplitude\n(µV or raw units)")
        ax.set_title(f"Node {node_number}\nTrigger {trigger_code}")
        
        # Viewing angle
        ax.view_init(elev=25, azim=-135)
    
    # Hide unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)
    
    # Overall title
    fig.suptitle(f"Full Action Visualization: Trigger {trigger_code} | "
                 f"{n_trials} trials | {n_channels} nodes", 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    return fig


# ------------------------------------------------------------
# 2b. Visualize single trial with all nodes
# ------------------------------------------------------------
def visualize_single_trial_all_nodes(
    mat_path: Union[str, Path],
    trigger_code: int,
    trial_number: int,
    display_mode: str = "3d",
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    max_nodes: Optional[int] = None,
    figsize: tuple = (14, 10),
    selected_nodes: Optional[List[int]] = None
):
    """
    Visualize a single trial showing all nodes.
    Can display in either 3D (surface plot) or 2D (heatmap) mode.
    This goes one step deeper than visualize_full_action - selects a specific trial.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    trigger_code : int
        Trigger code to filter by (e.g., 8, 11, 21, 61)
    trial_number : int
        0-based index of the trial within the filtered trials (0 to n_trials-1)
    display_mode : str, default="3d"
        Display mode: "3d" for 3D surface plot, "2d" for 2D heatmap
        - "3d": X-axis: Time samples, Y-axis: Node/channel index, Z-axis: Signal amplitude
        - "2d": X-axis: Time samples, Y-axis: Node number, Color: Amplitude
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    max_nodes : int, optional
        Maximum number of nodes to display. If None, displays all nodes.
    figsize : tuple, default=(14, 10)
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with either 3D surface plot or 2D heatmap showing all nodes
    """
    # Validate display_mode
    if display_mode not in ["2d", "3d"]:
        raise ValueError(f"display_mode must be either '2d' or '3d'. Got '{display_mode}'.")
    
    # Load data
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    
    # Extract trials for the trigger code
    filtered_labels, filtered_trial_data = extract_trials_by_trigger(
        labels, trial_data, trigger_code
    )
    
    n_channels = metadata['n_channels']
    n_samples = metadata['n_samples']
    n_trials = filtered_trial_data.shape[0]
    
    # Validate trial number
    if trial_number < 0 or trial_number >= n_trials:
        raise ValueError(
            f"trial_number must be in [0, {n_trials-1}] for trigger {trigger_code}. "
            f"Got {trial_number}."
        )
    
    # Limit number of nodes if specified
    if max_nodes is not None:
        n_channels = min(n_channels, max_nodes)
    
    # Extract the specific trial
    single_trial_data = filtered_trial_data[trial_number, :n_channels, :]  # (n_channels, n_samples)
    
    if display_mode == "3d":
        # Create 3D figure (matching other 3D graph styling)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        
        # Create meshgrid for 3D surface
        # X-axis: Time samples (0 to n_samples-1)
        # Y-axis: Node indices (0 to n_channels-1)
        t = np.arange(n_samples)
        nodes = np.arange(n_channels)
        X, Y = np.meshgrid(t, nodes, indexing="xy")
        
        # Z-axis: Signal amplitude values
        Z = single_trial_data  # (n_channels, n_samples)
        
        # Create 3D surface plot (matching other 3D graph styling)
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)
        
        # Add colorbar (matching other 3D graph styling)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Amplitude (µV or raw units)')
        
        # Labels and title (matching other 3D graph styling)
        ax.set_xlabel("Time sample")
        ax.set_ylabel("Node index")
        ax.set_zlabel("Amplitude (µV or raw units)")
        ax.set_title(f"Single Trial: Trigger {trigger_code} | Trial {trial_number} | {n_channels} nodes")
        
        # Viewing angle (matching other 3D graph styling)
        ax.view_init(elev=25, azim=-135)
        
    else:  # display_mode == "2d"
        # Create 2D figure for heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap using pcolormesh
        # X-axis: Time samples, Y-axis: Node numbers
        t = np.arange(n_samples)
        nodes = np.arange(n_channels)
        X, Y = np.meshgrid(t, nodes, indexing="xy")
        
        # Create 2D heatmap where color represents amplitude
        im = ax.pcolormesh(X, Y, single_trial_data, cmap='coolwarm', shading='auto')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label='Amplitude (µV or raw units)')
        
        # Labels and title
        ax.set_xlabel("Time sample")
        ax.set_ylabel("Node number")
        ax.set_title(f"Single Trial Heatmap: Trigger {trigger_code} | Trial {trial_number} | {n_channels} nodes")
        
        # Set y-axis ticks to show node numbers (1-based)
        ax.set_yticks(nodes)
        ax.set_yticklabels([f"Node {i+1}" for i in nodes])
        
        # Set x-axis to show some time samples
        n_ticks = min(10, n_samples)
        tick_positions = np.linspace(0, n_samples - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(t) for t in tick_positions])
        
        # Invert y-axis so Node 1 is at the top
        ax.invert_yaxis()
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    return fig


# ------------------------------------------------------------
# 2c. Visualize all trials for a trigger in a grid
# ------------------------------------------------------------
def visualize_all_trials_grid(
    mat_path: Union[str, Path],
    trigger_code: int,
    max_trials: int = 10,
    display_mode: str = "2d",
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    max_nodes: Optional[int] = None,
    selected_nodes: Optional[List[int]] = None
):
    """
    Visualize all trials for a trigger code in a grid layout.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    trigger_code : int
        Trigger code to filter by (e.g., 8, 11, 21, 61)
    max_trials : int, default=10
        Maximum number of trials to display
    display_mode : str, default="2d"
        Display mode: "2d" for 2D heatmap, "3d" for 3D surface plot
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    max_nodes : int, optional
        Maximum number of nodes to display. If None, displays all nodes.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with grid of subplots showing all trials
    """
    # Load data
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    
    # Extract trials for the trigger code
    filtered_labels, filtered_trial_data = extract_trials_by_trigger(
        labels, trial_data, trigger_code
    )
    
    n_trials = filtered_trial_data.shape[0]
    n_channels = metadata['n_channels']
    n_samples = metadata['n_samples']
    
    # Limit number of nodes if specified
    if max_nodes is not None:
        n_channels = min(n_channels, max_nodes)
        filtered_trial_data = filtered_trial_data[:, :n_channels, :]
    
    # Limit number of trials to display
    n_trials_to_show = min(max_trials, n_trials)
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_trials_to_show)))
    n_rows = int(np.ceil(n_trials_to_show / n_cols))
    
    # Create figure with subplots
    if display_mode == "3d":
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        # Flatten axes array for easier indexing
        if n_trials_to_show == 1:
            axes = np.array([axes])
        axes = axes.flatten()
    
    # Get min/max for consistent color scaling across all trials
    all_data_min = np.min(filtered_trial_data[:n_trials_to_show, :, :])
    all_data_max = np.max(filtered_trial_data[:n_trials_to_show, :, :])
    
    # Plot each trial in a subplot
    for trial_idx in range(n_trials_to_show):
        # Extract the specific trial
        single_trial_data = filtered_trial_data[trial_idx, :, :]  # (n_channels, n_samples)
        
        if display_mode == "3d":
            ax = fig.add_subplot(n_rows, n_cols, trial_idx + 1, projection="3d")
            
            # Create meshgrid for 3D surface
            t = np.arange(n_samples)
            nodes = np.arange(n_channels)
            X, Y = np.meshgrid(t, nodes, indexing="xy")
            Z = single_trial_data
            
            # Create 3D surface plot
            surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)
            
            # Labels and title
            ax.set_xlabel("Time sample")
            ax.set_ylabel("Node index")
            ax.set_zlabel("Amplitude")
            ax.set_title(f"Trial {trial_idx}")
            ax.view_init(elev=25, azim=-135)
            
        else:  # display_mode == "2d"
            ax = axes[trial_idx]
            
            # Create heatmap
            t = np.arange(n_samples)
            nodes = np.arange(n_channels)
            X, Y = np.meshgrid(t, nodes, indexing="xy")
            
            im = ax.pcolormesh(X, Y, single_trial_data, cmap='coolwarm', shading='auto',
                              vmin=all_data_min, vmax=all_data_max)
            
            # Labels and title
            ax.set_xlabel("Time sample")
            ax.set_ylabel("Node number")
            ax.set_title(f"Trial {trial_idx}")
            
            # Invert y-axis so Node 1 is at the top
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Hide unused subplots (for 2D mode)
    if display_mode == "2d":
        for idx in range(n_trials_to_show, len(axes)):
            axes[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle(f"All Trials Grid: Trigger {trigger_code} | {n_trials_to_show} trials | {n_channels} nodes | Mode: {display_mode.upper()}", 
                 fontsize=14, y=0.995)
    
    # Add shared colorbar for 2D mode
    if display_mode == "2d":
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap=plt.cm.coolwarm, norm=Normalize(vmin=all_data_min, vmax=all_data_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                           pad=0.05, shrink=0.8, label='Amplitude (µV or raw units)')
    
    plt.tight_layout()
    
    return fig


# ------------------------------------------------------------
# Raw data visualization - display data before any modifications
# ------------------------------------------------------------
def visualize_raw_data_3d(
    mat_path: Union[str, Path],
    trigger_code: Optional[int] = None,
    node_number: int = 1,
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    selected_nodes: Optional[List[int]] = None
):
    """
    Display raw data from the file in 3D before applying any modifications.
    Similar to the original visualization - shows time × trial × amplitude.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    trigger_code : int, optional
        If provided, filters trials by this trigger code. If None, uses all trials.
    node_number : int, default=1
        1-based index of the channel to visualize
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with 3D surface plot
    """
    # Load raw data
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    
    # Filter by trigger if specified
    if trigger_code is not None:
        filtered_labels, filtered_trial_data = extract_trials_by_trigger(
            labels, trial_data, trigger_code
        )
        title_suffix = f" | Trigger {trigger_code}"
    else:
        filtered_trial_data = trial_data
        filtered_labels = labels
        title_suffix = " | All Triggers"
    
    # Extract node data directly (raw data, no filtering)
    n_trials, n_channels, n_samples = filtered_trial_data.shape
    node_idx = node_number - 1
    
    if not (0 <= node_idx < n_channels):
        raise ValueError(f"node_number must be in [1, {n_channels}] (1-based). Got {node_number}.")
    
    # Extract all trials for the specified node
    node_epochs = filtered_trial_data[:, node_idx, :]  # (n_trials, n_samples)
    
    # Create 3D surface plot
    t = np.arange(n_samples)
    ytr = np.arange(node_epochs.shape[0])
    X, Y = np.meshgrid(t, ytr, indexing="xy")
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    Z = node_epochs
    
    # Surface plot with coolwarm colormap (like original)
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Amplitude (µV or raw units)')
    
    # Labels and title
    ax.set_xlabel("Time Sample (X-axis: 0 to {} samples)".format(n_samples - 1))
    ax.set_ylabel("Trial Index (Y-axis: 0 to {} trials)".format(node_epochs.shape[0] - 1))
    ax.set_zlabel("Amplitude (Z-axis: µV or raw units)")
    ax.set_title(f"Raw Data 3D Visualization: Node {node_number}{title_suffix} | "
                 f"{node_epochs.shape[0]} trials")
    
    # Viewing angle (same as original)
    ax.view_init(elev=25, azim=-135)
    
    plt.tight_layout()
    
    return fig


# ------------------------------------------------------------
# 3. 3D visualization functions for different data groupings
# ------------------------------------------------------------

def visualize_3d_by_trials(
    mat_path: Union[str, Path],
    trigger_code: int,
    node_number: int,
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    selected_nodes: Optional[List[int]] = None
):
    """
    3D visualization grouped by trials (time × trial × amplitude).
    This is similar to the original visualization.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    trigger_code : int
        Trigger code to filter by
    node_number : int
        1-based index of the channel
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with 3D plot
    """
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    node_epochs = extract_node_epochs(labels, trial_data, trigger_code, node_number)
    
    n_samples = metadata['n_samples']
    t = np.arange(n_samples)
    ytr = np.arange(node_epochs.shape[0])
    X, Y = np.meshgrid(t, ytr, indexing="xy")
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    surf = ax.plot_surface(X, Y, node_epochs, cmap='coolwarm', 
                          linewidth=0, antialiased=True)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Amplitude (µV)')
    
    ax.set_xlabel("Time Sample (X-axis: 0 to {} samples)".format(n_samples - 1))
    ax.set_ylabel("Trial Index (Y-axis: 0 to {} trials)".format(node_epochs.shape[0] - 1))
    ax.set_zlabel("Mean Amplitude (Z-axis: µV or raw units)")
    ax.set_title(f"3D Visualization by Trials\nNode {node_number} | Trigger {trigger_code}")
    
    ax.view_init(elev=25, azim=-135)
    plt.tight_layout()
    
    return fig


def visualize_3d_by_channels(
    mat_path: Union[str, Path],
    trigger_code: int,
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    selected_channels: Optional[List[int]] = None,
    selected_nodes: Optional[List[int]] = None
):
    """
    3D visualization grouped by channels (time × channel × amplitude).
    Shows how different channels vary over time for a specific trigger.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    trigger_code : int
        Trigger code to filter by
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    selected_channels : list of int, optional
        List of 1-based channel indices to include. If None, uses all channels.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with 3D plot
    """
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    filtered_labels, filtered_trial_data = extract_trials_by_trigger(
        labels, trial_data, trigger_code
    )
    
    # Average across trials to get mean response per channel
    mean_trial_data = np.mean(filtered_trial_data, axis=0)  # (n_channels, n_samples)
    
    # Select channels if specified
    if selected_channels is not None:
        channel_indices = [ch - 1 for ch in selected_channels]  # Convert to 0-based
        mean_trial_data = mean_trial_data[channel_indices, :]
        channel_labels = [f"Ch {ch}" for ch in selected_channels]
    else:
        channel_indices = list(range(metadata['n_channels']))
        channel_labels = [f"Ch {i+1}" for i in range(metadata['n_channels'])]
    
    n_channels_selected = mean_trial_data.shape[0]
    n_samples = metadata['n_samples']
    
    t = np.arange(n_samples)
    ch = np.arange(n_channels_selected)
    X, Y = np.meshgrid(t, ch, indexing="xy")
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    surf = ax.plot_surface(X, Y, mean_trial_data, cmap='viridis', 
                          linewidth=0, antialiased=True)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Mean Amplitude (µV)')
    
    ax.set_xlabel("Time Sample (X-axis: 0 to {} samples)".format(n_samples - 1))
    ax.set_ylabel("Channel Index (Y-axis: {} channels)".format(n_channels_selected))
    ax.set_zlabel("Mean Amplitude (Z-axis: µV or raw units)")
    ax.set_title(f"3D Visualization by Channels\nTrigger {trigger_code} | "
                 f"Mean across {filtered_trial_data.shape[0]} trials")
    
    # Set y-axis ticks to show channel numbers
    ax.set_yticks(ch)
    ax.set_yticklabels(channel_labels)
    
    ax.view_init(elev=25, azim=-135)
    plt.tight_layout()
    
    return fig


def visualize_3d_by_triggers(
    mat_path: Union[str, Path],
    node_number: int,
    trigger_codes: Optional[List[int]] = None,
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    selected_nodes: Optional[List[int]] = None
):
    """
    3D visualization comparing multiple triggers (time × trigger × amplitude).
    Shows how the same node responds to different trigger codes.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    node_number : int
        1-based index of the channel
    trigger_codes : list of int, optional
        List of trigger codes to compare. If None, uses all unique triggers in the file.
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with 3D plot
    """
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    
    # Get trigger codes
    if trigger_codes is None:
        trigger_codes = get_unique_triggers(labels).tolist()
    
    # Extract mean response for each trigger
    mean_responses = []
    trigger_labels = []
    
    for trigger_code in trigger_codes:
        try:
            node_epochs = extract_node_epochs(labels, trial_data, trigger_code, node_number)
            mean_response = np.mean(node_epochs, axis=0)  # Average across trials
            mean_responses.append(mean_response)
            trigger_labels.append(f"Trigger {trigger_code}")
        except ValueError:
            # Skip triggers with no trials
            continue
    
    if len(mean_responses) == 0:
        raise ValueError("No valid triggers found for visualization")
    
    mean_responses = np.array(mean_responses)  # (n_triggers, n_samples)
    
    n_samples = metadata['n_samples']
    t = np.arange(n_samples)
    trig = np.arange(len(trigger_labels))
    X, Y = np.meshgrid(t, trig, indexing="xy")
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    surf = ax.plot_surface(X, Y, mean_responses, cmap='plasma', 
                          linewidth=0, antialiased=True)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Mean Amplitude (µV)')
    
    ax.set_xlabel("Time Sample (X-axis: 0 to {} samples)".format(n_samples - 1))
    ax.set_ylabel("Trigger Code (Y-axis: {} triggers)".format(len(trigger_labels)))
    ax.set_zlabel("Mean Amplitude (Z-axis: µV or raw units)")
    ax.set_title(f"3D Visualization by Triggers\nNode {node_number}")
    
    # Set y-axis ticks to show trigger codes
    ax.set_yticks(trig)
    ax.set_yticklabels(trigger_labels)
    
    ax.view_init(elev=25, azim=-135)
    plt.tight_layout()
    
    return fig


# ------------------------------------------------------------
# 4. Debugging visualization - useful 3D graph for debugging
# ------------------------------------------------------------
def visualize_debug_3d(
    mat_path: Union[str, Path],
    labels_key: str = "labels",
    trials_key: str = "trial_data",
    selected_nodes: Optional[List[int]] = None
):
    """
    Debugging visualization showing data statistics and structure.
    Creates multiple 3D plots to help understand the data distribution.
    
    Parameters:
    -----------
    mat_path : str or Path
        Path to the .mat file
    labels_key : str, default="labels"
        Key for labels in the .mat file
    trials_key : str, default="trial_data"
        Key for trial data in the .mat file
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with multiple subplots for debugging
    """
    labels, trial_data, metadata = load_single_file(mat_path, labels_key, trials_key, selected_nodes)
    stats = get_data_statistics(labels, trial_data)
    
    unique_triggers = get_unique_triggers(labels)
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: Trial distribution across triggers
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    trigger_counts = [stats['trigger_counts'].get(trig, 0) for trig in unique_triggers]
    x_pos = np.arange(len(unique_triggers))
    y_pos = np.zeros_like(x_pos)
    z_pos = np.zeros_like(x_pos)
    
    # Create bar plot in 3D
    colors = cm.viridis(np.linspace(0, 1, len(unique_triggers)))
    ax1.bar3d(x_pos, y_pos, z_pos, 0.8, 0.8, trigger_counts, 
             color=colors, alpha=0.7)
    
    ax1.set_xlabel("Trigger Code Index")
    ax1.set_ylabel("Y-axis (constant)")
    ax1.set_zlabel("Number of Trials")
    ax1.set_title("Trial Distribution Across Triggers")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"T{trig}" for trig in unique_triggers])
    ax1.view_init(elev=20, azim=45)
    
    # Subplot 2: Mean amplitude across all channels for each trigger
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    
    # Calculate mean response per trigger across all channels
    mean_responses_by_trigger = []
    for trigger_code in unique_triggers:
        try:
            filtered_labels, filtered_trial_data = extract_trials_by_trigger(
                labels, trial_data, trigger_code
            )
            # Average across trials and channels
            mean_response = np.mean(filtered_trial_data, axis=(0, 1))
            mean_responses_by_trigger.append(mean_response)
        except ValueError:
            mean_responses_by_trigger.append(np.zeros(metadata['n_samples']))
    
    mean_responses_by_trigger = np.array(mean_responses_by_trigger)
    
    t = np.arange(metadata['n_samples'])
    trig = np.arange(len(unique_triggers))
    X, Y = np.meshgrid(t, trig, indexing="xy")
    
    surf2 = ax2.plot_surface(X, Y, mean_responses_by_trigger, cmap='coolwarm',
                             linewidth=0, antialiased=True)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=20, label='Mean Amplitude')
    
    ax2.set_xlabel("Time Sample")
    ax2.set_ylabel("Trigger Code")
    ax2.set_zlabel("Mean Amplitude (all channels)")
    ax2.set_title("Mean Response Across All Channels by Trigger")
    ax2.set_yticks(trig)
    ax2.set_yticklabels([f"T{trig}" for trig in unique_triggers])
    ax2.view_init(elev=25, azim=-135)
    
    # Subplot 3: Channel variance across time
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    
    # Calculate variance across trials for each channel
    channel_variance = np.var(trial_data, axis=0)  # (n_channels, n_samples)
    
    # Select a subset of channels if too many
    max_channels_show = 20
    if metadata['n_channels'] > max_channels_show:
        step = metadata['n_channels'] // max_channels_show
        channel_variance = channel_variance[::step, :]
        channel_indices = list(range(0, metadata['n_channels'], step))
    else:
        channel_indices = list(range(metadata['n_channels']))
    
    t = np.arange(metadata['n_samples'])
    ch = np.arange(len(channel_indices))
    X, Y = np.meshgrid(t, ch, indexing="xy")
    
    surf3 = ax3.plot_surface(X, Y, channel_variance, cmap='hot',
                             linewidth=0, antialiased=True)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=20, label='Variance')
    
    ax3.set_xlabel("Time Sample")
    ax3.set_ylabel("Channel Index")
    ax3.set_zlabel("Variance Across Trials")
    ax3.set_title("Channel Variance Across Time")
    ax3.view_init(elev=25, azim=-135)
    
    # Subplot 4: Data statistics text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = f"""
    Data Statistics:
    ================
    
    File: {Path(mat_path).name}
    
    Shape: {stats['data_shape']}
    Data Type: {stats['data_dtype']}
    
    Number of Trials: {stats['n_trials']}
    Number of Channels: {stats['n_channels']}
    Number of Samples: {stats['n_samples']}
    
    Unique Triggers: {stats['unique_triggers']}
    
    Trigger Counts:
    """
    for trigger, count in stats['trigger_counts'].items():
        stats_text += f"  Trigger {trigger}: {count} trials\n"
    
    stats_text += f"""
    
    Data Range:
    Min: {stats['data_min']:.4f}
    Max: {stats['data_max']:.4f}
    Mean: {stats['data_mean']:.4f}
    Std: {stats['data_std']:.4f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)
    
    fig.suptitle("Debug Visualization: Data Overview", fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig
