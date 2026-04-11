import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path so we can import visualize package
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualize import (
    visualize_full_action, 
    visualize_debug_3d, 
    visualize_raw_data_3d, 
    visualize_single_trial_all_nodes,
    visualize_all_trials_grid
)

mat_path = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files/EEG_session1_sub1_reaching_realMove_compact.mat"

# Optional: Specify which nodes to keep (1-based indices). If None, keeps all nodes.
# Example: selected_nodes = [1, 5, 10, 15, 20]  # Keep only nodes 1, 5, 10, 15, 20
minimal_nodes = [
    16,  # C3
    17,  # C1
    15,  # C5
    21,  # CP3
    22,  # CP1
    12,  # FC3
    13,  # FC1
    11,  # FC5
]

robust_nodes = [
    13,  # FC1
    12,  # FC3
    11,  # FC5
    17,  # C1
    16,  # C3
    15,  # C5
    18,  # Cz
    22,  # CP1
    21,  # CP3
    20,  # CP5
    23,  # CPz
]

all_nodes = list(range(1, 60))

selected_nodes = minimal_nodes

run_raw_data_3d = False
run_full_action_all_trials = False
run_single_trial_all_nodes = True
run_debug_3d = False
run_trigger_code_all_nodes = False

# Raw data 3D visualization (before any modifications)
if run_raw_data_3d:
    fig = visualize_raw_data_3d(
        mat_path=mat_path,
        trigger_code=11,
        node_number=15,
        #selected_nodes=selected_nodes
    )
    plt.show()

# Visualize full action with all nodes
if run_full_action_all_trials:
    fig = visualize_full_action(
        mat_path=mat_path,
        trigger_code=11,
        selected_nodes=selected_nodes
    )
    plt.show()

if run_single_trial_all_nodes:
    fig = visualize_single_trial_all_nodes(
        mat_path=mat_path, 
        trigger_code=21, 
        trial_number=0, 
        display_mode="2d",
        selected_nodes=selected_nodes
    )
    plt.show()

if run_trigger_code_all_nodes:
    fig = visualize_all_trials_grid(
        mat_path=mat_path,
        trigger_code=21,
        max_trials=20,
        display_mode="2d",
        selected_nodes=selected_nodes
    )
    plt.show()

# Debug visualization
if run_debug_3d:
    fig = visualize_debug_3d(
        mat_path=mat_path,
        #selected_nodes=selected_nodes
    )
    plt.show()
