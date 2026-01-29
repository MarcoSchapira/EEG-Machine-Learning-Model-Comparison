from visualize import visualize_full_action, visualize_debug_3d, visualize_raw_data_3d, visualize_single_trial_all_nodes
import matplotlib.pyplot as plt

mat_path = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files/EEG_session1_sub1_multigrasp_realMove_compact.mat"

run_raw_data_3d = True
run_full_action_all_trials = False
run_single_trial_all_nodes = True
run_debug_3d = False

# Raw data 3D visualization (before any modifications)
if run_raw_data_3d:
    fig = visualize_raw_data_3d(
        mat_path=mat_path,
        trigger_code=61,  # Optional: set to None to show all triggers
        node_number=15
    )
    plt.show()

# Visualize full action with all nodes
if run_full_action_all_trials:
    fig = visualize_full_action(
        mat_path=mat_path,
        trigger_code=61
    )
    plt.show()

# Debug visualization
if run_debug_3d:
    # Debug visualization
    fig = visualize_debug_3d(mat_path=mat_path)
    plt.show()

if run_single_trial_all_nodes:
    fig = visualize_single_trial_all_nodes(mat_path=mat_path, trigger_code=11, trial_number=2, display_mode="2d")
    plt.show()

# trigger codes cheat sheet
