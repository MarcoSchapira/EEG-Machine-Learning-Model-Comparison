# ------------------------------------------------------------
# Example Usage of Visualization Functions
# This script demonstrates how to use the new visualization functions
# ------------------------------------------------------------
import matplotlib.pyplot as plt
from pathlib import Path

# Import visualization functions
from visualize import (
    visualize_single_file_node,
    visualize_full_action,
    visualize_3d_by_trials,
    visualize_3d_by_channels,
    visualize_3d_by_triggers,
    visualize_debug_3d
)

# Example file path (update this to your actual path)
EXAMPLE_FILE = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files/EEG_session1_sub1_multigrasp_realMove_compact.mat"


def example_1_single_file_node():
    """Example 1: Load one file and show data on a graph (existing functionality)"""
    print("Example 1: Visualizing single file node...")
    node_epochs, fig = visualize_single_file_node(
        mat_path=EXAMPLE_FILE,
        trigger_code=61,
        node_number=15
    )
    print(f"Extracted array shape: {node_epochs.shape}")
    plt.show()


def example_2_full_action():
    """Example 2: Load a full action showing all nodes"""
    print("Example 2: Visualizing full action with all nodes...")
    fig = visualize_full_action(
        mat_path=EXAMPLE_FILE,
        trigger_code=61,
        max_nodes=10  # Limit to 10 nodes for faster visualization
    )
    plt.show()


def example_3_3d_by_trials():
    """Example 3: 3D visualization grouped by trials"""
    print("Example 3: 3D visualization by trials...")
    fig = visualize_3d_by_trials(
        mat_path=EXAMPLE_FILE,
        trigger_code=61,
        node_number=15
    )
    plt.show()


def example_4_3d_by_channels():
    """Example 4: 3D visualization grouped by channels"""
    print("Example 4: 3D visualization by channels...")
    fig = visualize_3d_by_channels(
        mat_path=EXAMPLE_FILE,
        trigger_code=61,
        selected_channels=[1, 5, 10, 15, 20, 25, 30]  # Select specific channels
    )
    plt.show()


def example_5_3d_by_triggers():
    """Example 5: 3D visualization comparing multiple triggers"""
    print("Example 5: 3D visualization by triggers...")
    fig = visualize_3d_by_triggers(
        mat_path=EXAMPLE_FILE,
        node_number=15,
        trigger_codes=[8, 11, 21, 61]  # Compare these triggers
    )
    plt.show()


def example_6_debug():
    """Example 6: Debugging visualization"""
    print("Example 6: Debug visualization...")
    fig = visualize_debug_3d(mat_path=EXAMPLE_FILE)
    plt.show()


if __name__ == "__main__":
    # Check if file exists
    if not Path(EXAMPLE_FILE).exists():
        print(f"ERROR: Example file not found at {EXAMPLE_FILE}")
        print("Please update EXAMPLE_FILE path in this script to point to your .mat file")
    else:
        # Run examples (uncomment the ones you want to run)
        
        # Example 1: Single file node (original functionality)
        example_1_single_file_node()
        
        # Example 2: Full action with all nodes
        # example_2_full_action()
        
        # Example 3: 3D by trials
        # example_3_3d_by_trials()
        
        # Example 4: 3D by channels
        # example_4_3d_by_channels()
        
        # Example 5: 3D by triggers
        # example_5_3d_by_triggers()
        
        # Example 6: Debug visualization
        # example_6_debug()
