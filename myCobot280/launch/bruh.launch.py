import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration

def generate_launch_description():
    res = []

    # 1. URDF Model Path
    model_launch_arg = DeclareLaunchArgument(
        "model",
        default_value=os.path.join(
            get_package_share_directory("mycobot_description"),
            "urdf/mycobot_280_rdkx5/mycobot_280_rdkx5_adaptive_gripper.urdf"
        )
    )
    res.append(model_launch_arg)

    # Foxglove Bridge
    foxglove_bridge_node = Node(
        name="foxglove_bridge",
        package="foxglove_bridge",
        executable="foxglove_bridge",
        parameters=[{'port': 8765}]
    )
    res.append(foxglove_bridge_node)

    # 3. Robot Description (Xacro/URDF)
    robot_description = ParameterValue(
        Command(['xacro ', LaunchConfiguration('model')]),
        value_type=str
    )

    # 4. Robot State Publisher
    robot_state_publisher_node = Node(
        name="robot_state_publisher",
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description}],
    )
    res.append(robot_state_publisher_node)

    sim_mode_arg = DeclareLaunchArgument(
        "simulation_mode",
        default_value="false",
        description="Set to 'true' to force simulation mode"
    )
    res.append(sim_mode_arg)

    # Robot Controller Node
    robot_controller_node = Node(
        package="mycobot_280_rdkx5",
        executable="robot_controller",
        name="robot_controller",
        parameters=[
            {'port': '/dev/ttyUSB0'},
            {'baud': 1000000},
            {'force_simulation': LaunchConfiguration("simulation_mode")}
        ]
    )
    res.append(robot_controller_node)

    # ML Inference Node - UPDATED WITH EEG PARAMETERS
    ml_node = Node(
        package="mycobot_280_rdkx5",
        executable="ml_inference",
        name="ml_inference",
        parameters=[
            # EMG Parameters
            {'data_root': '/home/lbran/1subject_TEST'}, 
            {'model_path': '/home/lbran/emg_model_6ch_tcn_lstm_multi.pth'},
            
            # EEG Parameters (Update these paths to point to your .pt and .pth files)
            {'eeg_data_path': '/home/lbran/EEGcode/EEG_Presentation/C_TCNet/sub_9_test_split.pt'},
            {'eeg_model_path': '/home/lbran/EEGcode/EEG_Presentation/C_TCNet/model_9_Production.pth'}
        ]
    )
    res.append(ml_node)

    # 6. Teleop Node (Your bruh.py)
    teleop_node = Node(
        package="mycobot_280_rdkx5",
        executable="bruh", 
        name="teleop_keyboard",
        output="screen",
        parameters=[
            {'port': '/dev/ttyUSB0'},
            {'baud': 1000000},
            {'force_simulation': LaunchConfiguration("simulation_mode")}
        ],
        prefix="xterm -e" 
    )
    res.append(teleop_node)

    return LaunchDescription(res)