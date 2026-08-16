import os
import fcntl
import time
import math
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from pymycobot import MyCobot280RDKX5

HW_CMD_CLOSE = 0  
HW_CMD_OPEN  = 1  

ARM_JOINT_NAMES = [
    'joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3', 
    'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6'
]

GRIPPER_JOINT_NAMES = [
    "gripper_controller",             
    "gripper_base_to_gripper_left2",  
    "gripper_left3_to_gripper_left1", 
    "gripper_base_to_gripper_right3", 
    "gripper_base_to_gripper_right2", 
    "gripper_right3_to_gripper_right1"
]

REST_POSE = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
LOCK_FILE = "/tmp/mycobot_lock"

PRESET_ACTIONS = {
    "Arm-reaching: Forward": [0.0, -45.0, -45.0, 0.0, 0.0, 0.0],           
    "Arm-reaching: Backward": [0.0, 45.0, -135.0, 0.0, 0.0, 0.0],         
    "Arm-reaching: Up": [0.0, 20.0, -50.0, 0.0, 0.0, 0.0],              
    "Arm-reaching: Down": [0.0, -20.0, -130.0, 0.0, 0.0, 0.0],             
    "Arm-reaching: Left": [90.0, -45.0, -45.0, 0.0, 0.0, 0.0],              
    "Arm-reaching: Right": [-90.0, -45.0, -45.0, 0.0, 0.0, 0.0],
    "Wrist-twisting: Pronation": [0.0, 0.0, -90.0, -45.0, 0.0, 0.0],
    "Wrist-twisting: Supination": [0.0, 0.0, -90.0, 45.0, 0.0, 0.0],
}

def acquire(lock_file):
    try:
        fd = os.open(lock_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (IOError, OSError):
        return None

def release(fd):
    if fd:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller_node')
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 1000000)
        self.declare_parameter('force_simulation', False)
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        force_sim = self.get_parameter('force_simulation').value
        
        self.connected = False
        self.mc = None
        self.current_joints = list(REST_POSE) 
        self.speed = 50
        self.visual_gripper_val = -0.5 
        self.is_moving = False

        if force_sim:
            self.get_logger().info("FORCED SIMULATION MODE.")
        else:
            try:
                self.mc = MyCobot280RDKX5(port, baud)
                lock = acquire(LOCK_FILE)
                if lock:
                    self.mc.init_gripper()       
                    time.sleep(0.1)
                    self.mc.set_gripper_mode(0)
                    time.sleep(0.1)
                    actual_angles = self.mc.get_angles()
                    if actual_angles:
                        self.current_joints = actual_angles
                        self.connected = True
                        self.get_logger().info("Hardware Connected.")
                    release(lock)
            except Exception as e:
                self.get_logger().error(f"Connection failed: {e}. Defaulting to Sim.")
        
        # ROS Setup
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.status_pub = self.create_publisher(String, '/gui/status', 10)
        self.cmd_sub = self.create_subscription(String, '/robot/command', self.command_callback, 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        self.publish_joint_state()
        self.publish_status()

    def publish_status(self):
        hw_state = "HW Connected" if self.connected else "Simulation"
        move_state = "MOVING..." if self.is_moving else "WAITING FOR COMMAND"
        msg = String()
        msg.data = f"[{hw_state}] - {move_state}"
        self.status_pub.publish(msg)

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
        arm_rads = [math.radians(x) for x in self.current_joints]
        g = self.visual_gripper_val
        gripper_pos = [g, g, -g, -g, -g, g]
        msg.position = arm_rads + gripper_pos
        self.joint_pub.publish(msg)

    def command_callback(self, msg):
        command = msg.data
        if self.is_moving:
            self.get_logger().warn("Robot is currently moving, ignoring command.")
            return

        if command == "STOP":
            # Implement hard stop logic if needed, currently resets to rest
            threading.Thread(target=self.execute_movement, args=(REST_POSE,)).start()
        elif command == "RESET" or command == "Rest":
            threading.Thread(target=self.execute_movement, args=(REST_POSE,)).start()
        elif command in PRESET_ACTIONS:
            threading.Thread(target=self.execute_movement, args=(PRESET_ACTIONS[command],)).start()
        elif "Hand-grasping" in command:
            target = 0.15 if "Card" in command else 0.0 # Adjust based on grasp type
            threading.Thread(target=self.move_gripper, args=(target,)).start()

    def execute_movement(self, target_angles, duration=2.0):
        self.is_moving = True
        start_angles = list(self.current_joints)
        steps = 20 
        dt = duration / steps
        for i in range(steps):
            alpha = (i + 1) / steps
            self.current_joints = [s + (e - s) * alpha for s, e in zip(start_angles, target_angles)]
            if self.connected:
                lock = acquire(LOCK_FILE)
                if lock:
                    self.mc.send_angles(self.current_joints, self.speed)
                    release(lock)
            time.sleep(dt)
        
        # Return to rest after movement
        time.sleep(0.5)
        start_angles = list(self.current_joints)
        for i in range(steps):
            alpha = (i + 1) / steps
            self.current_joints = [s + (e - s) * alpha for s, e in zip(start_angles, REST_POSE)]
            if self.connected:
                lock = acquire(LOCK_FILE)
                if lock:
                    self.mc.send_angles(self.current_joints, self.speed)
                    release(lock)
            time.sleep(dt)
        self.is_moving = False

    def move_gripper(self, visual_target):
        self.is_moving = True
        self.visual_gripper_val = visual_target
        if self.connected:
            lock = acquire(LOCK_FILE)
            if lock:
                hw_cmd = HW_CMD_CLOSE if visual_target > 0 else HW_CMD_OPEN
                self.mc.set_gripper_state(hw_cmd, 50)
                release(lock)
        time.sleep(1.0) # Simulate grasp time
        self.visual_gripper_val = -0.5 # Open back up
        if self.connected:
            lock = acquire(LOCK_FILE)
            if lock:
                self.mc.set_gripper_state(HW_CMD_OPEN, 50)
                release(lock)
        self.is_moving = False

def main():
    rclpy.init()
    node = RobotControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()