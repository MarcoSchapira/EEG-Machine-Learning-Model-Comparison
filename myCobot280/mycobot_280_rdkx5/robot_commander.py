import sys
import os
import fcntl
import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from pymycobot import MyCobot280RDKX5

# ==============================================================================
# HARDWARE CONFIGURATION
# ==============================================================================
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

# Direct mapping from ML class strings to target joint angles
ML_ACTIONS = {
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

class RobotCommanderNode(Node):
    def __init__(self):
        super().__init__('robot_commander_node')
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

        if force_sim:
            self.get_logger().info("FORCED SIMULATION MODE. Hardware disabled.")
        else:
            try:
                self.mc = MyCobot280RDKX5(port, baud)
                lock = acquire(LOCK_FILE)
                if lock:
                    self.get_logger().info("Initializing Gripper...")
                    self.mc.init_gripper()       
                    time.sleep(0.1)
                    self.mc.set_gripper_mode(0)
                    time.sleep(0.1)
                    
                    actual_angles = self.mc.get_angles()
                    if actual_angles:
                        self.current_joints = actual_angles
                        self.connected = True
                        self.get_logger().info("Hardware Connected successfully.")
                    release(lock)
            except Exception as e:
                self.get_logger().error(f"Hardware connection failed: {e}")
        
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # New Subscriber: Listens to the string from your ML node or GUI
        self.cmd_sub = self.create_subscription(String, '/robot/command', self.command_callback, 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.publish_joint_state()

    def command_callback(self, msg):
        cmd = msg.data
        self.get_logger().info(f"Executing command: {cmd}")
        
        # 1. Handle Gripper Grasping Actions
        if "Hand-grasping" in cmd:
            self.move_gripper("close")
            time.sleep(1.5) # Hold the grasp
            self.move_gripper("open")
            return
            
        # 2. Handle Arm Reaching Actions
        if cmd in ML_ACTIONS:
            target_angles = ML_ACTIONS[cmd]
            self.interpolate_joints(target_angles)
            time.sleep(0.5) # Hold position briefly
            self.interpolate_joints(REST_POSE) # Return to start
            return
            
        # 3. Handle System Commands
        if cmd in ["Rest", "RESET"]:
            self.interpolate_joints(REST_POSE)
            self.move_gripper("open")
        elif cmd == "STOP":
            self.get_logger().warn("STOP COMMAND RECEIVED.")

    def sync_robot(self):
        self.publish_joint_state()
        if self.connected:
            lock = acquire(LOCK_FILE)
            if lock:
                self.mc.send_angles(self.current_joints, self.speed)
                release(lock)
    
    def timer_callback(self):
        self.publish_joint_state()

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
        
        arm_rads = [math.radians(x) for x in self.current_joints]
        g = self.visual_gripper_val
        gripper_pos = [g, g, -g, -g, -g, g]

        msg.position = arm_rads + gripper_pos
        self.joint_pub.publish(msg)

    def move_gripper(self, target_state):
        if target_state == "close":
            visual_target = 0.15      
            hw_command = HW_CMD_CLOSE 
        else:
            visual_target = -0.5      
            hw_command = HW_CMD_OPEN  

        self.visual_gripper_val = visual_target
        self.publish_joint_state() 

        if self.connected:
            lock = acquire(LOCK_FILE)
            if lock:
                self.mc.set_gripper_state(hw_command, 50)
                release(lock)

    def interpolate_joints(self, target_angles, duration=1.5):
        start_angles = list(self.current_joints)
        steps = 15 
        dt = duration / steps
        for i in range(steps):
            alpha = (i + 1) / steps
            self.current_joints = [s + (e - s) * alpha for s, e in zip(start_angles, target_angles)]
            self.sync_robot()
            time.sleep(dt)

def main():
    if os.path.exists(LOCK_FILE):
        try: os.remove(LOCK_FILE)
        except: pass

    rclpy.init()
    node = RobotCommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()