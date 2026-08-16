import sys
import os
import fcntl
import termios
import tty
import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pymycobot import MyCobot280RDKX5

# ==============================================================================
# 1. HARDWARE CONFIGURATION (ADJUST THESE IF BUTTONS ARE SWAPPED)
# ==============================================================================
# If 'C' Opens and 'G' Closes, SWAP these two numbers (0 and 1).
HW_CMD_CLOSE = 0  
HW_CMD_OPEN  = 1  

# ==============================================================================
# 2. JOINT SETUP
# ==============================================================================
ARM_JOINT_NAMES = [
    'joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3', 
    'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6'
]

# EXACT NAMES FROM YOUR URDF
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
    'f': ([0.0, -45.0, -45.0, 0.0, 0.0, 0.0],   "Arm-reaching: Forward"),           
    'b': ([0.0, 45.0, -135.0, 0.0, 0.0, 0.0],    "Arm-reaching: Backward"),         
    'u': ([0.0, 20.0, -50.0, 0.0, 0.0, 0.0],       "Arm-reaching: Up"),              
    'd': ([0.0, -20.0, -130.0, 0.0, 0.0, 0.0],     "Arm-reaching: Down"),             
    'l': ([90.0, -45.0, -45.0, 0.0, 0.0, 0.0],  "Arm-reaching: Left"),              
    'r': ([-90.0, -45.0, -45.0, 0.0, 0.0, 0.0], "Arm-reaching: Right"),
    'p': ([0.0, 0.0, -90.0, -45.0, 0.0, 0.0],     "Wrist-twisting: Pronation"),
    's': ([0.0, 0.0, -90.0, 45.0, 0.0, 0.0],      "Wrist-twisting: Supination"),
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

class Raw(object):
    def __init__(self, stream):
        self.stream = stream
    def __enter__(self):
        self.original_stty = termios.tcgetattr(self.stream)
        tty.setcbreak(self.stream)
    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.stream, termios.TCSANOW, self.original_stty)

class TeleopKeyboardNode(Node):
    def __init__(self):
        super().__init__('teleop_keyboard_node')
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
        
        # VISUAL STATE ONLY (Doesn't affect hardware command)
        # -0.5 = Visually Open, 0.15 = Visually Closed
        self.visual_gripper_val = -0.5 

        if force_sim:
            self.get_logger().info("FORCED SIMULATION MODE.")
        else:
            try:
                self.mc = MyCobot280RDKX5(port, baud)
                lock = acquire(LOCK_FILE)
                if lock:
                    self.get_logger().info("Initializing Gripper...")
                    self.mc.init_gripper()       
                    time.sleep(0.1)
                    self.mc.set_gripper_mode(0) # Transparent Mode
                    time.sleep(0.1)
                    
                    actual_angles = self.mc.get_angles()
                    if actual_angles:
                        self.current_joints = actual_angles
                        self.connected = True
                        self.get_logger().info("Hardware Connected & Gripper Initialized.")
                    release(lock)
            except Exception as e:
                self.get_logger().error(f"Connection failed: {e}")
        
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.publish_joint_state()

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
        
        # MIMIC LOGIC FOR RVIZ
        g = self.visual_gripper_val
        gripper_pos = [g, g, -g, -g, -g, g]

        msg.position = arm_rads + gripper_pos
        self.joint_pub.publish(msg)

    def move_gripper(self, target_state):
        """
        target_state: "close" or "open"
        """
        # 1. DETERMINE VALUES BASED ON CONFIG AT TOP
        if target_state == "close":
            visual_target = 0.15      # URDF Closed
            hw_command = HW_CMD_CLOSE # User Defined
            action_str = "CLOSING"
        else:
            visual_target = -0.5      # URDF Open
            hw_command = HW_CMD_OPEN  # User Defined
            action_str = "OPENING"

        # 2. UPDATE VISUALS
        self.visual_gripper_val = visual_target
        self.publish_joint_state() 
        print(f"Action: {action_str} (Sent: {hw_command})")

        # 3. SEND TO HARDWARE
        if self.connected:
            lock = acquire(LOCK_FILE)
            if lock:
                self.mc.set_gripper_state(hw_command, 50)
                release(lock)
            else:
                print(" -> LOCK FAILED")
        else:
            print(" -> Simulation Only")

    def interpolate_joints(self, target_angles, duration=2.0):
        start_angles = list(self.current_joints)
        steps = 20 
        dt = duration / steps
        for i in range(steps):
            alpha = (i + 1) / steps
            self.current_joints = [s + (e - s) * alpha for s, e in zip(start_angles, target_angles)]
            self.sync_robot()
            time.sleep(dt)

    def keyboard_listener(self):
        print("Control: 0=Home, Space=Rest, f/b/u/d/l/r=Move, c=Close, g=Open, q=Quit")
        while rclpy.ok():
            with Raw(sys.stdin):
                key = sys.stdin.read(1)
            
            if key == 'q': break
            
            if key == '0': self.interpolate_joints([0.0]*6)
            elif key == ' ': self.interpolate_joints(REST_POSE)
            elif key in PRESET_ACTIONS:
                angles, name = PRESET_ACTIONS[key]
                self.get_logger().info(f"Action: {name}")
                self.interpolate_joints(angles)
                self.interpolate_joints(REST_POSE)
            
            # GRIPPER INPUTS
            elif key == 'c': 
                self.move_gripper("close")
            elif key in ['b', 'g']: 
                self.move_gripper("open")

def main():
    if os.path.exists(LOCK_FILE):
        try: os.remove(LOCK_FILE)
        except: pass

    rclpy.init()
    node = TeleopKeyboardNode()
    try:
        node.keyboard_listener()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()