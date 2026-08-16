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
# CONFIGURATION
# ==============================================================================
JOINT_NAMES = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3', 'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']
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

# --- Lock Management ---
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
        self.declare_parameter('force_simulation', False) # New parameter
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        force_sim = self.get_parameter('force_simulation').value
        
        self.connected = False
        self.mc = None

        self.current_joints = list(REST_POSE) # Timer will broadcast this list
        self.speed = 50

        # Check if we should even try to connect
        if force_sim:
            self.get_logger().info("FORCED SIMULATION MODE via launch argument.")
        else:
            try:
                self.mc = MyCobot280RDKX5(port, baud)
                lock = acquire(LOCK_FILE)
                if lock:
                    # Sync initial position from real robot if possible
                    actual_angles = self.mc.get_angles()
                    if actual_angles:
                        self.current_joints = actual_angles
                        self.connected = True
                    release(lock)
            except Exception as e:
                self.get_logger().error(f"Hardware connection failed: {e}. Falling back to SIM.")
        
        # Setup ROS Publisher
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # The Timer: Publishes the current state every 100ms (10Hz)
        # This keeps RViz "alive" even when you aren't typing.
        self.timer = self.create_timer(0.1, self.timer_callback)

        # 1. Attempt Hardware Connection
        try:
            self.mc = MyCobot280RDKX5(self.port, self.baud)
            # Test connection with a light read
            lock = acquire(LOCK_FILE)
            if lock and self.mc.get_angles():
                self.connected = True
                self.get_logger().info(f"Connected to Robot on {self.port}")
                release(lock)
            else:
                self.get_logger().warn("Hardware found but lock failed or no data. Switching to SIMULATION.")
        except Exception as e:
            self.get_logger().error(f"Connection failed: {e}. Switching to SIMULATION.")

        # 2. Setup ROS Communication
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.current_joints = list(REST_POSE)
        self.speed = 50

    def sync_robot(self):
        """Helper to move real robot or update RViz"""
        # Always update RViz
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [math.radians(x) for x in self.current_joints]
        self.joint_pub.publish(msg)

        # Move Real Hardware if connected
        if self.connected:
            lock = acquire(LOCK_FILE)
            if lock:
                self.mc.send_angles(self.current_joints, self.speed)
                release(lock)
    
    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        # Convert degrees to radians for URDF/RViz compatibility
        msg.position = [math.radians(x) for x in self.current_joints]
        self.joint_pub.publish(msg)

    def sync_hardware(self):
        """Only sends commands to the physical arm if we are connected"""
        if self.connected:
            lock = acquire(LOCK_FILE)
            if lock:
                self.mc.send_angles(self.current_joints, self.speed)
                release(lock) 

    def move_gripper(self, state):
        """state: 0 (close), 1 (open)"""
        if self.connected:
            lock = acquire(LOCK_FILE)
            if lock:
                self.mc.set_gripper_state(state, 50)
                release(lock)
        else:
            status = "OPEN" if state == 1 else "CLOSED"
            self.get_logger().info(f"[SIM] Gripper: {status}")

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
        print("Control: 0=Home, Space=Rest, f/b/u/d/l/r=Move, c/b/g=Gripper, q=Quit")
        while rclpy.ok():
            with Raw(sys.stdin):
                key = sys.stdin.read(1)
            
            if key == 'q': break
            elif key == '0': self.interpolate_joints([0.0]*6)
            elif key == ' ': self.interpolate_joints(REST_POSE)
            elif key in PRESET_ACTIONS:
                angles, name = PRESET_ACTIONS[key]
                self.get_logger().info(f"Action: {name}")
                self.interpolate_joints(angles)
                self.interpolate_joints(REST_POSE) # Auto-return to rest
            elif key == 'c': self.move_gripper(0)
            elif key in ['b', 'g']: self.move_gripper(1)

def main():
    # Cleanup stale locks at start
    if os.path.exists(LOCK_FILE):
        try: os.remove(LOCK_FILE)
        except: pass

    rclpy.init()
    node = TeleopKeyboardNode()
    node.keyboard_listener()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


'''
import sys
import os
import fcntl
import termios
import tty
import time
import rclpy
import pymycobot
from packaging import version

# min low version require
MIN_REQUIRE_VERSION = '3.8.0'

current_verison = pymycobot.__version__
print('current pymycobot library version: {}'.format(current_verison))
if version.parse(current_verison) < version.parse(MIN_REQUIRE_VERSION):
    raise RuntimeError('The version of pymycobot library must be greater than {} or higher. The current version is {}. Please upgrade the library version.'.format(MIN_REQUIRE_VERSION, current_verison))
else:
    print('pymycobot library version meets the requirements!')
    from pymycobot import MyCobot280RDKX5


LOCK_FILE = "/tmp/mycobot_lock"


# Avoid serial port conflicts and need to be locked
def acquire(lock_file):
    open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
    try:
        fd = os.open(lock_file, open_mode)
    except OSError as e:
        print(f"Failed to open lock file {lock_file}: {e}")
        return None

    lock_file_fd = None
    timeout = 30.0
    start_time = current_time = time.time()
    
    while current_time < start_time + timeout:
        try:
            # Attempt to grab the lock
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            time.sleep(1)
        else:
            lock_file_fd = fd
            break
        current_time = time.time()

    if lock_file_fd is None:
        print(f"Failed to acquire lock after {timeout} seconds. Forced cleanup may be required.")
        os.close(fd)
        # Optional: Safety cleanup if we are absolutely sure no other process is running
        # os.remove(lock_file) 
    return lock_file_fd


def release(lock_file_fd):
    # Fixed logic to prevent TypeError if lock_file_fd is None
    if lock_file_fd is None:
        return
        
    try:
        fcntl.flock(lock_file_fd, fcntl.LOCK_UN)
        os.close(lock_file_fd)
    except Exception as e:
        print(f"Failed to release lock: {e}")

# =ard coded positions and movements
# ------------------------------------------------------------------------------
JOINT_NAMES = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3', 'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']

REST_POSE = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]    # Rest

PRESET_ACTIONS = {

    'f': ([0.0, -45.0, -45.0, 0.0, 0.0, 0.0],   "Arm-reaching: Forward"),
    'b': ([0.0, 45.0, -135.0, 0.0, 0.0, 0.0],     "Arm-reaching: Backward"),
    'u': ([0.0, 0.0, -50.0, 0.0, 0.0, 0.0],       "Arm-reaching: Up"),
    'd': ([0.0, 0.0, -130.0, 0.0, 0.0, 0.0],     "Arm-reaching: Down"),
    'l': ([90.0, -45.0, -45.0, 0.0, 0.0, 0.0],  "Arm-reaching: Left"),
    'r': ([-90.0, -45.0, -45.0, 0.0, 0.0, 0.0], "Arm-reaching: Right"),
    'p': ([0.0, 0.0, -90.0, -45.0, 0.0, 0.0],     "Wrist-twisting: Pronation"),
    's': ([0.0, 0.0, -90.0, 45.0, 0.0, 0.0],      "Wrist-twisting: Supination"),

}

msg = """\

myCobot280 Keyboard Controller for movements
---------------------------
Keys:

    0: Zeros/Home
    Space Bar: Rest

    f: Arm-reaching: Forward
    b: Arm-reaching: Backward
    u: Arm-reaching: Up
    d: Arm-reaching: Down
    l: Arm-reaching: Left
    r: Arm-reaching: Right

    c: Hand-grasping: Card
    b: Hand-grasping: Ball 
    g: Hand-grasping: Cup

    p: Wrist-twisting: Pronation
    s: Wrist-twisting: Supination

    q: Quit
---------------------------

"""


def vels(speed, turn):
    return "currently:\tspeed: %s\tchange percent: %s  " % (speed, turn)

class Raw(object):
    def __init__(self, stream):
        self.stream = stream
        self.fd = self.stream.fileno()

    def __enter__(self):
        self.original_stty = termios.tcgetattr(self.stream)
        tty.setcbreak(self.stream)

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.stream, termios.TCSANOW, self.original_stty)

class TeleopKeyboard:
    def __init__(self):
        self.node = rclpy.create_node('teleop_keyboard')
        self.node.declare_parameter('port', '/dev/ttyUSB0')
        self.node.declare_parameter('baud', 1000000)
        port = self.node.get_parameter('port').value
        baud = self.node.get_parameter('baud').value
        self.mc = MyCobot280RDKX5(port, int(baud))
        time.sleep(0.05)
        
        if self.mc:
            lock = acquire(LOCK_FILE)
            if lock:
                if self.mc.get_fresh_mode() == 0:
                    self.mc.set_fresh_mode(1)
                release(lock)
        time.sleep(0.05)

        self.model = 1
        self.speed = 50
        self.change_percent = 5
        self.change_angle = 180 * self.change_percent / 100
        self.change_len = 250 * self.change_percent / 100
        self.init_pose = [[0, 0, 0, 0, 0, 0], self.speed]
        self.home_pose = [[0, 8, -127, 40, 0, 0], self.speed]
        self.record_coords = self.get_initial_coords()

    def get_initial_coords(self):
        while True:
            if self.mc:
                lock = acquire(LOCK_FILE)
                if lock:
                    res = self.mc.get_coords()
                    release(lock)
                    if res:
                        break
                time.sleep(0.1)
        return [res, self.speed, self.model]

    def print_status(self):
        print("\r current coords: %s" % self.record_coords)

    def keyboard_listener(self):
        print(msg)
        print(vels(self.speed, self.change_percent))
        while True:
            try:
                with Raw(sys.stdin):
                    key = sys.stdin.read(1)
                if key == "q":
                    break
                
                # Dictionary mapping for keys to coordinate changes would be cleaner,
                # but keeping your structure for simplicity:
                lock = None
                if key in ["w", "W"]:
                    self.record_coords[0][0] += self.change_len
                    lock = acquire(LOCK_FILE)
                    if lock: self.mc.send_coords(*self.record_coords)
                elif key in ["s", "S"]:
                    self.record_coords[0][0] -= self.change_len
                    lock = acquire(LOCK_FILE)
                    if lock: self.mc.send_coords(*self.record_coords)
                elif key in ["a", "A"]:
                    self.record_coords[0][1] -= self.change_len
                    lock = acquire(LOCK_FILE)
                    if lock: self.mc.send_coords(*self.record_coords)
                elif key in ["d", "D"]:
                    self.record_coords[0][1] += self.change_len
                    lock = acquire(LOCK_FILE)
                    if lock: self.mc.send_coords(*self.record_coords)
                elif key in ["z", "Z"]:
                    self.record_coords[0][2] -= self.change_len
                    lock = acquire(LOCK_FILE)
                    if lock: self.mc.send_coords(*self.record_coords)
                elif key in ["x", "X"]:
                    self.record_coords[0][2] += self.change_len
                    lock = acquire(LOCK_FILE)
                    if lock: self.mc.send_coords(*self.record_coords)
                # ... [Rotation and pose keys follow same pattern] ...
                elif key == "1":
                    lock = acquire(LOCK_FILE)
                    if lock: self.mc.send_angles(*self.init_pose)
                    release(lock)
                    time.sleep(3)
                    self.record_coords = self.get_initial_coords()
                    continue # Skip general release below

                # General release for movement keys
                if lock:
                    release(lock)
                
                self.print_status()
                time.sleep(0.1)
            except Exception as e:
                print(e)
                continue

def main():
    rclpy.init()
    # Before starting, clear any stale locks from previous crashes
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except:
            pass
            
    teleop_keyboard = TeleopKeyboard()
    teleop_keyboard.keyboard_listener()

if __name__ == '__main__':
    main()'''