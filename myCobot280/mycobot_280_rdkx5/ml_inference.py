import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
import torch
import numpy as np
import random

import sys
from . import TCNet_Model
sys.modules['TCNet_Model'] = TCNet_Model

# EMG Models
from .model import get_model
from .dataset import EMGDataset

import warnings

# Suppress PyTorch asymmetric padding warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths")

class MLInferenceNode(Node):
    def __init__(self):
        super().__init__('ml_inference_node')
        
        # --- EMG Parameters ---
        self.declare_parameter('data_root', '/home/lbran/1subject_TEST')
        self.declare_parameter('model_path', '/home/lbran/emg_model_6ch_tcn_lstm_multi.pth')
        
        # --- EEG Parameters ---
        self.declare_parameter('eeg_data_path', '/home/lbran/EEGcode/EEG_Presentation/C_TCNet/sub_9_test_split.pt')
        self.declare_parameter('eeg_model_path', '/home/lbran/EEGcode/EEG_Presentation/C_TCNet/model_9_Production.pth')
        
        self.mode = "HARDCODED"
        
        self.robot_cmd_pub = self.create_publisher(String, '/robot/command', 10)
        self.acc_pub = self.create_publisher(String, '/gui/accuracy', 10)
        
        self.emg_pubs = [self.create_publisher(Float64MultiArray, f'/sensor_data/emg/ch{i}', 10) for i in range(6)]
        # Create 27 separate publishers for the 27 EEG channels
        self.eeg_pubs = [self.create_publisher(Float64MultiArray, f'/sensor_data/eeg/ch{i}', 10) for i in range(27)]
        
        self.gui_sub = self.create_subscription(String, '/gui/command', self.gui_callback, 10)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Initializing ML pipelines on {self.device}...")
        
        # --- Load EMG Component ---
        try:
            self.dataset = EMGDataset(
                self.get_parameter('data_root').value, 
                is_train=False, scaling='constant', feature_ext='none', three_classes=False 
            )
            self.model = get_model('tcn_lstm', num_channels=6, feature_ext='none', num_classes=12).to(self.device)
            self.model.load_state_dict(torch.load(self.get_parameter('model_path').value, map_location=self.device))
            self.model.eval()
            self.get_logger().info("Successfully loaded EMG model.")
        except Exception as e:
            self.get_logger().error(f"Failed to load EMG components: {e}")
            self.dataset = None

        # --- Load EEG Component ---
        try:
            # 1. Load EEG Data Dictionary (ADDED weights_only=False)
            eeg_data_raw = torch.load(self.get_parameter('eeg_data_path').value, weights_only=False)
            self.eeg_data = eeg_data_raw.get('x_test', eeg_data_raw.get('data'))
            self.eeg_labels = eeg_data_raw.get('y_test', eeg_data_raw.get('label'))
            self.eeg_data = self.eeg_data[:, :27, :]
            
            # 2. Map the ROS 2 submodule to the top-level name PyTorch expects
            import sys
            from . import TCNet_Model
            sys.modules['TCNet_Model'] = TCNet_Model
            
            # 3. Load EEG Model (ADDED weights_only=False)
            self.eeg_model = torch.load(self.get_parameter('eeg_model_path').value, map_location=self.device, weights_only=False)
            self.eeg_model.eval()
            self.get_logger().info("Successfully loaded EEG model.")
        except Exception as e:
            self.get_logger().error(f"Failed to load EEG components: {e}")
            self.eeg_data = None

        # --- EEG Class Mappings ---
        # Maps standard Foxglove GUI strings to the EEG dataset's specific class indices
        self.gui_to_eeg_idx = {
            "Arm-reaching: Forward": 0, "Arm-reaching: Backward": 1,
            "Arm-reaching: Left": 2, "Arm-reaching: Right": 3,
            "Arm-reaching: Up": 4, "Arm-reaching: Down": 5,
            "Hand-grasping: Ball": 6, "Wrist-twisting: Pronation": 7,
            "Rest": 8
        }
        
        # Maps EEG predicted indices back to standard ROS strings for the robot
        self.eeg_idx_to_gui = {v: k for k, v in self.gui_to_eeg_idx.items()}

    def publish_gui_text(self, text):
        msg = String()
        msg.data = text
        self.acc_pub.publish(msg)
        self.get_logger().info(text)

    def gui_callback(self, msg):
        cmd = msg.data
        if cmd.startswith("MODE_"):
            self.mode = cmd.split("_")[1]
            self.publish_gui_text(f"MODE: {self.mode}")
            return
            
        if cmd in ["START", "STOP", "RESET", "Rest"]:
            self.publish_robot_command(cmd)
            self.publish_gui_text(f"SYSTEM COMMAND: {cmd}")
            return

        if self.mode == "HARDCODED":
            self.publish_robot_command(cmd)
            self.publish_gui_text(f"HARDCODED: Executing {cmd}")
        elif self.mode == "EMG":
            self.process_emg_inference(cmd)
        elif self.mode == "EEG":
            self.process_eeg_inference(cmd)

    def process_emg_inference(self, target_class):
        # [Existing process_emg_inference code remains exactly the same as your prompt]
        if not self.dataset:
            self.publish_gui_text("ERROR: Dataset not loaded.")
            return
            
        class_idx = self.dataset.active_class_to_idx.get(target_class)
        if class_idx is None:
            self.publish_gui_text(f"ERROR: {target_class} not found.")
            return
            
        valid_trial_indices = [i for i, label in enumerate(self.dataset.labels_class) if label == class_idx]
        if not valid_trial_indices:
            self.publish_gui_text(f"ERROR: No data for {target_class}.")
            return
            
        trial_idx = random.choice(valid_trial_indices)
        window_idx = random.randint(0, self.dataset.num_windows_per_trial - 1)
        dataset_idx = trial_idx * self.dataset.num_windows_per_trial + window_idx
        
        x_tensor, true_cls, true_reg = self.dataset[dataset_idx]
        
        for i in range(6):
            plot_msg = Float64MultiArray()
            plot_msg.data = x_tensor[i, :].cpu().numpy().tolist()
            self.emg_pubs[i].publish(plot_msg)
        
        x_batch = x_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out_cls, _ = self.model(x_batch, apply_gating=False)
            
            rest_idx = self.dataset.active_class_to_idx.get("Rest")
            if rest_idx is not None:
                out_cls[0, rest_idx] = -float('inf')
                
            probs = torch.nn.functional.softmax(out_cls, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
        accuracy_val = confidence.item() * 100
        predicted_class_name = list(self.dataset.active_class_to_idx.keys())[pred_idx.item()]
        
        match_status = "MATCH" if target_class == predicted_class_name else "MISMATCH"
        self.publish_gui_text(f"{match_status} - Target: {target_class} | Predicted: {predicted_class_name} ({accuracy_val:.2f}%)")
        self.publish_robot_command(predicted_class_name)

    def process_eeg_inference(self, target_class):
        if self.eeg_data is None:
            self.publish_gui_text("ERROR: EEG Dataset not loaded.")
            return
            
        target_idx = self.gui_to_eeg_idx.get(target_class)
        if target_idx is None:
            self.publish_gui_text(f"ERROR: {target_class} not supported by EEG model.")
            return

        # Fetch random sample for this class
        labels_np = self.eeg_labels.cpu().numpy() if torch.is_tensor(self.eeg_labels) else np.array(self.eeg_labels)
        valid_indices = np.where(labels_np == target_idx)[0]
        
        if len(valid_indices) == 0:
            self.publish_gui_text(f"ERROR: No EEG trial data found for {target_class}.")
            return
            
        chosen_idx = random.choice(valid_indices)
        sample = self.eeg_data[chosen_idx]
        if torch.is_tensor(sample):
            sample = sample.cpu().numpy()

        # Publish all 27 channels to their respective topics for Foxglove
        for i in range(27):
            plot_msg = Float64MultiArray()
            plot_msg.data = sample[i, :].tolist()
            self.eeg_pubs[i].publish(plot_msg)

        # Pre-process for inference (Z-Score Normalization from Test.py)
        sample_mean, sample_std = np.mean(sample), np.std(sample)
        norm_sample = (sample - sample_mean) / sample_std
        
        # Format for TCNet: (Batch, Channels, Time) -> (1, 27, X)
        tensor_sample = torch.from_numpy(norm_sample).float().to(self.device)
        tensor_sample = tensor_sample.unsqueeze(0)

        # Run Inference
        with torch.no_grad():
            # TCNetModel returns tuple: out, out
            _, logits = self.eeg_model(tensor_sample)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
            confidence, pred_idx = torch.max(probs, dim=0)

        accuracy_val = confidence.item() * 100
        pred_idx_val = int(pred_idx.cpu().numpy())
        predicted_class_name = self.eeg_idx_to_gui.get(pred_idx_val, f"Unknown ({pred_idx_val})")

        match_status = "MATCH" if target_class == predicted_class_name else "MISMATCH"
        self.publish_gui_text(f"{match_status} - Target: {target_class} | Predicted: {predicted_class_name} ({accuracy_val:.2f}%)")
        self.publish_robot_command(predicted_class_name)

    def publish_robot_command(self, cmd):
        msg = String()
        msg.data = cmd
        self.robot_cmd_pub.publish(msg)

def main():
    rclpy.init()
    node = MLInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()