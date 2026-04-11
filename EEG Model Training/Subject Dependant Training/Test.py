import os
import random
import tkinter as tk
from tkinter import font
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import torch.nn.functional as F
import numpy as np

# --- Import your custom models ---
from MSCFormerModel import MSCFormer, Parameters
from EEGEncoderModel import EEGEncoder
from TCNet_Model import TCNetModel 

def prepare_sample(eeg_sample, model_name):
    """
    Prepares a raw numpy array for inference.
    """
    # 1. Z-Score Normalization
    sample_mean, sample_std = np.mean(eeg_sample), np.std(eeg_sample)
    eeg_sample = (eeg_sample - sample_mean) / sample_std

    # 2. Convert to Torch Tensor
    tensor_sample = torch.from_numpy(eeg_sample).float().cuda()

    # 3. Reshape based on model requirements
    if model_name in ['MSCFormer', 'EEGEncoder']:
        tensor_sample = tensor_sample.unsqueeze(0).unsqueeze(0) 
    elif model_name == 'TCNet':
        tensor_sample = tensor_sample.unsqueeze(0)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return tensor_sample


class EEGInferenceApp:
    def __init__(self, root, model_path, model_name, data_path, video_dir, channels_to_keep=None):
        self.root = root
        self.root.title("EEG Motor Imagery/Execution Classifier")
        self.root.geometry("1100x700")
        
        self.model_name = model_name
        self.video_dir = video_dir
        
        self.classes = [
            "Arm Forward", "Arm Backward", "Arm Left", "Arm Right", 
            "Arm Up", "Arm Down", "Hand Grasping", "Wrist Rotation", "Rest"
        ]
        
        # Hardware setup
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        
        # Load Model once at startup to prevent GUI freezing
        print(f"Loading {model_name} from {model_path}...")
        self.model = torch.load(model_path).cuda()
        self.model.eval()
        
        # Load Test Data
        print(f"Loading test data from {data_path}...")
        # Assuming the .pt file is a dictionary with 'data' and 'label' keys
        test_data = torch.load(data_path)
        self.eeg_data = test_data['x_test']   # Expected shape: (Samples, Channels, Time)
        self.eeg_labels = test_data['y_test'] # Expected shape: (Samples,)
        #self.eeg_data = test_data['data']  
        #self.eeg_labels = test_data['label']

        if channels_to_keep is not None:
            # Slices the array/tensor to keep all samples (:), only specified channels, and all time steps (:)
            self.eeg_data = self.eeg_data[:, channels_to_keep, :]
            print(f"Filtered channels. New data shape: {list(self.eeg_data.shape)}")
        
        # Video playback variables
        self.video_cap = None
        self.video_after_id = None
        
        self.setup_gui()

    def setup_gui(self):
        # --- TOP FRAME: 9 Buttons ---
        btn_frame = tk.Frame(self.root, pady=10)
        btn_frame.pack(side=tk.TOP, fill=tk.X)
        
        for i, class_name in enumerate(self.classes):
            btn = tk.Button(
                btn_frame, text=class_name, width=12, height=2,
                command=lambda idx=i: self.on_action_click(idx)
            )
            btn.pack(side=tk.LEFT, padx=5, expand=True)

        # --- MIDDLE FRAME: Video (Left) and Plot (Right) ---
        mid_frame = tk.Frame(self.root, pady=20)
        mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Left: Video Display
        video_frame = tk.Frame(mid_frame, width=500, height=400, bg="black")
        video_frame.pack(side=tk.LEFT, padx=20, expand=True)
        video_frame.pack_propagate(False) # Keep size strict
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Right: Matplotlib Plot
        plot_frame = tk.Frame(mid_frame, width=500, height=400)
        plot_frame.pack(side=tk.RIGHT, padx=20, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Waiting for selection...")
        self.ax.set_xlabel("Time Steps")
        self.ax.set_ylabel("Amplitude")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        # --- BOTTOM FRAME: Results ---
        bot_frame = tk.Frame(self.root, pady=20)
        bot_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        custom_font = font.Font(size=18, weight="bold")
        self.result_label = tk.Label(bot_frame, text="Select an action to begin inference.", font=custom_font)
        self.result_label.pack()

    def play_video(self):
        """Reads frames from OpenCV and displays them in Tkinter, looping seamlessly."""
        if self.video_cap is not None and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                # Loop the video back to the start
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()
            
            if ret:
                # Resize to fit the UI square
                frame = cv2.resize(frame, (500, 400))
                # Convert BGR (OpenCV) to RGB (Tkinter/PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Schedule next frame (~30 fps)
                self.video_after_id = self.root.after(33, self.play_video)

    def run_inference(self, sample):
        """Runs the loaded model on the specific sample."""
        input_tensor = prepare_sample(sample, self.model_name)

        with torch.no_grad():
            _, logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze()
            winning_conf, winning_class = torch.max(probabilities, dim=0)

        win_class_idx = int(winning_class.cpu().numpy())
        win_conf_pct = float(winning_conf.cpu().numpy()) * 100
        return win_class_idx, win_conf_pct

    def on_action_click(self, class_idx):
        target_class_name = self.classes[class_idx]
        
        # 1. Handle Video Playback
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
        if self.video_cap:
            self.video_cap.release()
            
        # Assumes videos are named exactly like the classes (e.g., "Arm Forward.mp4")
        video_path = os.path.join(self.video_dir, f"{target_class_name}.mov")
        if os.path.exists(video_path):
            self.video_cap = cv2.VideoCapture(video_path)
            self.play_video()
        else:
            self.video_label.config(image='', text="Video Not Found", fg="white")

        # 2. Get random data sample corresponding to the chosen class
        # Convert labels to numpy if they are tensors to make finding indices easy
        labels_np = self.eeg_labels.cpu().numpy() if torch.is_tensor(self.eeg_labels) else np.array(self.eeg_labels)
        
        valid_indices = np.where(labels_np == class_idx)[0]
        if len(valid_indices) == 0:
            self.result_label.config(text=f"No test data found for {target_class_name}", fg="red")
            return
            
        chosen_idx = random.choice(valid_indices)
        
        # Extract the sample (Shape: Channels x Time)
        sample = self.eeg_data[chosen_idx]
        if torch.is_tensor(sample):
            sample = sample.cpu().numpy()

        # 3. Plot the data
        self.ax.clear()
        # Plotting the mean across all channels to avoid a messy plot of 44 overlapping lines.
        # Alternatively, you could plot just sample[0, :] for the first channel.
        Cz_channel_data = sample[0, :] 
        self.ax.plot(Cz_channel_data, color='blue')
        self.ax.set_title(f"EEG Data Segment: {target_class_name}")
        self.ax.set_xlabel("Time Steps")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

        # 4. Run Inference
        pred_idx, conf = self.run_inference(sample)
        pred_name = self.classes[pred_idx]

        # 5. Update UI with Results
        color = "green" if pred_idx == class_idx else "red"
        result_text = f"True: {target_class_name} | Predicted: {pred_name} | Certainty: {conf:.2f}%"
        self.result_label.config(text=result_text, fg=color)

def inspect_pt_file(file_path):
    print(f"--- Inspecting: {file_path} ---")
    try:
        # Load the file
        data = torch.load(file_path)
        print(f"Top-level data type: {type(data)}\n")
        
        # If it's a dictionary (most common for datasets)
        if isinstance(data, dict):
            print("Dictionary Keys found:")
            for key, value in data.items():
                if torch.is_tensor(value):
                    print(f" -> '{key}': PyTorch Tensor | Shape: {list(value.shape)} | dtype: {value.dtype}")
                elif isinstance(value, np.ndarray):
                    print(f" -> '{key}': NumPy Array | Shape: {value.shape} | dtype: {value.dtype}")
                elif isinstance(value, list):
                    print(f" -> '{key}': Python List | Length: {len(value)}")
                else:
                    print(f" -> '{key}': {type(value)}")
                    
        # If it's just a raw tensor
        elif torch.is_tensor(data):
            print(f"File contains a single PyTorch Tensor.")
            print(f" -> Shape: {list(data.shape)} | dtype: {data.dtype}")
            
        # If it's something else
        else:
            print("Data is not a dictionary or a tensor. Here is the raw printout:")
            print(data)
            
    except Exception as e:
        print(f"Error loading file: {e}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    TARGET_SUBJECT = 9
    MODEL_ARCHITECTURE = 'TCNet'
    DATASET_TYPE = "C"
    
    # Paths (Modify these to point to your actual directories)
    MODEL_FILE = f"./{DATASET_TYPE}_{MODEL_ARCHITECTURE}/model_{TARGET_SUBJECT}_Production.pth"
    TEST_DATA_FILE = f"./{DATASET_TYPE}_{MODEL_ARCHITECTURE}/sub_9_test_split.pt" # Point this to your test dataset .pt file
    VIDEO_DIRECTORY = "./videos"      # Directory containing snippet videos of each action
    
    #inspect_pt_file(TEST_DATA_FILE)
    CHANNELS_TO_KEEP = list(range(27))
    # Initialize the Tkinter Application
    root = tk.Tk()
    app = EEGInferenceApp(
        root=root, 
        model_path=MODEL_FILE, 
        model_name=MODEL_ARCHITECTURE, 
        data_path=TEST_DATA_FILE,
        video_dir=VIDEO_DIRECTORY,
        channels_to_keep=CHANNELS_TO_KEEP
    )
    
    # Start the GUI event loop
    root.mainloop()
    