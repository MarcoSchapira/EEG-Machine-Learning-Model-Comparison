import os
import glob
import mne
import torch
import numpy as np
from scipy.signal import decimate, cheb2ord, cheby2, filtfilt

# ==========================================
# 1. Configuration
# ==========================================
RAW_DATA_DIR = r"D:\EEG Data\Our Recordings"  
OUTPUT_DIR = r"D:\EEG Data\Our Recordings\Ben_EEG_Processed"       
os.makedirs(OUTPUT_DIR, exist_ok=True)
INITIAL_SAMPLING_RATE = 1000
TARGET_SAMPLING_RATE = 250
DOWNSAMPLE_FACTOR = int(INITIAL_SAMPLING_RATE / TARGET_SAMPLING_RATE) # 4

# 4 seconds at 1000 Hz = 4000 samples
TRIAL_SAMPLES = int(4 * INITIAL_SAMPLING_RATE) 

def get_action_label(filename):
    fname = filename.lower()
    if 'armreachingforward' in fname: return 0
    if 'armreachingbackwards' in fname: return 1
    if 'armreachingleft' in fname: return 2
    if 'armreachingright' in fname: return 3
    if 'armreachingup' in fname: return 4
    if 'armreachingdown' in fname: return 5
    if 'graspingcup' in fname: return 6
    if 'wristproination' in fname or 'wristpronation' in fname: return 7
    if 'wristsupination' in fname: return 7
    raise ValueError(f"Could not determine action label from filename: {filename}")

def apply_chebyshev_filter(eeg_data, fs):
    nyq = fs / 2.0
    wp = 42.0 / nyq
    ws = 49.0 / nyq
    N, Wn = cheb2ord(wp, ws, gpass=3, gstop=40)
    b, a = cheby2(N, rs=50, Wn=Wn, btype='lowpass')
    return filtfilt(b, a, eeg_data, axis=0)

# ==========================================
# 2. Processing Loop
# ==========================================
vhdr_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.vhdr")))
if not vhdr_files:
    print("No .vhdr files found in the specified directory.")

all_trial_data = []
all_trial_labels = []

for f in vhdr_files:
    print(f"\nProcessing {os.path.basename(f)}...")
    action_label = get_action_label(os.path.basename(f))
    
    raw = mne.io.read_raw_brainvision(f, preload=True, verbose=False)
    eeg_data = raw.get_data().T 
    total_samples = eeg_data.shape[0]
    
    # Apply the Chebyshev filter
    eeg_data = apply_chebyshev_filter(eeg_data, INITIAL_SAMPLING_RATE)
    
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    # 1. Find the exact integer ID MNE assigned to your S6 marker
    s6_id = None
    for marker_name, marker_id in event_dict.items():
        if 'S  6' in marker_name or 's6' in marker_name.lower():
            s6_id = marker_id
            break
            
    if s6_id is None:
        print(f"  Warning: Could not find an 'S6' marker in this file! Skipping.")
        continue
        
    # 2. Extract ONLY the S6 markers
    s6_events = events[events[:, 2] == s6_id]
    s6_positions = s6_events[:, 0]
    
    valid_trials_extracted = 0

    # 3. Use gap length to determine class and slice offset
    for i in range(len(s6_positions)):
        trigger_idx = int(s6_positions[i])
        
        # Calculate gap to the next trigger
        if i + 1 < len(s6_positions):
            gap = s6_positions[i+1] - trigger_idx
        else:
            # For the very last trigger in the file, gap is the remaining data
            gap = total_samples - trigger_idx
            
        # --- LOGIC EVALUATION ---
        
        # Condition A: Action (Gap between 2000 and 4500)
        if 2000 <= gap <= 4500:
            current_label = action_label
            slice_start = trigger_idx                # Start exactly at trigger
            
        # Condition B: Rest (Gap > 5500)
        elif gap > 5500:
            current_label = 8
            slice_start = trigger_idx + 1000         # Start 1000 samples after trigger
            
        # Condition C: False presses or weird anomalies (Gap < 2000 or 4501-5500)
        else:
            print(f"  Skipping trigger at {trigger_idx} (Gap: {gap} fits neither Action nor Rest).")
            continue
            
        # --- DATA SLICING ---
        slice_end = slice_start + TRIAL_SAMPLES
        
        # Ensure we don't try to grab data past the end of the recording
        if slice_end > total_samples:
            print(f"  Skipping trigger at {trigger_idx}: Not enough data left in recording.")
            continue
            
        segment = eeg_data[slice_start:slice_end, :]
        
        # Double check we got exactly 4000 samples
        if len(segment) < TRIAL_SAMPLES:
            continue 

        # Downsample the 4000 samples to 1000 samples
        segment_ds = decimate(segment, DOWNSAMPLE_FACTOR, axis=0, zero_phase=True)
            
        all_trial_data.append(segment_ds)
        all_trial_labels.append(current_label)
        valid_trials_extracted += 1

    print(f"  Successfully extracted {valid_trials_extracted} trials (Actions & Rests).")

# ==========================================
# 3. Format and Save to PyTorch (.pt)
# ==========================================
print("\nFormatting data for PyTorch...")

trial_data_3d = np.stack(all_trial_data, axis=0)
trial_data_3d = np.transpose(trial_data_3d, (0, 2, 1))
labels_np = np.array(all_trial_labels)

data_tensor = torch.tensor(trial_data_3d, dtype=torch.float32)
label_tensor = torch.tensor(labels_np, dtype=torch.int64)

dataset_dict = {
    "data": data_tensor,
    "label": label_tensor
}

output_path = os.path.join(OUTPUT_DIR, "EEG_Ben.pt")
torch.save(dataset_dict, output_path)

print(f"✅ Pipeline complete! Saved combined dataset to {output_path}")
print(f"Final Data Shape: {data_tensor.shape} | Labels Shape: {label_tensor.shape}")