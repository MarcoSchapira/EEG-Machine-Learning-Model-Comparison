import os
import glob
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import decimate

DOWNSAMPLE_FACTOR = 10
INITIAL_SAMPLING_RATE = 2500  # before downsampling
TRIAL_SAMPLES = 10000         # 4 seconds at 2500 Hz
DATA_PATH = 'D:/EEG Data/EEG_11/EEG_Mat'
OUTPUT_DIR = 'D:/EEG Data/EEG_11/EEG_Compact'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def downsample_array(arr, factor):
    return decimate(arr, factor, axis=0, zero_phase=True)

files = glob.glob(os.path.join(DATA_PATH, "EEG_session*_sub*_*_realMove.mat"))
if not files: print("no files")
files.sort()

for f in files:
    print(f"\nProcessing {f}...")
    data = loadmat(f, struct_as_record=False, squeeze_me=True)

    # ===== Load EEG Channels =====
    channels = []
    for i in range(1, 61):
        ch = data.get(f"ch{i}")
        if ch is None:
            raise ValueError(f"Missing channel ch{i} in {f}")
        channels.append(ch.reshape(-1, 1))

    eeg = np.hstack(channels)  # (samples, 60)
    total_samples = eeg.shape[0]

    # ===== Load Markers =====
    mrk = data["mrk"]
    pos = np.array(mrk.pos)  # movement onset markers
    toe = np.array(mrk.toe)

    misc_pos = np.array(mrk.misc.pos)  # full recording bounds
    start_idx = int(misc_pos[0])
    end_idx = int(misc_pos[-1])

    print(f"Total trials found: {len(pos)}")

    # ===== Filter valid movement markers =====
    valid = np.where((pos >= start_idx) & (pos <= end_idx))[0]
    pos = pos[valid]
    toe = toe[valid]

    print(f"Valid trials found: {len(pos)}")

    trial_data = []
    trial_labels = []

    for i in range(len(pos)):
        start = int(pos[i]) 
        end = start + TRIAL_SAMPLES
      
        if end > end_idx:
            print(f"  Skipping trial {i} (index {start}): not enough data.")
            continue  # Skip this trial
        
        
        segment = eeg[start:end, :]
        
        if len(segment) < 10000:
            print(f"  Skipping trial {i} (index {start}): segment was too short.")
            continue # Skip this trial

        # Downsample
        segment_ds = downsample_array(segment, DOWNSAMPLE_FACTOR)

        trial_data.append(segment_ds)
        trial_labels.append(toe[i])

    # Stack the list of 2D arrays (samples, channels) into a 3D array
    # Shape becomes (num_trials, samples, channels) -> (N, 1000, 60)
    trial_data_3d = np.stack(trial_data, axis=0)
    
    # Transpose to (num_trials, channels, samples) -> (N, 60, 1000)
    # This is a common format for deep learning (PyTorch/TensorFlow)
    trial_data_3d = np.transpose(trial_data_3d, (0, 2, 1))

    # Convert labels to a 2D column vector
    trial_labels = np.array(trial_labels).reshape(-1, 1)

    # ===== Save new compact file =====
    fname = os.path.basename(f)
    fname_compact = fname.replace(".mat", "_compact.mat")
    output_file = os.path.join(OUTPUT_DIR, fname_compact)

    savemat(output_file, {
        "trial_data": trial_data_3d,
        "labels": trial_labels,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "initial_sample_rate": INITIAL_SAMPLING_RATE
    })

    print(f"✅ Saved: {fname_compact}")
