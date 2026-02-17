import numpy as np
import os
import torch
import scipy.io as sio
from tqdm import tqdm # A progress bar, install with: pip install tqdm



# --- Configuration ---
MAT_DATA_DIR = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files"  # <-- Your .mat file directory
PREPROCESSED_DIR = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_PT_files"      # <-- New folder to save .pt files
ALL_SUBJECTS = list(range(1, 26))
ALL_SESSIONS = [1, 2, 3]
ALL_ACTIONS = ["multigrasp", "reaching", "twist"]

# --- Your Custom Label Remapping Logic ---
LABEL_REMAPPING = {
    'reaching': {
        11: 0, 21: 1, 31: 2, 41: 3, 51: 4, 61: 5, 8: 11
    },
    'multigrasp': {
        11: 6, 21: 7, 61: 8, 8: 11
    },
    'twist': {
        91: 9, 101: 10, 8: 11
    }
}

def preprocess_data():
    print(f"Starting preprocessing...")
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    global_trial_count = 0
    all_labels = []
    metadata = {} # To store n_samples, n_channels, etc.

    for sub in tqdm(ALL_SUBJECTS, desc="Processing Subjects"):
        subject_output_dir = os.path.join(PREPROCESSED_DIR, f"sub_{sub}")
        os.makedirs(subject_output_dir, exist_ok=True)
        trial_count_for_subject = 0
        
        for sess in ALL_SESSIONS:
            for action in ALL_ACTIONS:
                filename = f"EEG_session{sess}_sub{sub}_{action}_realMove_compact.mat"
                file_path = os.path.join(MAT_DATA_DIR, filename)
                
                if os.path.exists(file_path):
                    try:
                        mat = sio.loadmat(file_path)
                        trial_data = mat['trial_data'] # Shape (trials, channels, samples)
                        labels_raw = mat['labels'].squeeze()
                        
                        # --- Get metadata from first valid file ---
                        if 'n_channels' not in metadata:
                            metadata['n_channels'] = trial_data.shape[1]
                            metadata['n_samples'] = trial_data.shape[2]
                            print(f"\nDetected {metadata['n_channels']} channels.")
                            print(f"Detected {metadata['n_samples']} samples per trial.\n")

                        # --- Apply custom remapping ---
                        mapping = LABEL_REMAPPING.get(action, {})
                        remapped_labels = np.copy(labels_raw)
                        if mapping:
                            for old_label, new_label in mapping.items():
                                remapped_labels[labels_raw == old_label] = new_label
                        
                        all_labels.extend(remapped_labels)

                        # --- Save each trial individually ---
                        for i in range(trial_data.shape[0]):
                            data_trial = torch.tensor(trial_data[i], dtype=torch.float32)
                            label_trial = torch.tensor(remapped_labels[i], dtype=torch.long)
                            
                            save_obj = {'data': data_trial, 'label': label_trial}
                            
                            output_filename = f"trial_{trial_count_for_subject:05d}.pt"
                            output_path = os.path.join(subject_output_dir, output_filename)
                            torch.save(save_obj, output_path)
                            
                            trial_count_for_subject += 1
                            global_trial_count += 1
                            
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    # --- Save metadata file ---
    unique_labels = np.unique(all_labels)
    metadata['n_classes'] = len(unique_labels)
    metadata['total_trials'] = global_trial_count
    
    torch.save(metadata, os.path.join(PREPROCESSED_DIR, 'metadata.pt'))
    
    print("\n--- Preprocessing Complete ---")
    print(f"Total Trials Saved: {metadata['total_trials']}")
    print(f"Total Classes: {metadata['n_classes']} (Labels: {unique_labels})")
    print(f"Data saved to: {PREPROCESSED_DIR}")

if __name__ == '__main__':
    preprocess_data()