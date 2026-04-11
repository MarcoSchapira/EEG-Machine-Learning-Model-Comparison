import numpy as np
import os
import torch
import scipy.io as sio
from tqdm import tqdm


# --- Configuration ---
MAT_DATA_DIR = "D:/EEG Data/EEG_11/EEG_Compact/"
PREPROCESSED_DIR = os.path.join(MAT_DATA_DIR, "Processed_PerSubject_PT")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

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
    
    global_trial_count = 0
    all_labels = []
    metadata = {} # To store n_samples, n_channels, etc.

    #! Loop through all subjects
    for sub in tqdm(ALL_SUBJECTS, desc="Processing Subjects"):
        subject_data = []
        subject_labels = []
        
        #! Loop through all sessions
        for sess in ALL_SESSIONS:
            
            #! Loop through all actions (3 actions: multigrasp, reaching, twist)
            for action in ALL_ACTIONS:
                
                #! Load the data
                filename = f"EEG_session{sess}_sub{sub}_{action}_realMove_compact.mat"
                file_path = os.path.join(MAT_DATA_DIR, filename)
                
                if os.path.exists(file_path):
                    try:
                        mat = sio.loadmat(file_path)

                        #! Get the trial data and labels
                        trial_data = mat['trial_data'] # Shape (trials, channels, samples)
                        labels_raw = mat['labels'].squeeze() # Shape (trials,)
                        #print(f"Trial data shape: {trial_data.shape}")
                        #print(f"Labels shape: {labels_raw.shape}")
                        
                        #! --- Get metadata from first valid file ---
                        if 'n_channels' not in metadata:
                            metadata['n_channels'] = trial_data.shape[1]
                            metadata['n_samples'] = trial_data.shape[2]
                            print(f"\nDetected {metadata['n_channels']} channels.")
                            print(f"Detected {metadata['n_samples']} samples per trial.\n")
                        else:
                            if metadata['n_channels'] != trial_data.shape[1]:
                                print(f"Warning: n_channels mismatch in {file_path}: {metadata['n_channels']} != {trial_data.shape[1]}")
                            if metadata['n_samples'] != trial_data.shape[2]:
                                print(f"Warning: n_samples mismatch in {file_path}: {metadata['n_samples']} != {trial_data.shape[2]}")

                        #! --- Apply custom remapping ---
                        mapping = LABEL_REMAPPING.get(action, {})
                        remapped_labels = np.copy(labels_raw)
                        if mapping:
                            for old_label, new_label in mapping.items():
                                remapped_labels[labels_raw == old_label] = new_label

                        #! --- Gather trials for the subject ---
                        subject_data.append(trial_data)
                        all_labels.extend(remapped_labels) 
                        subject_labels.append(remapped_labels)
                            
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    print(f"File {file_path} does not exist")

        #! --- Combine, Cast, and Save for the Subject ---
        if subject_data: # Make sure we actually found data for this subject
            # Concatenate all gathered trials into single arrays
            sub_data_np = np.concatenate(subject_data, axis=0)
            sub_labels_np = np.concatenate(subject_labels, axis=0)
            
            # Update global trial count for your metadata file
            global_trial_count += sub_data_np.shape[0]
            
            # Cast to PyTorch tensors with correct dtypes
            tensor_data = torch.tensor(sub_data_np, dtype=torch.float32)
            tensor_labels = torch.tensor(sub_labels_np, dtype=torch.long)
            
            # Save as a single .pt file per subject
            output_path = os.path.join(PREPROCESSED_DIR, f"sub_{sub:02d}.pt")
            torch.save({'data': tensor_data, 'label': tensor_labels}, output_path)
    
    
    #! --- Save metadata file ---
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