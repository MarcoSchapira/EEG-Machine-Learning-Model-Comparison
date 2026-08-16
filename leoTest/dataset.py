"""
dataset.py
Loads multi-channel EMG data from .mat files, generates V-shaped target trajectories, 
balances data by full trials, and provides PyTorch DataLoaders.
Utilizes on-the-fly window slicing, dynamic scaling (including constant scaling for TD), 
feature extraction, digital filtering, and includes plotting utilities.
Supports dynamic class reduction (dropping the 'Rest' state).
"""
import os
import glob
import argparse
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

TARGET_STATES = {
    "Rest": [0.0, 0.0, -90.0, 0.0, 0.0, 0.0, 0],
    "Arm-reaching: Forward": [0.0, -45.0, -45.0, 0.0, 0.0, 0.0, 0],
    "Arm-reaching: Backward": [0.0, 45.0, -135.0, 0.0, 0.0, 0.0, 0],
    "Arm-reaching: Up": [0.0, 20.0, -50.0, 0.0, 0.0, 0.0, 0],
    "Arm-reaching: Down": [0.0, -20.0, -130.0, 0.0, 0.0, 0.0, 0],
    "Arm-reaching: Left": [90.0, -45.0, -45.0, 0.0, 0.0, 0.0, 0],
    "Arm-reaching: Right": [-90.0, -45.0, -45.0, 0.0, 0.0, 0.0, 0],
    "Wrist-twisting: Pronation": [0.0, 0.0, -90.0, -45.0, 0.0, 0.0, 0],
    "Wrist-twisting: Supination": [0.0, 0.0, -90.0, 45.0, 0.0, 0.0, 0],
    "Hand-grasping: Card": [0.0, 0.0, -90.0, 0.0, 0.0, 0.0, 100],
    "Hand-grasping: Ball": [0.0, 0.0, -90.0, 0.0, 0.0, 0.0, 60],
    "Hand-grasping: Cup": [0.0, 0.0, -90.0, 0.0, 0.0, 0.0, 40]
}

def get_active_classes(exclude_rest=False, three_classes=False):
    """Dynamically filters the target states based on CLI flags."""
    if three_classes:
        allowed = ["Arm-reaching: Backward", "Arm-reaching: Left", "Hand-grasping: Ball"]
        return {k: v for k, v in TARGET_STATES.items() if k in allowed}
    if exclude_rest:
        return {k: v for k, v in TARGET_STATES.items() if k != "Rest"}
    return TARGET_STATES

def get_class_to_idx(exclude_rest=False, three_classes=False):
    """Rebuilds the index mapping so labels remain contiguous integers starting at 0."""
    active_states = get_active_classes(exclude_rest, three_classes)
    return {k: v for v, k in enumerate(active_states.keys())}

# Defaults to full 12 classes for backward compatibility
CLASS_TO_IDX = get_class_to_idx(False, False)

def _extract_struct_field(struct_obj, field_name):
    """Safely extracts a field from a MATLAB struct loaded via scipy.io."""
    if hasattr(struct_obj, field_name):
        return getattr(struct_obj, field_name)
    elif isinstance(struct_obj, np.ndarray) and struct_obj.dtype.names and field_name in struct_obj.dtype.names:
        return struct_obj[field_name].item()
    else:
        raise KeyError(f"Field '{field_name}' not found in structure.")

def apply_digital_filter(data, fs=2500, cutoff=20.0, order=4):
    """Applies zero-mean subtraction and a 4th-order Butterworth high-pass filter."""
    data = data - np.mean(data, axis=0)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

class EMGDataset(Dataset):
    def __init__(self, data_root, is_train=True, window_size=500, stride=125, fs=2500, 
                 electrodes=None, scaling='zscore', feature_ext='none', exclude_rest=False,
                 three_classes=False, apply_filter=True, highpass_cutoff=20.0):
        self.data_root = data_root
        self.is_train = is_train
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.electrodes = electrodes if electrodes is not None else [0, 1, 2, 3, 4, 5]
        self.scaling = scaling.lower()
        self.feature_ext = feature_ext.lower()
        self.apply_filter = apply_filter
        self.highpass_cutoff = highpass_cutoff
        
        self.exclude_rest = exclude_rest
        self.three_classes = three_classes
        self.active_target_states = get_active_classes(self.exclude_rest, self.three_classes)
        self.active_class_to_idx = get_class_to_idx(self.exclude_rest, self.three_classes)
        
        self.trials, self.labels_class = self._load_and_process_data()
        
        self.class_trajectories = {}
        for c_name, c_idx in self.active_class_to_idx.items():
            self.class_trajectories[c_idx] = self._generate_v_trajectory(self.active_target_states[c_name])
            
        self.trial_length = int(4.0 * self.fs)
        self.num_windows_per_trial = (self.trial_length - self.window_size) // self.stride + 1
        self.total_windows = len(self.trials) * self.num_windows_per_trial
        
        self.pca_models = None
        if self.feature_ext == 'pca':
            self._fit_pca()
            
        if self.three_classes:
            mode_str = "3-Class Custom Subset"
        elif self.exclude_rest:
            mode_str = "11-Class (No Rest)"
        else:
            mode_str = "12-Class (Full)"
            
        print(f"Dataset initialized: {len(self.trials)} trials mapped to {self.total_windows} on-the-fly windows. [{mode_str}]")
        print(f"Pipeline -> Scaling: {self.scaling.upper()} | Feature Ext: {self.feature_ext.upper()} | Filter: {self.apply_filter} ({self.highpass_cutoff}Hz)")

    def _extract_td_features(self, window):
        """
        Augmented Time-Domain features.
        Input window shape: (window_size, channels)
        Output features shape: (6, channels) -> RMS, MAV, WL, ZC, SSC, VAR
        """
        thresh = 0.01
        
        # 1. Root Mean Square (Intensity)
        rms = np.sqrt(np.mean(window**2, axis=0))
        # 2. Mean Absolute Value
        mav = np.mean(np.abs(window), axis=0)
        # 3. Waveform Length (Complexity)
        wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0) / self.window_size
        # 4. Zero Crossing (Frequency proxy)
        zc = np.sum((window[:-1, :] * window[1:, :] < 0) & (np.abs(np.diff(window, axis=0)) > thresh), axis=0) / self.window_size
        # 5. Slope Sign Change (NEW - Motor unit firing rate proxy)
        diff = np.diff(window, axis=0)
        ssc = np.sum((diff[:-1, :] * diff[1:, :] < 0) & (np.abs(diff[:-1, :] - diff[1:, :]) > thresh), axis=0) / self.window_size
        # 6. Variance (NEW - Contraction power)
        var = np.var(window, axis=0)
        
        return np.vstack((rms, mav, wl, zc, ssc, var))

    def _fit_pca(self):
        print("Fitting PCA on a random subset of 5000 windows...")
        num_channels = len(self.electrodes)
        self.pca_models = [PCA(n_components=25) for _ in range(num_channels)]
        subset_windows = []
        indices = np.random.choice(self.total_windows, min(5000, self.total_windows), replace=False)
        for idx in indices:
            trial_idx = idx // self.num_windows_per_trial
            window_idx = idx % self.num_windows_per_trial
            start = window_idx * self.stride
            window = self.trials[trial_idx][start:start + self.window_size, self.electrodes]
            subset_windows.append(window)
        subset_windows = np.array(subset_windows) 
        for ch in range(num_channels):
            ch_data = subset_windows[:, :, ch]
            self.pca_models[ch].fit(ch_data)

    def _generate_v_trajectory(self, target_state, duration_sec=4.0):
        num_samples = int(duration_sec * self.fs)
        half_samples = num_samples // 2
        rest = np.array(TARGET_STATES["Rest"])
        target = np.array(target_state)
        ext_phase = np.linspace(rest, target, half_samples)
        ret_phase = np.linspace(target, rest, num_samples - half_samples)
        trajectory = np.vstack((ext_phase, ret_phase))
        limits_min = np.array([-135, -135, -135, -135, -135, -135, 0])
        limits_max = np.array([90, 90, 90, 90, 90, 90, 100])
        return 2 * ((trajectory - limits_min) / (limits_max - limits_min)) - 1

    def _load_and_process_data(self):
        mat_files = glob.glob(os.path.join(self.data_root, "*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {self.data_root}.")

        all_raw_trials, all_trial_classes = [], []
        for file_path in mat_files:
            file_name = os.path.basename(file_path).lower()
            try:
                mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
                channel_arrays = []
                for i in range(1, 8):
                    ch_key = f'ch{i}'
                    if ch_key in mat_data: channel_arrays.append(mat_data[ch_key])
                if not channel_arrays: continue
                signals = np.column_stack(channel_arrays)
                if 'mrk' not in mat_data: continue
                trigger_times = _extract_struct_field(mat_data['mrk'], 'pos')
                trigger_codes = _extract_struct_field(mat_data['mrk'], 'toe')
                if np.isscalar(trigger_times):
                    trigger_times = np.array([trigger_times]); trigger_codes = np.array([trigger_codes])
            except Exception: continue

            mapping = {}
            if 'reaching' in file_name: mapping = {11: "Arm-reaching: Forward", 21: "Arm-reaching: Backward", 31: "Arm-reaching: Left", 41: "Arm-reaching: Right", 51: "Arm-reaching: Up", 61: "Arm-reaching: Down", 8: "Rest"}
            elif 'grasp' in file_name: mapping = {11: "Hand-grasping: Cup", 21: "Hand-grasping: Ball", 61: "Hand-grasping: Card", 8: "Rest"}
            elif 'twist' in file_name: mapping = {91: "Wrist-twisting: Pronation", 101: "Wrist-twisting: Supination", 8: "Rest"}
            else: continue

            trial_length = int(4.0 * self.fs) 
            for pos, code in zip(trigger_times, trigger_codes):
                code_val = int(str(code).replace('S', '').strip())
                if code_val in mapping:
                    class_name = mapping[code_val]
                    if class_name not in self.active_class_to_idx: continue
                    start_idx = int(pos); end_idx = start_idx + trial_length
                    if end_idx <= signals.shape[0]:
                        trial_data = signals[start_idx:end_idx, :].astype(np.float64)
                        if self.apply_filter:
                            trial_data = apply_digital_filter(trial_data, fs=self.fs, cutoff=self.highpass_cutoff)
                        all_raw_trials.append(trial_data); all_trial_classes.append(class_name)

        trials_by_class = {c_idx: [] for c_idx in self.active_class_to_idx.values()}
        for i, c_name in enumerate(all_trial_classes):
            if c_name in self.active_class_to_idx:
                trials_by_class[self.active_class_to_idx[c_name]].append(i)
        
        movement_counts = [len(trials) for trials in trials_by_class.values() if len(trials) > 0]
        target_trial_count = int(np.mean(movement_counts))
        balanced_trials, balanced_labels = [], []
        for c_idx, trials in trials_by_class.items():
            if not trials: continue
            selected = np.random.choice(trials, target_trial_count, replace=False) if len(trials) > target_trial_count else trials
            for trial_idx in selected:
                balanced_trials.append(all_raw_trials[trial_idx]); balanced_labels.append(c_idx)
        shuffle_indices = np.random.permutation(len(balanced_trials))
        return [balanced_trials[i] for i in shuffle_indices], [balanced_labels[i] for i in shuffle_indices]

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        trial_idx = idx // self.num_windows_per_trial
        window_idx = idx % self.num_windows_per_trial
        start = window_idx * self.stride; end = start + self.window_size
        trial = self.trials[trial_idx]; c_idx = self.labels_class[trial_idx]
        window = trial[start:end, self.electrodes].astype(np.float32)
        
        if self.scaling == 'minmax':
            w_min = np.min(window, axis=0); w_max = np.max(window, axis=0)
            window = (window - w_min) / (w_max - w_min + 1e-8)
        elif self.scaling == 'zscore':
            w_mean = np.mean(window, axis=0); w_std = np.std(window, axis=0) + 1e-8
            window = (window - w_mean) / w_std
        elif self.scaling == 'constant':
            window = window / 1000.0
            
        if self.feature_ext == 'td':
            features = self._extract_td_features(window)
        elif self.feature_ext == 'pca':
            features = np.zeros((25, len(self.electrodes)))
            for ch in range(len(self.electrodes)):
                ch_data = window[:, ch].reshape(1, -1)
                features[:, ch] = self.pca_models[ch].transform(ch_data).flatten()
        else:
            features = window.T 
            
        reg_label = self.class_trajectories[c_idx][end - 1]
        x = torch.tensor(features.T, dtype=torch.float32) if self.feature_ext in ['td', 'pca'] else torch.tensor(features, dtype=torch.float32)
        return x, torch.tensor(c_idx, dtype=torch.long), torch.tensor(reg_label, dtype=torch.float32)

def get_dataloaders(data_root, batch_size=16, num_workers=2, electrodes=None, scaling='zscore', feature_ext='none', exclude_rest=False, three_classes=False, apply_filter=True, highpass_cutoff=20.0):
    subdirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    train_dir = next((d for d in subdirs if 'train' in d.lower()), None)
    test_dir = next((d for d in subdirs if 'test' in d.lower()), None)
    if not train_dir or not test_dir:
        raise FileNotFoundError(f"Missing train/test subfolders in {data_root}.")
    train_ds = EMGDataset(train_dir, True, electrodes=electrodes, scaling=scaling, feature_ext=feature_ext, exclude_rest=exclude_rest, three_classes=three_classes, apply_filter=apply_filter, highpass_cutoff=highpass_cutoff)
    test_ds = EMGDataset(test_dir, False, electrodes=electrodes, scaling=scaling, feature_ext=feature_ext, exclude_rest=exclude_rest, three_classes=three_classes, apply_filter=apply_filter, highpass_cutoff=highpass_cutoff)
    return DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=True), DataLoader(test_ds, batch_size, False, num_workers=num_workers, pin_memory=True), len(train_ds.active_class_to_idx)

# --- PLOTTING UTILITIES ---

def _extract_representative_trial(data_root, electrodes, fs=2500, apply_filter=True, highpass_cutoff=20.0):
    mat_files = glob.glob(os.path.join(data_root, "**", "*.mat"), recursive=True)
    if not mat_files: raise FileNotFoundError(f"No .mat files found in {data_root}.")
    for file_path in mat_files:
        try:
            mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
            channel_arrays = []
            for i in range(1, 8):
                ch_key = f'ch{i}'
                if ch_key in mat_data: channel_arrays.append(mat_data[ch_key])
            if not channel_arrays: continue
            signals = np.column_stack(channel_arrays)
            if 'mrk' not in mat_data: continue
            trigger_times = _extract_struct_field(mat_data['mrk'], 'pos')
            if np.isscalar(trigger_times): trigger_times = np.array([trigger_times])
            start_idx = int(trigger_times[0]); end_idx = start_idx + int(4.0 * fs)
            if end_idx <= signals.shape[0]:
                trial_data = signals[start_idx:end_idx, electrodes].astype(np.float64)
                if apply_filter: trial_data = apply_digital_filter(trial_data, fs=fs, cutoff=highpass_cutoff)
                return trial_data
        except Exception: continue
    raise ValueError("Could not extract a valid 4-second trial.")

def plot_average_emg(data_root, electrodes, apply_filter=True, highpass_cutoff=20.0):
    num_channels = len(electrodes)
    trial_data = _extract_representative_trial(data_root, electrodes, apply_filter=apply_filter, highpass_cutoff=highpass_cutoff)
    time_axis = np.linspace(0, 4, trial_data.shape[0]); avg_signal = np.mean(np.abs(trial_data), axis=1)
    plt.figure(figsize=(10, 4)); plt.plot(time_axis, avg_signal, label="Average EMG Envelope", color="purple")
    plt.title(f"Average Pre-Processed EMG Envelope ({num_channels} Channels) over 4-Second Trial Window")
    plt.xlabel("Time (s)"); plt.ylabel("Absolute Amplitude (uV)"); plt.grid(True); plt.legend()
    plt.savefig(f"average_emg_{num_channels}_electrodes.png"); plt.close()

def plot_individual_emg(data_root, electrodes, apply_filter=True, highpass_cutoff=20.0):
    num_channels = len(electrodes)
    trial_data = _extract_representative_trial(data_root, electrodes, apply_filter=apply_filter, highpass_cutoff=highpass_cutoff)
    time_axis = np.linspace(0, 4, trial_data.shape[0])
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), sharex=True)
    if num_channels == 1: axes = [axes]
    for i in range(num_channels):
        axes[i].plot(time_axis, trial_data[:, i], label=f"Electrode Index {electrodes[i]}", color=f"C{i}")
        axes[i].set_ylabel("Amp (uV)"); axes[i].grid(True); axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)"); fig.suptitle("Individual Pre-Processed EMG Signals", fontsize=14)
    plt.tight_layout(); plt.savefig(f"individual_emg_{num_channels}_electrodes.png"); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Loader and Visualizer CLI")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data parent directory")
    parser.add_argument("--plot_avg", action="store_true", help="Plot and save average EMG signal")
    parser.add_argument("--plot_individual", action="store_true", help="Plot and save individual EMG signals")
    parser.add_argument("--electrodes", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], help="List of EMG electrode indices")
    parser.add_argument("--no_filter", action="store_true", help="Disable high-pass filter")
    parser.add_argument("--highpass_cutoff", type=float, default=20.0, help="Cutoff frequency")
    args = parser.parse_args(); apply_filter = not args.no_filter
    if args.plot_avg: plot_average_emg(args.data_root, args.electrodes, apply_filter, args.highpass_cutoff)
    if args.plot_individual: plot_individual_emg(args.data_root, args.electrodes, apply_filter, args.highpass_cutoff)