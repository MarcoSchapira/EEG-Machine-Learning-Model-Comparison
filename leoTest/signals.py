"""
signals.py
Global variance analyzer for EMG datasets. Combines train and test data to visualize
crosstalk, feature separability, and temporal overlap. 
Includes Time-Frequency Spectrograms, DTW Similarity Matrices, and Time-Aware t-SNE.
Optimized for Google Colab memory with real-time terminal tracking.
Includes hardware noise filtering via zero-mean and Butterworth high-pass.
"""
import os
import glob
import argparse
import json
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    print("Error: 'fastdtw' is not installed. Please run '!pip install fastdtw' in Colab.")
    exit(1)

# Import existing class definitions
from dataset import TARGET_STATES, _extract_struct_field

def apply_digital_filter(data, fs=2500, cutoff=20.0, order=4):
    """Applies zero-mean subtraction and a 4th-order Butterworth high-pass filter."""
    data = data - np.mean(data, axis=0)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def load_entire_dataset(data_root, electrodes, fs=2500, trial_duration=4.0, apply_filter=True, highpass_cutoff=20.0):
    """Recursively loads .mat files, perfectly balances classes, and decimates to LSTM window sizes."""
    print(f"\n[1/7] Scanning {data_root} for all .mat files...")
    mat_files = glob.glob(os.path.join(data_root, "**", "*.mat"), recursive=True)
    
    total_files = len(mat_files)
    if total_files == 0:
        raise FileNotFoundError(f"No .mat files found in {data_root}.")

    print(f"  -> Found {total_files} files. Beginning extraction...")

    all_trials_by_class = {c_name: [] for c_name in TARGET_STATES.keys()}
    trial_length_raw = int(trial_duration * fs)
    decimation_factor = int(fs / 125) # Reduces 10,000 hardware samples to 500 LSTM steps
    
    for idx, file_path in enumerate(mat_files):
        if (idx + 1) % 10 == 0 or (idx + 1) == total_files:
            print(f"\r  -> Parsing file {idx + 1}/{total_files}...", end="", flush=True)

        file_name = os.path.basename(file_path).lower()
        try:
            mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
            channel_arrays = []
            for i in range(1, 8):
                ch_key = f'ch{i}'
                if ch_key in mat_data: channel_arrays.append(mat_data[ch_key])
                    
            if not channel_arrays: continue
            signals = np.column_stack(channel_arrays)[:, electrodes] 
            
            if 'mrk' not in mat_data: continue
            trigger_times = _extract_struct_field(mat_data['mrk'], 'pos')
            trigger_codes = _extract_struct_field(mat_data['mrk'], 'toe')
            
            if np.isscalar(trigger_times):
                trigger_times = np.array([trigger_times])
                trigger_codes = np.array([trigger_codes])
                
        except Exception:
            continue

        mapping = {}
        if 'reaching' in file_name:
            mapping = {11: "Arm-reaching: Forward", 21: "Arm-reaching: Backward", 
                       31: "Arm-reaching: Left", 41: "Arm-reaching: Right", 
                       51: "Arm-reaching: Up", 61: "Arm-reaching: Down", 8: "Rest"}
        elif 'grasp' in file_name:
            mapping = {11: "Hand-grasping: Cup", 21: "Hand-grasping: Ball", 
                       61: "Hand-grasping: Card", 8: "Rest"}
        elif 'twist' in file_name:
            mapping = {91: "Wrist-twisting: Pronation", 101: "Wrist-twisting: Supination", 8: "Rest"}

        for pos, code in zip(trigger_times, trigger_codes):
            code_val = int(str(code).replace('S', '').strip())
            if code_val in mapping:
                class_name = mapping[code_val]
                start_idx = int(pos)
                end_idx = start_idx + trial_length_raw
                if end_idx <= signals.shape[0]:
                    trial_data = signals[start_idx:end_idx, :].astype(np.float64)
                    
                    if apply_filter:
                        trial_data = apply_digital_filter(trial_data, fs=fs, cutoff=highpass_cutoff)
                        
                    trial_data = trial_data / 1000.0 
                    # Decimate down to the chronological timeline the LSTM actually evaluates
                    trial_data = trial_data[::decimation_factor, :].astype(np.float32)
                    all_trials_by_class[class_name].append(trial_data)
                    
    print(f"\n  -> Data extraction complete. Filter Applied: {apply_filter} ({highpass_cutoff}Hz). Balancing classes...")
    
    active_counts = [len(trials) for c_name, trials in all_trials_by_class.items() if c_name != "Rest" and len(trials) > 0]
    
    if active_counts:
        target_trial_count = int(np.mean(active_counts))
        print(f"  -> Target trial count per class set to: {target_trial_count}")
        
        balanced_trials_by_class = {}
        for c_name, trials in all_trials_by_class.items():
            if len(trials) == 0:
                balanced_trials_by_class[c_name] = []
                continue
                
            if len(trials) > target_trial_count:
                indices = np.random.choice(len(trials), target_trial_count, replace=False)
                balanced_trials_by_class[c_name] = [trials[i] for i in indices]
            else:
                balanced_trials_by_class[c_name] = trials
                
        all_trials_by_class = balanced_trials_by_class

    print("\n  -> Final balanced 500-sample trials by class:")
    for k, v in all_trials_by_class.items():
        print(f"     [{k}]: {len(v)} trials")
        
    return all_trials_by_class

def plot_spectrograms(trials_by_class, num_channels, fs=125):
    """Generates Time-Frequency heatmaps for each electrode across all classes."""
    print("\n[2/7] Generating Time-Frequency Spectrograms per Electrode...")
    
    class_names = [k for k, v in trials_by_class.items() if len(v) > 0]
    
    for ch in range(num_channels):
        print(f"  -> Processing Spectrograms for Channel {ch}...")
        fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for idx, c_name in enumerate(class_names):
            ax = axes[idx]
            trials = trials_by_class[c_name]
            stacked_trials = np.stack(trials, axis=0)
            mean_sig = np.mean(stacked_trials[:, :, ch], axis=0)
            
            f, t, Sxx = signal.spectrogram(mean_sig, fs=fs, nperseg=64, noverlap=32)
            
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-8), shading='gouraud', cmap='inferno')
            ax.set_title(c_name, fontsize=10)
            if idx >= 9: ax.set_xlabel("Time (s)")
            if idx % 3 == 0: ax.set_ylabel("Frequency (Hz)")
            
        fig.suptitle(f"Temporal Frequency Evolution - Electrode {ch}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"spectrogram_elec_{ch}.png", dpi=200, bbox_inches='tight')
        plt.close()
    print("  -> Saved all spectrogram plots.")

def plot_dtw_heatmaps(trials_by_class, num_channels, results_dict):
    """Generates an matrix using true Pairwise DTW to prevent destructive interference."""
    print("\n[3/7] Generating DTW Temporal Similarity Matrices per Electrode...")
    
    class_names = [k for k, v in trials_by_class.items() if len(v) > 0]
    num_classes = len(class_names)
    results_dict['dtw_similarity'] = {}
    
    # Subsample size for pairwise comparison. 
    # 20 trials per class means 400 DTW calculations per grid square. Fast and statistically robust.
    n_samples = 20 
    
    for ch in range(num_channels):
        print(f"  -> Calculating Pairwise DTW matrix for Channel {ch} (Accuracy Mode)...")
        dtw_matrix = np.zeros((num_classes, num_classes))
        
        # Extract and format the random subset of physical trials
        subset_trials = {}
        for c_name in class_names:
            trials = trials_by_class[c_name]
            k = min(n_samples, len(trials))
            indices = np.random.choice(len(trials), k, replace=False)
            
            # Subsample sequence to 100 steps and reshape to (N, 1) for SciPy strictness
            subset_trials[c_name] = [trials[i][::5, ch].reshape(-1, 1) for i in indices]
            
        for i in range(num_classes):
            for j in range(i, num_classes):
                class_a_trials = subset_trials[class_names[i]]
                class_b_trials = subset_trials[class_names[j]]
                
                distances = []
                # Calculate the exact chronological distance between every single pair of real trials
                for sig_a in class_a_trials:
                    for sig_b in class_b_trials:
                        dist, _ = fastdtw(sig_a, sig_b, dist=euclidean)
                        distances.append(dist)
                
                # The true distance is the average of the pairwise alignments
                avg_distance = np.mean(distances)
                dtw_matrix[i, j] = avg_distance
                dtw_matrix[j, i] = avg_distance
                
        # Convert raw distance to a normalized similarity percentage
        max_dist = np.max(dtw_matrix)
        if max_dist == 0:
            similarity_matrix = np.full((num_classes, num_classes), 100.0)
        else:
            similarity_matrix = 100 * (1 - (dtw_matrix / max_dist))
        
        results_dict['dtw_similarity'][f'Channel_{ch}'] = similarity_matrix.tolist()
        
        plt.figure(figsize=(10, 8))
        # annot=False keeps the grid clean, visually exposing the crossover
        sns.heatmap(similarity_matrix, annot=False, cmap='magma', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"DTW Pairwise Shape Similarity (%) - Electrode {ch}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"dtw_heatmap_elec_{ch}.png", dpi=200, bbox_inches='tight')
        plt.close()
    print("  -> Saved all DTW heatmaps.")

def plot_time_aware_tsne(trials_by_class, num_channels, results_dict):
    """Calculates true temporal clustering per electrode, ignoring phase shifts."""
    print("\n[4/7] Calculating Time-Aware t-SNE Clustering per Electrode...")
    
    class_names = [k for k, v in trials_by_class.items() if len(v) > 0]
    samples_per_class = 20 # Strictly enforced to prevent Colab OOM on O(N^2) math
    
    for ch in range(num_channels):
        print(f"  -> Computing pairwise DTW manifold for Channel {ch} (This takes time)...")
        
        subset_signals = []
        subset_labels = []
        
        for c_name in class_names:
            trials = trials_by_class[c_name][:samples_per_class]
            for trial in trials:
                # Subsample sequence length and reshape to (N, 1)
                subset_signals.append(trial[::10, ch].reshape(-1, 1)) 
                subset_labels.append(c_name)
                
        N = len(subset_signals)
        distance_matrix = np.zeros((N, N))
        
        # Build the exact O(N^2) elastic distance grid
        for i in range(N):
            for j in range(i + 1, N):
                dist, _ = fastdtw(subset_signals[i], subset_signals[j], dist=euclidean)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
        print(f"  -> Handing off to t-SNE algorithm for Channel {ch}...")
        tsne = TSNE(metric='precomputed', n_components=2, perplexity=15, random_state=42, init='random')
        X_tsne = tsne.fit_transform(distance_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=subset_labels, palette="tab20", s=50, alpha=0.8)
        plt.title(f"Time-Aware t-SNE (Phase Shifts Removed) - Electrode {ch}")
        plt.xlabel("Temporal Dimension 1")
        plt.ylabel("Temporal Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(f"time_tsne_elec_{ch}.png", dpi=200, bbox_inches='tight')
        plt.close()
    print("  -> Saved all Time-Aware t-SNE plots.")

# --- Previous analytical functions retained below for the 'all' option ---

def plot_chronological_envelopes(trials_by_class, num_channels, results_dict):
    print("\n[5/7] Generating Chronological Variance Envelopes...")
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels), sharex=True)
    if num_channels == 1: axes = [axes]
    
    colors = cm.get_cmap('tab20', len(trials_by_class))
    time_axis = np.linspace(0, 4.0, list(trials_by_class.values())[0][0].shape[0])
    results_dict['temporal_envelopes'] = {}
    
    for ch in range(num_channels):
        ax = axes[ch]
        for i, (c_name, trials) in enumerate(trials_by_class.items()):
            if not trials: continue
            stacked_trials = np.stack(trials, axis=0) 
            mean_sig = np.mean(np.abs(stacked_trials[:, :, ch]), axis=0)
            std_sig = np.std(np.abs(stacked_trials[:, :, ch]), axis=0)
            
            if c_name not in results_dict['temporal_envelopes']:
                results_dict['temporal_envelopes'][c_name] = {}
                
            skip_json = max(1, len(mean_sig) // 100)
            results_dict['temporal_envelopes'][c_name][f'Channel_{ch}'] = {
                'mean': mean_sig[::skip_json].tolist(),
                'std': std_sig[::skip_json].tolist()
            }
            
            skip_plot = max(1, len(mean_sig) // 100)
            ax.plot(time_axis[::skip_plot], mean_sig[::skip_plot], color=colors(i), label=c_name, linewidth=1.5)
            ax.fill_between(time_axis[::skip_plot], (mean_sig - std_sig)[::skip_plot], (mean_sig + std_sig)[::skip_plot], color=colors(i), alpha=0.1)
            
        ax.set_title(f"Electrode Channel {ch} Variance Envelope")
        ax.set_ylabel("Absolute Amplitude")
        ax.grid(True, alpha=0.3)
        if ch == 0: ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=8)
        
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("variance_envelopes.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_radar(trials_by_class, num_channels, results_dict):
    print("\n[6/7] Generating Spatial Muscle Signatures (Radar)...")
    angles = np.linspace(0, 2 * np.pi, num_channels, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = cm.get_cmap('tab20', len(trials_by_class))
    results_dict['spatial_rms_power'] = {}
    
    for i, (c_name, trials) in enumerate(trials_by_class.items()):
        if not trials: continue
        stacked_trials = np.stack(trials, axis=0)
        rms_per_channel = np.sqrt(np.mean(stacked_trials**2, axis=(0, 1))) 
        results_dict['spatial_rms_power'][c_name] = rms_per_channel.tolist()
        
        values = rms_per_channel.tolist()
        values += values[:1] 
        ax.plot(angles, values, color=colors(i), linewidth=2, label=c_name)
        ax.fill(angles, values, color=colors(i), alpha=0.05)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"Elec {i}" for i in range(num_channels)])
    ax.set_title("Geometric Muscle Activation Shape (Surface Crosstalk)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    plt.tight_layout()
    plt.savefig("spatial_radar.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_violin(trials_by_class, num_channels, results_dict):
    print("\n[7/7] Generating Feature Distribution Density (Violin Plots)...")
    import pandas as pd
    
    data = []
    results_dict['feature_statistics'] = {}
    
    for c_name, trials in trials_by_class.items():
        if not trials: continue
        sub_trials = trials[:min(200, len(trials))]
        class_rms_data = {ch: [] for ch in range(num_channels)}
        
        for trial in sub_trials:
            rms_vals = np.sqrt(np.mean(trial**2, axis=0))
            for ch in range(num_channels):
                data.append({"Class": c_name, "Electrode": f"Ch {ch}", "RMS": rms_vals[ch]})
                class_rms_data[ch].append(rms_vals[ch])
                
        results_dict['feature_statistics'][c_name] = {}
        for ch in range(num_channels):
            arr = np.array(class_rms_data[ch])
            results_dict['feature_statistics'][c_name][f'Channel_{ch}'] = {
                'min': float(np.min(arr)), 'max': float(np.max(arr)),
                'mean': float(np.mean(arr)), 'std': float(np.std(arr))
            }
                
    df = pd.DataFrame(data)
    plt.figure(figsize=(16, 8))
    sns.violinplot(data=df, x="Class", y="RMS", hue="Electrode", inner="quartile", density_norm="width")
    plt.title("Statistical Distribution of Mathematical Features (RMS)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("feature_violins.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Variance & Separability Analyzer")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--electrodes", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--option", type=str, choices=['envelope', 'radar', 'violin', 'tsne_flat', 'spectrogram', 'dtw_heatmap', 'time_tsne', 'all_temporal', 'all'], default='all')
    parser.add_argument("--no_filter", action="store_true", help="Disable the high-pass butterworth filter")
    parser.add_argument("--highpass_cutoff", type=float, default=20.0, help="Cutoff frequency for the filter")
    args = parser.parse_args()
    
    num_channels = len(args.electrodes)
    apply_filter = not args.no_filter
    trials_by_class = load_entire_dataset(args.data_root, args.electrodes, apply_filter=apply_filter, highpass_cutoff=args.highpass_cutoff)
    
    numerical_results = {}
    
    # New Temporal Functions
    if args.option in ['spectrogram', 'all_temporal', 'all']: plot_spectrograms(trials_by_class, num_channels)
    if args.option in ['dtw_heatmap', 'all_temporal', 'all']: plot_dtw_heatmaps(trials_by_class, num_channels, numerical_results)
    if args.option in ['time_tsne', 'all_temporal', 'all']: plot_time_aware_tsne(trials_by_class, num_channels, numerical_results)
    
    # Retained Spatial Functions
    if args.option in ['envelope', 'all']: plot_chronological_envelopes(trials_by_class, num_channels, numerical_results)
    if args.option in ['radar', 'all']: plot_spatial_radar(trials_by_class, num_channels, numerical_results)
    if args.option in ['violin', 'all']: plot_feature_violin(trials_by_class, num_channels, numerical_results)
    
    json_filename = "signal_metrics.json"
    with open(json_filename, "w") as f:
        json.dump(numerical_results, f, indent=4)
        
    print(f"\nAnalysis complete.")
    print(f"  -> Saved all plots to working directory.")
    print(f"  -> Saved raw mathematical data to {json_filename}.")