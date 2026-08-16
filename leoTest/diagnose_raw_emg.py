"""
diagnose_raw_emg.py
Directly analyzes a single raw .mat session file to diagnose SNR, DC Offset,
Powerline Interference, and physical sampling rate discrepancies.
"""
import argparse
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

def extract_field(struct_obj, field_name):
    """Safely extracts a field from a MATLAB struct."""
    if hasattr(struct_obj, field_name):
        return getattr(struct_obj, field_name)
    elif isinstance(struct_obj, np.ndarray) and struct_obj.dtype.names and field_name in struct_obj.dtype.names:
        return struct_obj[field_name].item()
    return None

def diagnose_session(file_path, channel_idx=0):
    print(f"Loading raw session file: {file_path}")
    
    try:
        mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Extract all 6 primary channels
    channel_arrays = []
    for i in range(1, 7):
        ch_key = f'ch{i}'
        if ch_key in mat_data:
            channel_arrays.append(mat_data[ch_key])
            
    if not channel_arrays:
        print("Error: Could not find ch1-ch6 in the file.")
        return
        
    signals = np.column_stack(channel_arrays)
    total_samples = signals.shape[0]
    print(f"Total Session Array Shape: {signals.shape} (Samples, Channels)")

    # Extract markers
    if 'mrk' not in mat_data:
        print("Error: No 'mrk' structure found in the file.")
        return
        
    trigger_times = extract_field(mat_data['mrk'], 'pos')
    trigger_codes = extract_field(mat_data['mrk'], 'toe')
    
    if trigger_times is None or trigger_codes is None:
        print("Error: Could not extract trigger positions/codes.")
        return

    if np.isscalar(trigger_times):
        trigger_times = np.array([trigger_times])
        trigger_codes = np.array([trigger_codes])

    print(f"Found {len(trigger_times)} total triggers.")

    # Find the first Rest (8) and the first Reaching Forward (11)
    rest_pos = None
    reach_pos = None
    
    for pos, code in zip(trigger_times, trigger_codes):
        code_val = int(str(code).replace('S', '').strip())
        if code_val == 8 and rest_pos is None:
            rest_pos = int(pos)
        elif code_val == 11 and reach_pos is None:
            reach_pos = int(pos)
            
        if rest_pos is not None and reach_pos is not None:
            break

    if rest_pos is None or reach_pos is None:
        print("Could not find both a Rest (8) and a Reach Forward (11) trigger.")
        return

    # The hardware sampling rate is approximately 120 samples/second.
    # Therefore, a 4-second physical window is only 480 data points.
    fs = 120
    window_samples = int(4.0 * fs) 
    
    print(f"\n--- Diagnostic Extraction ---")
    print(f"Hardware Sampling Rate: ~{fs} Hz")
    print(f"Extracting exactly {window_samples} samples (4 seconds) per movement.")
    
    rest_signal = signals[rest_pos : rest_pos + window_samples, channel_idx]
    reach_signal = signals[reach_pos : reach_pos + window_samples, channel_idx]

    # Plot 1: Raw Voltage Comparison (Signal to Noise check)
    plt.figure(figsize=(12, 6))
    time_axis = np.linspace(0, 4.0, window_samples)
    
    plt.plot(time_axis, rest_signal, label="Rest (Trigger 8)", color='blue', alpha=0.7)
    plt.plot(time_axis, reach_signal, label="Arm-Reaching Forward (Trigger 11)", color='red', alpha=0.7)
    
    plt.title(f"Raw Voltage SNR Diagnostic - Channel {channel_idx}")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Raw Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"diagnostic_raw_voltage_ch{channel_idx}.png")
    print("Saved -> diagnostic_raw_voltage.png")
    plt.close()

    # Plot 2: Fast Fourier Transform (DC Offset & Powerline check)
    print("\nCalculating FFT Frequency Spectrum...")
    freqs, psd = signal.welch(signals[:, channel_idx], fs=fs, nperseg=1024)
    
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd, color='purple')
    plt.title(f"Full Session FFT Spectrum - Channel {channel_idx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"diagnostic_fft_spectrum_ch{channel_idx}.png")
    print("Saved -> diagnostic_fft_spectrum.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the .mat file")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to diagnose (0-5)")
    args = parser.parse_args()
    
    diagnose_session(args.file, args.channel)