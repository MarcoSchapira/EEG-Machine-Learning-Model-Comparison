import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CHANGE THIS PATH ---
PT_FILE_PATH = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_PT_files/sub_1/trial_00200.pt"


def inspect_and_plot(pt_path):
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{pt_path} not found")

    print(f"\nLoading file: {pt_path}\n")

    obj = torch.load(pt_path)

    print("Keys inside .pt file:")
    for k in obj.keys():
        print(f"  - {k}")

    data = obj['data']
    label = obj['label']

    print("\n--- Tensor Information ---")
    print(f"Data type: {type(data)}")
    print(f"Data dtype: {data.dtype}")
    print(f"Data shape: {data.shape}")  # (channels, samples)

    print(f"\nLabel type: {type(label)}")
    print(f"Label dtype: {label.dtype}")
    print(f"Label value: {label.item()}")

    # Convert to numpy for plotting
    data_np = data.numpy()

    # --- Plot Heatmap ---
    plt.figure(figsize=(12, 6))
    plt.imshow(
        data_np,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time Samples")
    plt.ylabel("Channels")
    plt.title(f"EEG Trial Heatmap (Label: {label.item()})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inspect_and_plot(PT_FILE_PATH)
