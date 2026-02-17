import torch
import os

# Path to your metadata file
PREPROCESSED_DIR = "/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_PT_files"
metadata_path = os.path.join(PREPROCESSED_DIR, "metadata.pt")

# Load metadata
metadata = torch.load(metadata_path)

# Print contents clearly
print("\n--- Metadata Contents ---\n")

if isinstance(metadata, dict):
    for key, value in metadata.items():
        print(f"{key}: {value}")
else:
    print("Metadata is not a dictionary. Raw contents:")
    print(metadata)

print("\n--------------------------\n")
