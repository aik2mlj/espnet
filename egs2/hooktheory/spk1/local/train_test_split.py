"""
This script splits the HookTheory dataset into training and testing sets based on the provided split in singer_data_split.json
"""

# imports
import os
import json
import shutil
from pathlib import Path
import time

# Base paths
base_dir = "/home/aik2/sc-rawnet3/datasets/hooktheory/audio_16k"
source_wav_dir = os.path.join(base_dir, "wav")
train_dir = os.path.join(base_dir, "train", "wav")
test_dir = os.path.join(base_dir, "test", "wav")

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# open singer_data_split.json
split_file_path = "/home/aik2/sc-rawnet3/datasets/hooktheory/singer_data_split.json"

try:
    with open(split_file_path, "r") as f:
        singer_data = json.load(f)
    print(f"Successfully loaded {split_file_path}")
except Exception as e:
    print(f"Error loading split file: {e}")
    exit(1)

# Counters for statistics
total_files = 0
train_files = 0
test_files = 0
existing_files = 0
missing_files = 0
start_time = time.time()

# Access each singer_id in singer_data_split.json
for singer_id, singer_info in singer_data.items():
    print(f"Processing singer: {singer_id}")

    # for each singer_id, access the audio_paths
    for audio_item in singer_info["audio_paths"]:
        total_files += 1

        # Extract info
        original_path = audio_item["path"]
        path_16k = audio_item["path_16k"]
        is_train = audio_item["is_train"]

        # Extract song_id from the path
        song_id = Path(original_path).parent.name

        # Check if the file exists in the audio_16k folder
        if os.path.exists(path_16k):
            existing_files += 1

            # Determine destination based on is_train flag
            if is_train == 1:
                # Move to train directory
                dest_dir = os.path.join(train_dir, singer_id, song_id)
                train_files += 1
            else:
                # Move to test directory
                dest_dir = os.path.join(test_dir, singer_id, song_id)
                test_files += 1

            # Create destination directory
            os.makedirs(dest_dir, exist_ok=True)

            # Define destination file path
            dest_file = os.path.join(dest_dir, "00001.wav")

            # Move the file to destination instead of copying
            try:
                shutil.move(path_16k, dest_file)
                print(f"Moved: {path_16k} -> {dest_file}")

                # Print progress every 100 files
                if (train_files + test_files) % 100 == 0:
                    elapsed_time = time.time() - start_time
                    files_per_second = (
                        (train_files + test_files) / elapsed_time
                        if elapsed_time > 0
                        else 0
                    )
                    print(
                        f"Progress: {train_files + test_files}/{existing_files} files processed"
                    )
                    print(f"Speed: {files_per_second:.2f} files/second")
            except Exception as e:
                print(f"Error moving file {path_16k}: {str(e)}")
        else:
            missing_files += 1
            print(f"Missing file: {path_16k}")

# Print final statistics
print("\nProcessing complete!")
print(f"Total files in split JSON: {total_files}")
print(f"Existing files found: {existing_files}")
print(f"Missing files: {missing_files}")
print(f"Files moved to train directory: {train_files}")
print(f"Files moved to test directory: {test_files}")
print(f"Total processing time: {(time.time() - start_time):.2f} seconds")
