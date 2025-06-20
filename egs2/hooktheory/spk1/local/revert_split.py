'''
This script reverts the train/test split and moves all files back to the original wav directory.
It removes the train/test layer from the directory structure.
'''

import os
import shutil
from pathlib import Path
import time
import glob

# Base paths
base_dir = '/home/aik2/sc-rawnet3/datasets/hooktheory/audio_16k'
wav_dir = os.path.join(base_dir, 'wav')
train_dir = os.path.join(base_dir, 'train', 'wav')
test_dir = os.path.join(base_dir, 'test', 'wav')

# Ensure the target directory exists
os.makedirs(wav_dir, exist_ok=True)

# Counters for statistics
files_moved = 0
errors = 0
start_time = time.time()

# Function to process files from a source directory
def process_directory(source_dir):
    global files_moved, errors
    
    # Get all wav files recursively
    wav_files = glob.glob(os.path.join(source_dir, "**", "*.wav"), recursive=True)
    total_files = len(wav_files)
    
    print(f"Found {total_files} files in {source_dir}")
    
    for i, file_path in enumerate(wav_files):
        # Get the path components
        path = Path(file_path)
        # The relative path should be something like id00001/song_id/00001.wav
        # We need to extract the parts after the 'wav' directory
        parts = path.parts
        
        # Find the 'wav' index in the path
        try:
            wav_index = parts.index('wav')
            # Get the parts after 'wav' (singer_id, song_id, filename)
            rel_parts = parts[wav_index+1:]
            
            # Create the destination path
            dest_path = os.path.join(wav_dir, *rel_parts)
            dest_dir = os.path.dirname(dest_path)
            
            # Create the destination directory
            os.makedirs(dest_dir, exist_ok=True)
            
            # Move the file
            try:
                shutil.move(file_path, dest_path)
                files_moved += 1
                
                # Print progress every 100 files
                if files_moved % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = files_moved / elapsed if elapsed > 0 else 0
                    print(f"Progress: {files_moved}/{total_files} files moved from {os.path.basename(source_dir)} directory")
                    print(f"Speed: {speed:.2f} files/second")
            except Exception as e:
                print(f"Error moving file {file_path}: {str(e)}")
                errors += 1
        except ValueError:
            print(f"Could not find 'wav' in path: {file_path}")
            errors += 1

# Process both train and test directories
print("Processing train directory...")
process_directory(train_dir)

print("\nProcessing test directory...")
process_directory(test_dir)

# Print statistics
elapsed_time = time.time() - start_time
print("\nRevert operation complete!")
print(f"Total files moved: {files_moved}")
print(f"Errors encountered: {errors}")
print(f"Total processing time: {elapsed_time:.2f} seconds") 