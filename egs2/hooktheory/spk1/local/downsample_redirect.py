'''
This script is used to downsample the redirect hooktheory to 16kHz and save the audio files in the structure required for Kaldi.
It supports resuming by skipping files that have already been processed.
'''

import os
import pandas as pd
import json
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import time

# redirect save path
redirect_path = '/home/aik2/sc-rawnet3/datasets/hooktheory/audio_16k'

# mkdir if not exists
os.makedirs(redirect_path, exist_ok=True)

# Load the singer data from the JSON file
with open('/home/aik2/sc-rawnet3/datasets/hooktheory/singer_data_complete.json', 'r') as f:
    singer_data = json.load(f)

# Create a directory for each singer based on their ID in the JSON
for singer_id in singer_data.keys():
    # Create directory using the singer_id (which is already in "idXXXXX" format)
    os.makedirs(os.path.join(redirect_path, singer_id), exist_ok=True)
    print(f"Created directory for {singer_id}")

# Count total files to process
total_files = sum(len(info['audio_paths']) for info in singer_data.values())
processed_files = 0
skipped_files = 0
start_time = time.time()

print(f"Starting processing of {total_files} total files...")

# Process audio files for each singer
for singer_id, singer_info in singer_data.items():
    # Access audio paths
    audio_paths = singer_info['audio_paths']
    
    # Process each audio file for the singer
    for audio_path in audio_paths:
        try:
            # Extract the folder name containing vocals.wav
            # e.g., from "/path/to/bWgMwEPPolX/vocals.wav" extract "bWgMwEPPolX"
            folder_name = Path(audio_path).parent.name
            
            # Create target directory
            target_dir = os.path.join(redirect_path, singer_id, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # Define output file path
            output_path = os.path.join(target_dir, 'vocals.wav')
            
            # Check if file already exists (resume functionality)
            if os.path.exists(output_path):
                print(f"Skipping already processed file: {output_path}")
                skipped_files += 1
                processed_files += 1
                continue
            
            print(f"Processing {audio_path} for ID: {singer_id}")
            print(f"-> Saving to {output_path}")
            
            # Load the audio file - explicitly set mono=True to ensure mono output
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Downsample to 16kHz if needed
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Save the downsampled mono audio
            sf.write(output_path, y, 16000, subtype='PCM_16')
            print(f"Successfully processed: {output_path}")
            processed_files += 1
            
            # Print progress every 10 files
            if processed_files % 10 == 0:
                elapsed_time = time.time() - start_time
                files_per_second = processed_files / elapsed_time if elapsed_time > 0 else 0
                estimated_remaining = (total_files - processed_files) / files_per_second if files_per_second > 0 else 0
                
                print(f"Progress: {processed_files}/{total_files} files processed ({(processed_files/total_files)*100:.1f}%)")
                print(f"Speed: {files_per_second:.2f} files/second, Est. remaining time: {estimated_remaining/60:.1f} minutes")
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue

total_time = time.time() - start_time
print(f"Downsampling completed!")
print(f"Processed: {processed_files}/{total_files} files")
print(f"Skipped (already existed): {skipped_files} files")
print(f"Total time: {total_time/60:.1f} minutes")




