import os
import json
import librosa
import soundfile as sf
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys


# Function to process a single audio file
def process_audio_file(audio_path, redirect_path, set_type, singer_id):
    try:
        # Extract the folder name by removing the "-vocals.wav" suffix
        filename = Path(audio_path).name
        folder_name = filename.rsplit("-vocals.wav", 1)[0]

        # Create target directory based on set type and singer_id
        target_dir = os.path.join(redirect_path, set_type, singer_id, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        # Define output file path with new naming convention
        output_path = os.path.join(target_dir, "00001.wav")

        # Check if file already exists (resume functionality)
        if os.path.exists(output_path):
            # If file exists but is too small (potentially corrupted), delete and reprocess
            if os.path.getsize(output_path) < 1000:
                try:
                    os.remove(output_path)
                    print(
                        f"Deleted small output file (<1KB): {output_path}, will reprocess"
                    )
                except Exception as e:
                    print(f"Error deleting small file {output_path}: {str(e)}")
            else:
                # File exists and is of reasonable size, skip processing
                return f"Skipping already processed file: {output_path}"

        # Load the audio file - explicitly set mono=True to ensure mono output
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Downsample to 16kHz if needed
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Save the downsampled mono audio
        sf.write(output_path, y, 16000, subtype="PCM_16")
        return f"Successfully processed: {output_path}"

    except Exception as e:
        return f"Error processing {audio_path}: {str(e)}"


# Redirect save path
redirect_path = "/home/aik2/sc-rawnet3/datasets/metamidi/audio_16k"

# Load the singer data from the JSON file
with open("/home/aik2/sc-rawnet3/datasets/metamidi/singer_data.json", "r") as f:
    singer_data = json.load(f)

# Create train and test directories
os.makedirs(os.path.join(redirect_path, "train"), exist_ok=True)
os.makedirs(os.path.join(redirect_path, "test"), exist_ok=True)

# Count total files to process
total_files = sum(len(info["audio_paths"]) for info in singer_data.values())
processed_files = 0
skipped_files = 0
start_time = time.time()

print(f"Starting processing of {total_files} total files...")

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor() as executor:
    futures = []
    for singer_id, singer_info in singer_data.items():
        set_type = "test" if singer_info.get("is_test", 0) == 1 else "train"
        audio_paths = singer_info["audio_paths"]
        for audio_path in audio_paths:
            futures.append(
                executor.submit(
                    process_audio_file, audio_path, redirect_path, set_type, singer_id
                )
            )

    for future in as_completed(futures):
        result = future.result()
        print(result)
        if "Successfully processed" in result:
            processed_files += 1
        elif "Skipping" in result:
            skipped_files += 1
            total_files -= 1

        # Print progress every 10 files
        if processed_files % 10 == 0:
            elapsed_time = time.time() - start_time
            files_per_second = processed_files / elapsed_time if elapsed_time > 0 else 0
            estimated_remaining = (
                (total_files - processed_files) / files_per_second
                if files_per_second > 0
                else 0
            )

            print(
                f"Progress: {processed_files}/{total_files} files processed ({(processed_files / (total_files or 0.0001)) * 100:.1f}%)"
            )
            print(
                f"Speed: {files_per_second:.2f} files/second, Est. remaining time: {estimated_remaining / 60:.1f} minutes"
            )
            sys.stdout.flush()

total_time = time.time() - start_time
print(f"Downsampling completed!")
print(f"Processed: {processed_files}/{total_files} files")
print(f"Skipped (already existed): {skipped_files} files")
print(f"Total time: {total_time / 60:.1f} minutes")

