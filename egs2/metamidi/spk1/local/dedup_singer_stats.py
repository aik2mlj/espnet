import os
import random
import pandas as pd

# print number of singer_id directories in dedup_singers
output_dir = "dedup_singing_sad"
total_singers = len([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])
print(f"Total number of singer directories: {total_singers}")


# check if all files in dedup_singing are not tiny (> 10 kb). If not, print the file name and size.
print("Checking for tiny files (< 10KB)...")
tiny_files = []
for singer_dir in os.listdir(output_dir):
    singer_path = os.path.join(output_dir, singer_dir)
    if os.path.isdir(singer_path):
        for song_dir in os.listdir(singer_path):
            song_path = os.path.join(singer_path, song_dir)
            if os.path.isdir(song_path):
                # Check all WAV files in the song directory
                for file in os.listdir(song_path):
                    if file.endswith('.wav'):
                        wav_file = os.path.join(song_path, file)
                        size = os.path.getsize(wav_file)
                        if size < 10240:  # 10KB in bytes
                            tiny_files.append((wav_file, size))

if tiny_files:
    print(f"\nFound {len(tiny_files)} files smaller than 10KB:")
    for file_path, size in tiny_files:
        print(f"{file_path}: {size} bytes")
else:
    print("No files smaller than 10KB found.")

print("\n" + "="*50 + "\n")



# count number of files in each singer_id directory, return singer with 1 file, 2-4 files, and 4+ files
singer_with_1_file = 0
singer_with_2_4_files = 0
singer_with_4_plus_files = 0

for singer_dir in os.listdir(output_dir):
    singer_path = os.path.join(output_dir, singer_dir)
    if os.path.isdir(singer_path):
        num_files = len([f for f in os.listdir(singer_path) if os.path.isdir(os.path.join(singer_path, f))])
        if num_files == 1:
            singer_with_1_file += 1
        elif 2 <= num_files <= 4:
            singer_with_2_4_files += 1
        else:
            singer_with_4_plus_files += 1

# print number of singers with 1 file, 2-4 files, and 4+ files
print(f"Number of singers with 1 file: {singer_with_1_file}")
print(f"Number of singers with 2-4 files: {singer_with_2_4_files}")
print(f"Number of singers with 4+ files: {singer_with_4_plus_files}")

# Print percentages
print(f"\nPercentages:")
print(f"Singers with 1 file: {(singer_with_1_file/total_singers)*100:.2f}%")
print(f"Singers with 2-4 files: {(singer_with_2_4_files/total_singers)*100:.2f}%")
print(f"Singers with 4+ files: {(singer_with_4_plus_files/total_singers)*100:.2f}%")

# randomly sample 30 singers in the dedup_singing directory, print their corresponding names accoridng to their singer_id. reference to the file artist_to_singer_id_mapping_merged.csv.
print("\nRandomly sampling 30 singers...")

# Read the mapping file
mapping_df = pd.read_csv("artist_to_singer_id_mapping_merged.csv", header=None, names=["artist", "singer_id"])

# Get list of all singer directories
singer_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

# Randomly sample 10 singers
sampled_singers = random.sample(singer_dirs, min(10, len(singer_dirs)))

print("\nSampled singers and their artist names:")
print("-" * 50)
for singer_id in sampled_singers:
    # Look up the artist name in the mapping
    row = mapping_df[mapping_df["singer_id"] == singer_id]
    if not row.empty:
        artist_name = row.iloc[0]["artist"]
    else:
        artist_name = "Unknown"
    
    # Count number of songs for this singer
    singer_path = os.path.join(output_dir, singer_id)
    num_songs = len([f for f in os.listdir(singer_path) if os.path.isdir(os.path.join(singer_path, f))])
    
    print(f"Singer ID: {singer_id}")
    print(f"Artist Name: {artist_name}")
    print(f"Number of Songs: {num_songs}")
    print("-" * 50)
