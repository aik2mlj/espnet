'''
This script creates pairs of audio files for testing singer verification.
For each singer in the test set:
1. Creates pairs of files from the same singer (positive pairs, label 1)
2. Creates pairs with files from different singers (negative pairs, label 0)
'''

# imports
import os
import json
import random
import itertools
from pathlib import Path
import glob

# Base paths
base_dir = '/home/aik2/sc-rawnet3/datasets/hooktheory/audio_16k'
split_path = os.path.join(base_dir, 'split_by_singer.json')
output_path = os.path.join(base_dir, 'test_pairs.txt')
test_wav_dir = os.path.join(base_dir, 'test', 'wav')

# open split_by_singer.json
try:
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    print(f"Successfully loaded {split_path}")
except Exception as e:
    print(f"Error loading split_by_singer.json: {e}")
    exit(1)

# make a list of all the wav file paths in the TEST set only
all_wav_files = []
test_singers = {}  # Dictionary to store test singers and their files

# Group files by singer
for singer_id, singer_info in split_data.items():
    # Get all paths for this singer
    for audio_item in singer_info.get('audio_paths', []):
        path_16k = audio_item.get('path_16k')
        split = audio_item.get('split')
        
        # Only include test files
        if path_16k and os.path.exists(path_16k) and split == 'test':
            # Convert to relative path starting from after test/wav
            rel_path = os.path.relpath(path_16k, start=test_wav_dir)
            all_wav_files.append(rel_path)
            
            # Add to the test_singers dictionary
            if singer_id not in test_singers:
                test_singers[singer_id] = []
            test_singers[singer_id].append(rel_path)

print(f"Total test wav files found: {len(all_wav_files)}")
print(f"Test singers found: {len(test_singers)}")

# Create pairs
all_pairs = []
used_pairs = set()  # Track all pairs to ensure no duplicates

# For each singer in the test set
for singer_id, singer_files in test_singers.items():
    print(f"Processing test singer {singer_id} with {len(singer_files)} files")
    
    # Make all combinations of 2 files from this singer
    same_singer_pairs = list(itertools.combinations(singer_files, 2))
    print(f"  Created {len(same_singer_pairs)} positive pairs")
    
    # Get files not from this singer - do this once per singer
    other_singers_files = [f for f in all_wav_files if singer_id not in f]
    random.shuffle(other_singers_files)  # Shuffle to ensure random selection
    
    # For each positive pair, create a corresponding negative pair
    for i, (file1, file2) in enumerate(same_singer_pairs):
        # Create a unique key for this pair to check for duplicates
        pair_key = tuple(sorted([file1, file2]))
        if pair_key in used_pairs:
            print(f"  Skipping duplicate positive pair: {file1}, {file2}")
            continue
            
        # Add the positive pair (label 1)
        all_pairs.append((1, file1, file2))
        used_pairs.add(pair_key)
        
        # Find a file from a different singer for a negative pair
        if other_singers_files:
            # Take the next file from the shuffled list
            negative_idx = i % len(other_singers_files)
            random_file = other_singers_files[negative_idx]
            
            # Create a unique key for this negative pair
            neg_pair_key1 = tuple(sorted([file1, random_file]))
            neg_pair_key2 = tuple(sorted([file2, random_file]))
            
            # Check if this negative pair already exists
            if neg_pair_key1 not in used_pairs and neg_pair_key2 not in used_pairs:
                # Add the negative pair (label 0)
                all_pairs.append((0, file1, random_file))
                used_pairs.add(neg_pair_key1)
            else:
                # Try to find another file that hasn't been used
                found_unused = False
                for alt_file in other_singers_files:
                    alt_pair_key = tuple(sorted([file1, alt_file]))
                    if alt_pair_key not in used_pairs:
                        all_pairs.append((0, file1, alt_file))
                        used_pairs.add(alt_pair_key)
                        found_unused = True
                        break
                
                if not found_unused:
                    print(f"  Warning: Could not find unused pair for {file1}")
        else:
            print(f"  Warning: No files found from other singers")

# Final check for duplicate pairs
unique_pairs = set()
final_pairs = []
duplicates_found = 0

for label, file1, file2 in all_pairs:
    # Normalize the pair by sorting
    if label == 1:
        # For positive pairs, order doesn't matter
        pair_key = (label, tuple(sorted([file1, file2])))
    else:
        # For negative pairs, first file is from test set, second is from other singers
        pair_key = (label, file1, file2)
    
    if pair_key not in unique_pairs:
        unique_pairs.add(pair_key)
        final_pairs.append((label, file1, file2))
    else:
        duplicates_found += 1

if duplicates_found > 0:
    print(f"Found and removed {duplicates_found} duplicate pairs")

print(f"Total pairs created: {len(final_pairs)}")
print(f"Positive pairs: {sum(1 for label, _, _ in final_pairs if label == 1)}")
print(f"Negative pairs: {sum(1 for label, _, _ in final_pairs if label == 0)}")

# Write pairs to file
try:
    with open(output_path, 'w') as f:
        for label, file1, file2 in final_pairs:
            f.write(f"{label} {file1} {file2}\n")
    print(f"Successfully saved pairs to {output_path}")
except Exception as e:
    print(f"Error saving pairs to {output_path}: {e}")

print("Done!")

