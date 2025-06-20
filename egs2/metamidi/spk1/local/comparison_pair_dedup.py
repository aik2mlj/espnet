import json
import os
import random
import pandas as pd
import time
from itertools import combinations

# Load the singer_data.json
with open('/home/aik2/sc-rawnet3/datasets/metamidi/singer_data_deduplicated.json', 'r') as f:
    singer_data = json.load(f)

# Make a list of all 16k audio file paths in the test set
all_files = []
test_singers = []

# First, identify all test singers and collect their file paths
for singer_id, singer_info in singer_data.items():
    if singer_info.get("is_test", 0) == 1:
        test_singers.append(singer_id)
        all_files.extend(singer_info.get("audio_16k_split", []))

print(f"Found {len(test_singers)} test singers with a total of {len(all_files)} audio files.")

# Initialize list to store all pairs
all_pairs = []
# Set to track existing pairs (as tuples of (file1, file2))
existing_pairs = set()

# Iterate over all singers in the test set
for singer_id in test_singers:
    start_time = time.time()
    singer_files = singer_data[singer_id].get("audio_16k_split", [])
    num_files = len(singer_files)
    
    print(f"\nProcessing singer {singer_id} with {num_files} files...")
    
    if num_files < 2:
        print(f"Skipping singer {singer_id} with fewer than 2 files.")
        continue
    
    # Create a list of files from other singers
    other_singers_files = [f for f in all_files if f not in singer_files]
    
    # Special case for singers with 2 or 3 files
    if num_files in [2, 3]:
        print(f"  Special handling for singer with {num_files} files")
        # Create all possible true pairs
        true_pairs_created = []
        for file1, file2 in combinations(singer_files, 2):
            pair_tuple = tuple(sorted([file1, file2]))
            if pair_tuple not in existing_pairs:
                all_pairs.append([1, file1, file2])
                existing_pairs.add(pair_tuple)
                true_pairs_created.append((file1, file2))
        
        # Create equal number of false pairs (one for each true pair)
        if num_files == 2:
            # For 2 files, create only 1 false pair to match the 1 true pair
            if other_singers_files and true_pairs_created:
                file1 = true_pairs_created[0][0]  # Use the first file from the true pair
                file2_false = random.choice(other_singers_files)
                pair_tuple = tuple(sorted([file1, file2_false]))
                if pair_tuple not in existing_pairs:
                    all_pairs.append([0, file1, file2_false])
                    existing_pairs.add(pair_tuple)
        else:
            # For 3 files, create one false pair for each file
            for file1 in singer_files:
                if other_singers_files:
                    file2_false = random.choice(other_singers_files)
                    pair_tuple = tuple(sorted([file1, file2_false]))
                    if pair_tuple not in existing_pairs:
                        all_pairs.append([0, file1, file2_false])
                        existing_pairs.add(pair_tuple)
    else:
        # For each file in the singer's files, create one true pair and one false pair
        for file1 in singer_files:
            max_attempts = 24  # Maximum number of attempts to create unique pairs
            attempts = 0
            true_pair_created = False
            
            # Create true pair: pair with another file from the same singer
            other_same_singer_files = [f for f in singer_files if f != file1]
            if other_same_singer_files:
                while attempts < max_attempts:
                    attempts += 1
                    file2_true = random.choice(other_same_singer_files)
                    # Verify the files are from the same singer (using -3 to get singer_id from path)
                    file1_parts = file1.split('/')
                    file2_parts = file2_true.split('/')
                    file1_singer = file1_parts[-3]  # Get singer ID from path
                    file2_singer = file2_parts[-3]
                    
                    if file1_singer != file2_singer:
                        print(f"WARNING: True pair files from different singers: {file1} and {file2_true}")
                        break
                    
                    # Check if this pair already exists
                    pair_tuple = tuple(sorted([file1, file2_true]))  # Sort to ensure consistent ordering
                    if pair_tuple not in existing_pairs:
                        all_pairs.append([1, file1, file2_true])
                        existing_pairs.add(pair_tuple)
                        true_pair_created = True
                        break
                
                if attempts >= max_attempts:
                    print(f"  Skipping pair creation for {file1} after {max_attempts} attempts")
                    continue
            
            # Only create false pair if we successfully created a true pair
            if true_pair_created and other_singers_files:
                attempts = 0
                while attempts < max_attempts:
                    attempts += 1
                    file2_false = random.choice(other_singers_files)
                    # Verify the files are from different singers (using -3 to get singer_id from path)
                    file1_parts = file1.split('/')
                    file2_parts = file2_false.split('/')
                    file1_singer = file1_parts[-3]
                    file2_singer = file2_parts[-3]
                    
                    if file1_singer == file2_singer:
                        print(f"WARNING: False pair files from same singer: {file1} and {file2_false}")
                        break
                    
                    # Check if this pair already exists
                    pair_tuple = tuple(sorted([file1, file2_false]))  # Sort to ensure consistent ordering
                    if pair_tuple not in existing_pairs:
                        all_pairs.append([0, file1, file2_false])
                        existing_pairs.add(pair_tuple)
                        break
                
                if attempts >= max_attempts:
                    print(f"  Skipping false pair creation for {file1} after {max_attempts} attempts")
                    # Remove the true pair we created since we couldn't create its corresponding false pair
                    all_pairs.pop()
                    existing_pairs.remove(tuple(sorted([file1, file2_true])))
    
    # Calculate statistics for this singer
    end_time = time.time()
    duration = end_time - start_time
    singer_pairs = [p for p in all_pairs if p[1].split('/')[-3] == singer_id]
    singer_true_pairs = [p for p in singer_pairs if p[0] == 1]
    singer_false_pairs = [p for p in singer_pairs if p[0] == 0]
    
    print(f"\nCompleted singer {singer_id}:")
    print(f"  Time taken: {duration:.2f} seconds")
    print(f"  Total pairs created: {len(singer_pairs)}")
    print(f"  True pairs: {len(singer_true_pairs)}")
    print(f"  False pairs: {len(singer_false_pairs)}")

# Convert to DataFrame
comparison_df = pd.DataFrame(all_pairs, columns=["is_same_singer", "file_path_1", "file_path_2"])

# Verify the pairs
true_pairs = comparison_df[comparison_df['is_same_singer'] == 1]
false_pairs = comparison_df[comparison_df['is_same_singer'] == 0]

print("\nFinal Verification:")
print(f"Total pairs created: {len(comparison_df)}")
print(f"True pairs: {len(true_pairs)}")
print(f"False pairs: {len(false_pairs)}")

# Save to comparison_pairs.txt
comparison_df.to_csv("/home/aik2/sc-rawnet3/datasets/metamidi/comparison_pairs_dedup.txt", 
                      sep="\t", index=False, header=False)

print(f"\nSaved to comparison_pairs_dedup.txt")