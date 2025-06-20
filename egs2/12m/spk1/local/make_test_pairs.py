# Make test pairs

import json
import random
from pathlib import Path
from itertools import combinations

# first, load the test_set_paths.json file
print("Loading test_set_paths.json...")
with open('test_set_paths.json', 'r') as f:
    test_data = json.load(f)

# get a set of all file paths in the test set by going through each singer_id and getting the list of paths. Store those in a set called all_test_set_paths
all_test_set_paths = set()
for singer_id, data in test_data.items():
    for path in data['song_paths']:
        all_test_set_paths.add(path)

print(f"Total test set paths: {len(all_test_set_paths)}")
print(f"Total singers in test set: {len(test_data)}")

pairs = []
used_pairs = set()  # To track used pairs and avoid duplicates

# Progress tracking
singers_processed = 0
total_singers = len(test_data)

for singer_id, data in test_data.items():
    singer_paths = data['song_paths']
    num_songs = len(singer_paths)
    
    # Skip singers with less than 2 songs (can't make true pairs)
    if num_songs < 2:
        print(f"Warning: Singer {singer_id} has only {num_songs} songs, skipping...")
        singers_processed += 1
        continue
    
    # Get other singers' songs for false pairs
    other_singer_paths = list(all_test_set_paths - set(singer_paths))
    
    if len(other_singer_paths) == 0:
        print(f"Warning: No other singer paths available for false pairs for {singer_id}")
        singers_processed += 1
        continue
    
    # print(f"Processing singer {singer_id} with {num_songs} files...")
    
    if num_songs == 2:
        # Special case: 1 true pair, 1 false pair
        true_pairs = list(combinations(singer_paths, 2))
        
        # Add the single true pair
        pair = tuple(sorted(true_pairs[0]))
        if pair not in used_pairs:
            pairs.append(f"1 {pair[0]} {pair[1]}")
            used_pairs.add(pair)
        
        # Add 1 false pair
        singer_song = random.choice(singer_paths)
        other_song = random.choice(other_singer_paths)
        false_pair = tuple(sorted([singer_song, other_song]))
        attempts = 0
        while false_pair in used_pairs and attempts < 100:
            other_song = random.choice(other_singer_paths)
            false_pair = tuple(sorted([singer_song, other_song]))
            attempts += 1
        
        if false_pair not in used_pairs:
            pairs.append(f"0 {singer_song} {other_song}")
            used_pairs.add(false_pair)

    # Add a special case for singers with 3 songs, since the true pairs have to be 12, 13, and 23.
    elif num_songs == 3:
        # Special case: 3 true pairs (all combinations), 3 false pairs
        true_pairs = list(combinations(singer_paths, 2))
        
        # Add all 3 true pairs (12, 13, 23)
        files_with_true_pairs = []
        for pair in true_pairs:
            sorted_pair = tuple(sorted(pair))
            if sorted_pair not in used_pairs:
                pairs.append(f"1 {pair[0]} {pair[1]}")
                used_pairs.add(sorted_pair)
                # Track the first file of each pair for false pair creation
                if pair[0] not in files_with_true_pairs:
                    files_with_true_pairs.append(pair[0])
        
        # Create 3 false pairs: one for each song
        for singer_song in singer_paths:
            # Try to create a unique false pair
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts:
                other_song = random.choice(other_singer_paths)
                false_pair = tuple(sorted([singer_song, other_song]))
                
                if false_pair not in used_pairs:
                    pairs.append(f"0 {singer_song} {other_song}")
                    used_pairs.add(false_pair)
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"  Warning: Could not create unique false pair for {singer_song} after {max_attempts} attempts")
    
    else:
        # For singers with more than 3 files: 1 true pair and 1 false pair per file
        # Each file is used exactly once for true pairs and once for false pairs
        
        # Track which files successfully got true pairs
        files_with_true_pairs = []
        
        # Create true pairs: each file paired with another random file from same singer
        for file1 in singer_paths:
            # Get other files from same singer (excluding current file)
            other_same_singer_files = [f for f in singer_paths if f != file1]
            
            # Try to create a unique true pair
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts:
                file2_true = random.choice(other_same_singer_files)
                true_pair = tuple(sorted([file1, file2_true]))
                
                if true_pair not in used_pairs:
                    pairs.append(f"1 {file1} {file2_true}")
                    used_pairs.add(true_pair)
                    files_with_true_pairs.append(file1)  # Track successful true pair
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"  Warning: Could not create unique true pair for {file1} after {max_attempts} attempts")
        
        # Create false pairs: only for files that successfully got true pairs
        for file1 in files_with_true_pairs:
            # Try to create a unique false pair
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts:
                file2_false = random.choice(other_singer_paths)
                false_pair = tuple(sorted([file1, file2_false]))
                
                if false_pair not in used_pairs:
                    pairs.append(f"0 {file1} {file2_false}")
                    used_pairs.add(false_pair)
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"  Warning: Could not create unique false pair for {file1} after {max_attempts} attempts")
    
    # Update progress counter
    singers_processed += 1
    
    # Print checkpoint every 100 singers
    if singers_processed % 100 == 0:
        true_pairs_so_far = sum(1 for pair in pairs if pair.startswith('1'))
        false_pairs_so_far = sum(1 for pair in pairs if pair.startswith('0'))
        print(f"\n=== CHECKPOINT ===")
        print(f"Processed {singers_processed}/{total_singers} singers ({singers_processed/total_singers*100:.1f}%)")
        print(f"Generated {len(pairs)} pairs so far ({true_pairs_so_far} true, {false_pairs_so_far} false)")
        print(f"==================\n")

# Count true and false pairs
true_pairs_count = sum(1 for pair in pairs if pair.startswith('1'))
false_pairs_count = sum(1 for pair in pairs if pair.startswith('0'))

print(f"\nGenerated {len(pairs)} pairs ({true_pairs_count} true pairs, {false_pairs_count} false pairs)")
print(f"Used pairs tracking: {len(used_pairs)} unique pairs tracked")

# Store the resulting columns containing the pairs in comparison_pairs.txt
# don't include any headers. Just have the first column as 0 or 1, and the second and third the corresponding paths.
with open('comparison_pairs.txt', 'w') as f:
    for pair in pairs:
        f.write(pair + '\n')

print("Saved pairs to comparison_pairs.txt")
print("Format: label path1 path2 (1=same singer, 0=different singers)")
