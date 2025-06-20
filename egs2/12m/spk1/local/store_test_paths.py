# Make test pairs

import json
import os
from pathlib import Path
import statistics

print("Loading singer ID mapping...")
# Load the singer_id_mapping_filtered.json file
with open('/home/aik2/sc-rawnet3/datasets/12m/singer_id_mapping_filtered.json', 'r') as f:
    singer_mapping = json.load(f)

def process_split_directory(split_dir, output_filename):
    """Process a split directory (test or exp) and create JSON with paths"""
    split_path = Path(f'/home/sc/espnet/egs2/12m/12m/{split_dir}')
    
    if not split_path.exists():
        print(f"Warning: {split_path} does not exist")
        return
    
    result = {}
    wav_counts = []  # Track number of .wav files per singer for statistics
    
    # Get all singer_id folders in the split directory
    singer_folders = [f for f in split_path.iterdir() if f.is_dir()]
    print(f"Processing {len(singer_folders)} singers in {split_dir} set...")
    
    for singer_folder in singer_folders:
        singer_id = singer_folder.name  # e.g., "id12345"
        
        # Get artist name from singer mapping
        artist_name = None
        if singer_id in singer_mapping:
            # Get the lowercase artist name (already processed)
            artist_name = singer_mapping[singer_id].get('lowercase')
            if not artist_name:
                print(f"DEBUG: Singer {singer_id} found in mapping but has no lowercase artist name")
        else:
            print(f"DEBUG: Singer {singer_id} not found in singer_mapping")
        
        # Get all .wav files recursively under this singer's folder
        wav_paths = []
        for root, dirs, files in os.walk(singer_folder):
            for file in files:
                if file.lower().endswith('.wav'):
                    # Create absolute path
                    absolute_path = os.path.join(root, file)
                    wav_paths.append(absolute_path)
        
        # Track count for statistics
        wav_counts.append(len(wav_paths))
        
        # Store in result
        result[singer_id] = {
            "artist_name": artist_name,
            "song_paths": sorted(wav_paths)  # Sort for consistency
        }
    
    # Print statistics about .wav files per singer
    if wav_counts:
        print(f"\n=== .wav File Statistics for {split_dir} set ===")
        print(f"Total singers: {len(wav_counts)}")
        print(f"Maximum .wav files per singer: {max(wav_counts)}")
        print(f"Minimum .wav files per singer: {min(wav_counts)}")
        print(f"Mean .wav files per singer: {statistics.mean(wav_counts):.2f}")
        print(f"Median .wav files per singer: {statistics.median(wav_counts):.2f}")
        print(f"Total .wav files: {sum(wav_counts)}")
        
        # Show distribution
        from collections import Counter
        count_distribution = Counter(wav_counts)
        print(f"Distribution of .wav files per singer:")
        for count, num_singers in sorted(count_distribution.items()):
            print(f"  {count} files: {num_singers} singers")
    
    # Save to JSON file
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(result)} singers to {output_filename}")
    
    return result

# Process test set
print("\n=== Processing Test Set ===")
test_data = process_split_directory('test/wav', 'test_set_paths.json')

# Process exp set
print("\n=== Processing Exp Set ===")
exp_data = process_split_directory('exp/wav', 'exp_set_paths.json')

print("\n=== Summary ===")
if test_data:
    print(f"Test set: {len(test_data)} singers")
if exp_data:
    print(f"Exp set: {len(exp_data)} singers")

# Show a sample of what was created
if test_data:
    sample_singer = next(iter(test_data.keys()))
    print(f"\nSample entry from test set ({sample_singer}):")
    print(f"  Artist: {test_data[sample_singer]['artist_name']}")
    print(f"  Songs: {len(test_data[sample_singer]['song_paths'])}")
    if test_data[sample_singer]['song_paths']:
        print(f"  First song: {test_data[sample_singer]['song_paths'][0]}")

print("\nCompleted successfully!")
