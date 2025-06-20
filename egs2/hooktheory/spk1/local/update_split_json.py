'''
This script updates the "path_16k" values in singer_data_split.json to reflect 
the new locations in train/test directories
'''

import os
import json

# Path to the JSON file
split_file_path = '/home/aik2/sc-rawnet3/datasets/hooktheory/singer_data_split.json'

# Load the JSON file
try:
    with open(split_file_path, 'r') as f:
        singer_data = json.load(f)
    print(f"Successfully loaded {split_file_path}")
except Exception as e:
    print(f"Error loading split file: {e}")
    exit(1)

# Counter for changes
updated_count = 0

# Base paths
base_dir = '/home/aik2/sc-rawnet3/datasets/hooktheory/audio_16k'
old_base_path = os.path.join(base_dir, 'wav')
train_base_path = os.path.join(base_dir, 'train', 'wav')
test_base_path = os.path.join(base_dir, 'test', 'wav')

# Update the path_16k values for each audio item
for singer_id, singer_info in singer_data.items():
    print(f"Processing singer: {singer_id}")
    
    for audio_item in singer_info['audio_paths']:
        old_path = audio_item['path_16k']
        is_train = audio_item['is_train']
        
        # Determine if it's train or test
        if is_train == 1:
            # Replace the base path with train path
            new_path = old_path.replace(old_base_path, train_base_path)
        else:
            # Replace the base path with test path
            new_path = old_path.replace(old_base_path, test_base_path)
        
        # Check if the file exists in the new location
        if os.path.exists(new_path):
            # Update the path_16k value
            audio_item['path_16k'] = new_path
            updated_count += 1
            
            # Print every 100 updates
            if updated_count % 100 == 0:
                print(f"Updated {updated_count} paths...")
        else:
            print(f"Warning: File not found at expected location: {new_path}")

# Save the updated JSON back to the file
try:
    with open(split_file_path, 'w') as f:
        json.dump(singer_data, f, indent=2)
    print(f"\nSuccessfully saved updated JSON to {split_file_path}")
    print(f"Updated {updated_count} paths")
except Exception as e:
    print(f"Error saving updated JSON: {e}")
    exit(1)

print("Update complete!")
