# Imports
import pandas as pd
import torch
import numpy as np
import json
import os
import random
import shutil
from collections import defaultdict


def singer_data_json(directory):
    # Initialize a dictionary to store file paths per singer_id
    singer_data = {}
    
    # Read the artist to singer_id mapping
    mapping_file = "/home/aik2/sc-rawnet3/datasets/metamidi/artist_to_singer_id_mapping_merged.csv"
    # Read CSV with no header, names for columns
    mapping_df = pd.read_csv(mapping_file, header=None, names=['artist_name', 'singer_id'])
    
    # Create a dictionary for quick lookup of artist names
    artist_to_singer = dict(zip(mapping_df['singer_id'], mapping_df['artist_name']))
    
    # Iterate over all folders in the directory
    for singer_id in os.listdir(directory):
        singer_dir = os.path.join(directory, singer_id)
        if not os.path.isdir(singer_dir):
            continue
            
        # Get artist name from mapping
        artist_name = artist_to_singer.get(singer_id, "Unknown")
        
        # Initialize the singer's data structure
        singer_data[singer_id] = {
            "artist_name": artist_name,
            "audio_16k": []
        }
        
        # Iterate through subdirectories to find audio files
        for subdir in os.listdir(singer_dir):
            subdir_path = os.path.join(singer_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            # Look for all audio files in the subdirectory
            for file in os.listdir(subdir_path):
                if file.endswith('.wav'):  # Collect all WAV files
                    full_path = os.path.join(subdir_path, file)
                    singer_data[singer_id]["audio_16k"].append(full_path)
    
    # Save the JSON structure to a file
    json_file_path = "singer_data_deduplicated.json"
    with open(json_file_path, 'w') as json_file:
        json.dump(singer_data, json_file, indent=2)
    
    print(f"Number of unique singers: {len(singer_data)}")
    print(f"Singer data saved to {json_file_path}")


def train_test_split(directory, file_name):
    # Read the json file (apparently the one we just created from singer_data_json)
    with open(file_name, 'r') as f:
        singer_data = json.load(f)
    
    # Get unique_artists as a set of all singer_ids in the json file
    unique_artists = set(singer_data.keys())
    
    # Count number of songs (directories) per singer
    singer_song_counts = {}
    for singer_id in unique_artists:
        singer_dir = os.path.join(directory, singer_id)
        if os.path.isdir(singer_dir):
            # Count number of subdirectories (songs)
            num_songs = len([d for d in os.listdir(singer_dir) if os.path.isdir(os.path.join(singer_dir, d))])
            singer_song_counts[singer_id] = num_songs
    
    # From singer_data, make a list of singers with 2-4 songs
    two_to_four_songs_artists_set = set(
        singer_id for singer_id, num_songs in singer_song_counts.items() 
        if 2 <= num_songs <= 4
    )
    
    # Calculate the number of test singers if we want to do a 10% test split
    num_test_singers = int(len(unique_artists) * 0.1)
    
    # Calculate the ratio of singers with 2-4 songs we need to assign to the test set
    ratio_test_singers = num_test_singers / len(two_to_four_songs_artists_set)
    
    # Random seed for reproducibility
    random.seed(42)
    
    # Convert the set to a list before sampling
    two_to_four_songs_artists_list = list(two_to_four_songs_artists_set)
    
    # Randomly assign ratio_test_singers in singers with 2-4 songs to the test set
    test_singers = random.sample(two_to_four_songs_artists_list, num_test_singers)
    
    # Create a new set of test singers
    test_singers_set = set(test_singers)
    
    print(f"Number of test singers: {len(test_singers_set)}")
    print(f"Number of singers with 2-4 songs: {len(two_to_four_songs_artists_set)}")
    
    # Assign the rest of the singers in unique_artists who are not in test_singers_set to the train set
    train_singers_set = unique_artists - test_singers_set
    
    # For each singer_id in test_singers_set, assign is_test=1
    for singer_id in test_singers_set:
        singer_data[singer_id]["is_test"] = 1
    
    # For each singer_id in train_singers_set, assign is_test=0
    for singer_id in train_singers_set:
        singer_data[singer_id]["is_test"] = 0
    
    # Save the updated singer_data to the same json file
    with open(file_name, 'w') as json_file:
        json.dump(singer_data, json_file, indent=2)
    
    print(f"Singer data saved to {file_name}")


def redirect_audio_files(directory, file_name):
    # Read the json file
    with open(file_name, 'r') as f:
        singer_data = json.load(f)
    
    # Get the directory name from the input path
    dir_name = os.path.basename(directory)

    # print the directory name for sanity check
    print(f"Directory name: {dir_name}")
    
    # For each singer in the data
    for singer_id, data in singer_data.items():
        # Initialize the new audio_16k_split list
        data["audio_16k_split"] = []
        
        # Get the is_test value
        is_test = data.get("is_test", 0)
        
        # Determine the split type (test or train)
        split_type = "test" if is_test == 1 else "train"
        
        # Process each audio path
        for audio_path in data["audio_16k"]:
            # Split the path into components
            path_parts = audio_path.split(os.sep)
            
            # Find the index of the directory name
            dir_index = path_parts.index(dir_name)
            
            # Insert the split type (test/train) after the directory name
            new_path_parts = path_parts[:dir_index + 1] + [split_type] + path_parts[dir_index + 1:]
            
            # Join the path parts back together
            new_path = os.sep.join(new_path_parts)
            
            # Add the new path to audio_16k_split
            data["audio_16k_split"].append(new_path)
    
    # Save the updated json file
    with open(file_name, 'w') as json_file:
        json.dump(singer_data, json_file, indent=2)
    
    print(f"Updated audio paths with {split_type} split and saved to {file_name}")


def move_split_files(directory, file_name):
    # Read the json file
    with open(file_name, 'r') as f:
        singer_data = json.load(f)
    
    # Keep track of moved files to avoid duplicates
    moved_files = set()
    
    # For each singer in the data
    for singer_id, data in singer_data.items():
        # Get the original and new paths
        original_paths = data["audio_16k"]
        new_paths = data["audio_16k_split"]
        
        # Move each file from original to new location
        for orig_path, new_path in zip(original_paths, new_paths):
            # Skip if file has already been moved
            if orig_path in moved_files:
                continue
                
            # Create the target directory if it doesn't exist
            target_dir = os.path.dirname(new_path)
            os.makedirs(target_dir, exist_ok=True)
            
            # Move the file
            try:
                shutil.move(orig_path, new_path)
                moved_files.add(orig_path)
                print(f"Moved: {orig_path} -> {new_path}")
            except Exception as e:
                print(f"Error moving {orig_path}: {str(e)}")
    
    print(f"Successfully moved {len(moved_files)} files")


if __name__ == "__main__":
    dedup_dir = '/home/aik2/sc-rawnet3/datasets/metamidi/dedup_singing_sad'

    # singer_data_json(dedup_dir)

    # # After creating the initial JSON file, run the train/test split
    # train_test_split(dedup_dir, "singer_data_deduplicated.json")

    # # Finally, redirect the audio files based on the split
    # redirect_audio_files(dedup_dir, "singer_data_deduplicated.json")

    # Move the files to their new locations
    move_split_files(dedup_dir, "singer_data_deduplicated.json")

