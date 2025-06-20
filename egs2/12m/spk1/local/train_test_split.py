# this script is used to split the 12m dataset into train and test sets
# reference dataframe: /home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav-with-local-and-singer-id.csv

# imports
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

print("Loading dataframe...")
# load dataframe
df = pd.read_csv('/home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav-with-local-and-singer-id.csv')

# Get singer counts (reusing the logic from your notebook)
singer_id_counts = df.groupby('singer_id').agg({
    'artist_name': ['count', lambda x: x.str.lower().unique()]
}).reset_index()

# Flatten column names
singer_id_counts.columns = ['singer_id', 'song_count', 'artist_names']

print(f"Total singer IDs: {len(singer_id_counts)}")

# get a count of total number of singer_ids
total_singers = len(singer_id_counts)

# calculate the number we need to sample for test set, which is 10% of total number of singers
num_test_singers = int(total_singers * 0.1)
print(f"Target test singers: {num_test_singers}")

# randomly sample num_test_singers from singers with 2-5 songs
singers_2_5 = singer_id_counts[singer_id_counts['song_count'].between(2, 5)]
print(f"Singers with 2-5 songs: {len(singers_2_5)}")

if len(singers_2_5) < num_test_singers:
    print(f"Warning: Not enough singers with 2-5 songs ({len(singers_2_5)}) for test set ({num_test_singers})")
    test_singers = singers_2_5['singer_id'].tolist()
else:
    test_singers = singers_2_5['singer_id'].sample(n=num_test_singers, random_state=42).tolist()

print(f"Selected {len(test_singers)} singers for test set")

# from the remaining singers that are not in the test set, randomly sample for exp set
remaining_singers = singer_id_counts[~singer_id_counts['singer_id'].isin(test_singers)]

# Define the ranges and sample 10 from each
exp_singers = []
ranges = [
    (1, 1, "1 song"),
    (2, 5, "2-5 songs"), 
    (5, 10, "5-10 songs"),
    (10, 30, "10-30 songs"),
    (30, 100, "30-100 songs"),
    (100, float('inf'), "100+ songs")
]

for min_songs, max_songs, description in ranges:
    if max_songs == float('inf'):
        range_singers = remaining_singers[remaining_singers['song_count'] >= min_songs]
    else:
        range_singers = remaining_singers[remaining_singers['song_count'].between(min_songs, max_songs)]
    
    print(f"Singers with {description}: {len(range_singers)}")
    
    if len(range_singers) >= 10:
        sampled = range_singers['singer_id'].sample(n=10, random_state=42).tolist()
    else:
        sampled = range_singers['singer_id'].tolist()
        print(f"  Warning: Only {len(sampled)} singers available, taking all")
    
    exp_singers.extend(sampled)
    print(f"  Selected {len(sampled)} singers for exp set")

print(f"Total exp singers: {len(exp_singers)}")

# Create a mapping from singer_id to split for quick lookup
singer_split_map = {}
for singer_id in test_singers:
    singer_split_map[singer_id] = 'test'
for singer_id in exp_singers:
    singer_split_map[singer_id] = 'exp'
# All others default to 'train'

# Add split column to dataframe
def assign_split(singer_id):
    return singer_split_map.get(singer_id, 'train')

df['split'] = df['singer_id'].apply(assign_split)

# Convert to numeric for consistency with your original code
df['split'] = df['split'].map({'train': 0, 'test': 1, 'exp': 2})

# Print statistics
train_songs = (df['split'] == 0).sum()
test_songs = (df['split'] == 1).sum() 
exp_songs = (df['split'] == 2).sum()

print(f"\nDataset split statistics:")
print(f"Train songs: {train_songs}")
print(f"Test songs: {test_songs}")
print(f"Exp songs: {exp_songs}")
print(f"Total songs: {len(df)}")

# save the dataframe to a new csv file
output_file = '/home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav-id-split.csv'
df.to_csv(output_file, index=False)
print(f"Saved dataframe with split info to: {output_file}")

# now inside the sad_id folder, move each folder to the train, test, or exp folder
sad_id_path = Path('/home/aik2/sc-rawnet3/datasets/12m/sad_id')
train_path = sad_id_path / 'train'
test_path = sad_id_path / 'test' 
exp_path = sad_id_path / 'exp'

# Create directories if they don't exist
train_path.mkdir(exist_ok=True)
test_path.mkdir(exist_ok=True)
exp_path.mkdir(exist_ok=True)

print(f"\nMoving folders from {sad_id_path}...")

# Get all folders in sad_id directory (these should be singer_ids)
if sad_id_path.exists():
    folders = [f for f in sad_id_path.iterdir() if f.is_dir() and f.name not in ['train', 'test', 'exp']]
    total_folders = len(folders)
    print(f"Found {total_folders} folders to move")
    
    moved_counts = {'train': 0, 'test': 0, 'exp': 0}
    progress_count = 0
    
    for i, folder in enumerate(folders, 1):
        singer_id = folder.name  # folder name should be the singer_id (e.g., id09707)
        
        # Look up split for this singer_id
        split_name = singer_split_map.get(singer_id, 'train')  # default to train
        
        # Determine destination
        if split_name == 'train':
            destination = train_path / singer_id
            moved_counts['train'] += 1
        elif split_name == 'test':
            destination = test_path / singer_id
            moved_counts['test'] += 1
        elif split_name == 'exp':
            destination = exp_path / singer_id
            moved_counts['exp'] += 1
        
        # Move the folder
        try:
            shutil.move(str(folder), str(destination))
            progress_count += 1
        except Exception as e:
            print(f"Error moving {singer_id}: {e}")
            continue
        
        # Print progress every 1000 folders
        if progress_count % 1000 == 0:
            print(f"Progress: {progress_count}/{total_folders} folders moved "
                  f"({progress_count/total_folders*100:.1f}%) | "
                  f"Train: {moved_counts['train']}, Test: {moved_counts['test']}, Exp: {moved_counts['exp']}")
    
    print(f"\nFolder movement completed:")
    print(f"Train folders: {moved_counts['train']}")
    print(f"Test folders: {moved_counts['test']}")
    print(f"Exp folders: {moved_counts['exp']}")
    print(f"Total moved: {sum(moved_counts.values())}")
    
else:
    print(f"Error: sad_id directory not found at {sad_id_path}")

print("\nSplit completed successfully!")