# This script assigns a singer ID to each singer in the 12m dataset artist_name column

# imports
import pandas as pd
import numpy as np
import json
import os

print("Starting singer ID assignment process...")

# Load the filtered dataset with local file mappings
print("Loading filtered dataset...")
filtered_df_path = '/home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav-with-local.csv'
df = pd.read_csv(filtered_df_path)
print(f"Loaded filtered dataset with {len(df)} rows")



def filter_and_assign_singer_ids(df):
    # All filter patterns in one list - patterns we want to exclude
    filters = [
        'feat\\.', 
        'Orchestra|Philharmonic|Symphonica|Chamber|Ensemble|Piano|string|strings|Symphony|Symphonie|symphon|Concerto|choir|chorus|Philharmoniker',
        '交响|乐团|协奏曲|合唱',
        'DJ |D\\.J\\.',  # Matching both "DJ " and "D.J."
        'vs\\.',
        '\' \'',
        '&',
        'unknown|anonymous|unknown artist',
        'with', 'collection', 'collective', ','
    ]
    
    # Combine all patterns with OR operator
    combined_pattern = '|'.join(filters)
    
    # Create masks for both pattern matches and NaN values
    pattern_mask = df['artist_name'].str.contains(combined_pattern, case=False, na=False)
    nan_mask = df['artist_name'].isna()
    
    # Count NaN values
    nan_count = nan_mask.sum()
    nan_percentage = (nan_count / len(df)) * 100
    
    # Combine both masks
    filter_mask = pattern_mask | nan_mask
    
    # Count unique artist names matching any filter (excluding NaN)
    filtered_artists = df[pattern_mask]['artist_name'].unique()
    filtered_artist_count = len(filtered_artists)
    
    # Count songs (rows) with filtered artists or NaN values
    filtered_song_count = filter_mask.sum()
    
    # Calculate percentages
    total_artists = df['artist_name'].nunique()
    total_songs = len(df)
    artist_percentage = (filtered_artist_count / total_artists) * 100
    song_percentage = (filtered_song_count / total_songs) * 100
    
    # Print results
    print(f"Total unique artist names in filtered dataset: {total_artists}")
    print(f"NaN artist names: {nan_count} ({nan_percentage:.2f}% of total songs)")
    print(f"Artist names matching at least one filter: {filtered_artist_count} ({artist_percentage:.2f}%)")
    print(f"\nTotal songs in filtered dataset: {total_songs}")
    print(f"Songs with filtered artists or NaN: {filtered_song_count} ({song_percentage:.2f}%)")
    
    # Get all unique artists
    all_artists = df['artist_name'].unique()
    
    # Convert filtered_artists to a set for faster lookup
    filtered_artists_set = set(filtered_artists)
    
    # Get artists that are NOT in filtered_artists and are not NaN
    unfiltered_artists = [artist for artist in all_artists 
                         if artist not in filtered_artists_set and not pd.isna(artist)]
    
    print(f"Number of unique unfiltered artists (case sensitive): {len(set(unfiltered_artists))}")
    
    # Process artists case-insensitively
    # Create a dictionary mapping lowercase artist names to their original forms
    case_mapping = {}
    for artist in unfiltered_artists:
        if isinstance(artist, str):
            lowercase = artist.lower()
            if lowercase in case_mapping:
                case_mapping[lowercase].append(artist)
            else:
                case_mapping[lowercase] = [artist]
    
    print(f"Number of unique unfiltered artists (case insensitive): {len(case_mapping)}")
    
    # Assign singer IDs to each unique lowercase artist name
    singer_id_mapping = {}
    id_counter = 1
    
    for lowercase_name in sorted(case_mapping.keys()):
        # Create 5-digit ID with padding
        singer_id = f"id{id_counter:05d}"
        # Store the mapping with original variations
        singer_id_mapping[singer_id] = {
            "lowercase": lowercase_name,
            "variations": case_mapping[lowercase_name]
        }
        id_counter += 1
    
    print(f"Created mapping for {len(singer_id_mapping)} unique singers")
    
    return singer_id_mapping

# Run the filtering and ID assignment
print("Filtering artists and assigning IDs...")
singer_id_mapping = filter_and_assign_singer_ids(df)

# Save the singer ID mapping to a file
output_file = '/home/aik2/sc-rawnet3/datasets/12m/singer_id_mapping_filtered.json'
print(f"Saving singer ID mapping to {output_file}")
with open(output_file, 'w') as f:
    json.dump(singer_id_mapping, f, indent=2)

# Create a reverse mapping (from artist name to ID) for easier lookup
reverse_mapping = {}
for singer_id, info in singer_id_mapping.items():
    lowercase = info["lowercase"]
    reverse_mapping[lowercase] = singer_id
    for variation in info["variations"]:
        reverse_mapping[variation] = singer_id

# Add singer_id column to the dataframe
print("Adding singer_id column to the dataframe...")
def get_singer_id(artist_name):
    if pd.isna(artist_name):
        return None
    
    # Check direct mapping first (case-sensitive)
    if artist_name in reverse_mapping:
        return reverse_mapping[artist_name]
    
    # Try lowercase version
    lowercase = artist_name.lower()
    if lowercase in reverse_mapping:
        return reverse_mapping[lowercase]
    
    # If not found, it's a filtered artist
    return None

df['singer_id'] = df['artist_name'].apply(get_singer_id)

# Save the updated dataframe
output_csv = '/home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav-with-local-and-singer-id.csv'
print(f"Saving updated dataframe to {output_csv}")
df.to_csv(output_csv, index=False)

# Print some statistics
assigned_count = df['singer_id'].notna().sum()
print(f"\nAssigned singer IDs to {assigned_count} rows ({assigned_count/len(df)*100:.2f}% of the filtered dataset)")
print(f"Saved reverse mapping for {len(reverse_mapping)} artist name variations")

print("\nProcess completed successfully!")

# Add sample code showing how to use the mapping
print("\nExample of using the mapping:")
print("import json")
print("with open('/home/aik2/sc-rawnet3/datasets/12m/singer_id_mapping_filtered.json', 'r') as f:")
print("    singer_mapping = json.load(f)")
print("# Get artist variations for singer id00001:")
print("print(singer_mapping['id00001'])")

