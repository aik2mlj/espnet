import numpy as np
import pandas as pd
import os
import re
import random

# load file /home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav.csv and open
df = pd.read_csv('/home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav.csv')

# print headers of each column
print(df.columns.tolist())


print("Extracting song names from CSV...")
# Create only Set 1: Basenames with .wav removed
csv_names_no_wav = {}  # Set 1: Basenames with .wav removed -> original gcs_link

# Track the index of each filename to map back to the dataframe
for index, row in df.iterrows():
    try:
        link = row['gcs_link']
        filename = os.path.basename(link)
        
        # Set 1: Remove .wav extension
        if filename.endswith('.wav'):
            csv_names_no_wav[filename[:-4]] = index
    except:
        continue

# Get all items in ./sad directory (only depth 1, no subdirectories)
print("Reading items from ./sad directory...")
sad_items = [item for item in os.listdir('./sad') if item != '.DS_Store']

# Create a dictionary to map df indices to their local_file_name in ./sad
local_file_mapping = {}
# Track duplicate mappings (when multiple folder names match to the same gcs_link)
duplicate_mappings = []
# Store folders that need to find alternative matches
folders_needing_rematch = []

# Helper function to handle duplicates
def handle_duplicate(index, existing_name, new_name):
    if len(new_name) > len(existing_name):
        # New name is longer, prefer it
        print(f"Duplicate mapping: Preferring longer name '{new_name}' over '{existing_name}' for index {index}")
        folders_needing_rematch.append(existing_name)
        return new_name
    else:
        # Existing name is longer or equal, keep it
        print(f"Duplicate mapping: Keeping longer name '{existing_name}' over '{new_name}' for index {index}")
        folders_needing_rematch.append(new_name)
        return existing_name

# First pass: check exact matches only with Set 1
files_in_csv = []
potentially_not_found = []

for item in sad_items:
    found = False
    
    # Check Set 1 (no_wav)
    if item in csv_names_no_wav:
        files_in_csv.append(item)
        index = csv_names_no_wav[item]
        if index in local_file_mapping:
            local_file_mapping[index] = handle_duplicate(index, local_file_mapping[index], item)
        else:
            local_file_mapping[index] = item
        found = True
    
    if not found:
        potentially_not_found.append(item)

print(f"After first pass checks (no_wav only):")
print(f"  - Found in CSV: {len(files_in_csv)}")
print(f"  - Potentially not found: {len(potentially_not_found)}")
print(f"  - Folders needing rematch: {len(folders_needing_rematch)}")

# Print some examples of potentially not found items
if potentially_not_found:
    print("\nSample of potentially not found items:")
    sample_size = min(10, len(potentially_not_found))
    for item in random.sample(potentially_not_found, sample_size):
        print(f"  - {item}")

# Print some examples of found items
if files_in_csv:
    print("\nSample of found items:")
    sample_size = min(10, len(files_in_csv))
    for item in random.sample(files_in_csv, sample_size):
        print(f"  - {item}")

# Print total number of matches
print(f"\nTotal unique mappings: {len(local_file_mapping)}")

# Also note that, some files in the original dataframe may not be found in the ./sad directoyr. If they are not found, do not save them to the new dataframe. 

# Add local_file_name column to the dataframe
print("Adding local_file_name column to dataframe...")
df['local_file_name'] = None

# Populate the local_file_name column using the mapping
for index, folder_name in local_file_mapping.items():
    df.at[index, 'local_file_name'] = folder_name

# Filter the dataframe to only include rows with local_file_name
filtered_df = df[df['local_file_name'].notna()].copy()
print(f"Original dataframe size: {len(df)} rows")
print(f"Filtered dataframe size: {len(filtered_df)} rows")
print(f"Removed {len(df) - len(filtered_df)} rows without local file matches")

# Save the filtered dataframe to a new CSV file
output_path = '/home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav-with-local.csv'
print(f"Saving filtered dataframe to {output_path}...")
filtered_df.to_csv(output_path, index=False)
print("Done!")

# Print summary of the updated dataframe
print(f"Total records in saved dataframe: {len(filtered_df)}")
