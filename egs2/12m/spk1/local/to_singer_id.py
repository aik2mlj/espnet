# use datasets/12m/12m-youtube-wav-with-local-and-singer-id.csv and assign each song in this dataframe to a singer_id. 

# TODO:
# Copy all content in the corresponding folder name from the local_file_name column to another one under singer_id/song_name/. singer_id is a string in the form: idXXXXX under the column singer_id.
# For example, if the folder name under local_file_name is songName, then copy all content from ./sad/songName/ to ./sad_id/singer_id/songName/
# Make sure to create the singer_id folder if it doesn't exist.
# Make sure to copy all content, including subfolders and files.


import pandas as pd
import os
import shutil
from pathlib import Path
import glob
import filecmp
import hashlib

# Load the dataset
print("Loading dataset...")
csv_path = '/home/aik2/sc-rawnet3/datasets/12m/12m-youtube-wav-with-local-and-singer-id.csv'
df = pd.read_csv(csv_path)

# Define source and destination base paths
source_base = './sad'
dest_base = './sad_id'

# Create destination base directory if it doesn't exist
os.makedirs(dest_base, exist_ok=True)

# Count for tracking progress
total_rows = len(df)
processed = 0
skipped = 0
errors = 0
already_exists = 0

print(f"Processing {total_rows} songs...")

# Function to check if directories have identical content
def dirs_are_identical(dir1, dir2):
    """
    Check if two directories have identical content
    """
    if not os.path.exists(dir2):
        return False
        
    comparison = filecmp.dircmp(dir1, dir2)
    
    # Check if there are any differences in files
    if comparison.left_only or comparison.right_only or comparison.diff_files:
        return False
        
    # Recursively check subdirectories
    for subdir in comparison.common_dirs:
        subdir1 = os.path.join(dir1, subdir)
        subdir2 = os.path.join(dir2, subdir)
        if not dirs_are_identical(subdir1, subdir2):
            return False
            
    return True

# Function to copy all files from source to destination
def copy_folder_contents(src_folder, dest_folder):
    try:
        # Create destination folder if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)
        
        # First check if all content already exists
        if os.path.exists(dest_folder) and dirs_are_identical(src_folder, dest_folder):
            return 'exists'
        
        # Use shutil.copytree with dirs_exist_ok for Python 3.8+
        # For older Python versions, we'd need a more manual approach
        try:
            # This works in Python 3.8+
            shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
        except TypeError:
            # For older Python versions
            for item in os.listdir(src_folder):
                src_item = os.path.join(src_folder, item)
                dest_item = os.path.join(dest_folder, item)
                
                if os.path.isdir(src_item):
                    if not os.path.exists(dest_item):
                        shutil.copytree(src_item, dest_item)
                    else:
                        # Recursively copy contents of subdirectories
                        for sub_item in os.listdir(src_item):
                            src_sub_item = os.path.join(src_item, sub_item)
                            dest_sub_item = os.path.join(dest_item, sub_item)
                            
                            if os.path.isdir(src_sub_item):
                                if not os.path.exists(dest_sub_item):
                                    shutil.copytree(src_sub_item, dest_sub_item)
                            else:
                                shutil.copy2(src_sub_item, dest_sub_item)
                else:
                    shutil.copy2(src_item, dest_item)
                    
        return True
    except Exception as e:
        print(f"Error copying {src_folder}: {str(e)}")
        return False

# Process each row in the DataFrame
for index, row in df.iterrows():
    # Update progress counter
    processed += 1
    
    # Show progress every 1000 songs
    if processed % 1000 == 0:
        print(f"Progress: {processed}/{total_rows} ({processed/total_rows*100:.1f}%) | Skipped: {skipped} | Already exists: {already_exists}")
    
    # Skip if singer_id is missing
    if pd.isna(row['singer_id']):
        skipped += 1
        continue
        
    # Skip if local_file_name is missing
    if pd.isna(row['local_file_name']):
        skipped += 1
        continue
    
    # Get singer_id and local_file_name
    singer_id = row['singer_id']
    folder_name = row['local_file_name']
    
    # Define source and destination paths
    src_path = os.path.join(source_base, folder_name)
    dest_path = os.path.join(dest_base, singer_id, folder_name)
    
    # Check if source exists
    if not os.path.exists(src_path):
        print(f"Warning: Source folder not found: {src_path}")
        skipped += 1
        continue
    
    # Create singer_id directory if it doesn't exist
    singer_dir = os.path.join(dest_base, singer_id)
    os.makedirs(singer_dir, exist_ok=True)
    
    # Copy the contents
    result = copy_folder_contents(src_path, dest_path)
    if result == 'exists':
        already_exists += 1
    elif not result:
        errors += 1

# Print summary
print("\nCopy operation complete!")
print(f"Total songs processed: {processed}")
print(f"Successfully copied: {processed - skipped - errors - already_exists}")
print(f"Already existed (skipped copying): {already_exists}")
print(f"Skipped (missing singer_id or folder): {skipped}")
print(f"Errors during copy: {errors}")