import os
import shutil
import pandas as pd
from tqdm import tqdm

# ---------------------------
# Step 1. Read the mapping CSV and build merged mapping
# ---------------------------
# The original CSV is assumed to have no header and two columns:
# artist, singer_id
mapping_file = "artist_to_singer_id_mapping.csv"
mapping_df = pd.read_csv(
    mapping_file,
    header=None,
    names=["artist", "singer_id"],
    dtype=str,
    keep_default_na=False,
)

# Build dictionary: normalized artist (lowercase) -> details (list of singer_ids, chosen rep, original artist name)
merged_mapping = {}
for idx, row in mapping_df.iterrows():
    artist = row["artist"]
    singer_id = row["singer_id"]
    # print(artist)
    norm_artist = artist.lower()
    if norm_artist not in merged_mapping:
        merged_mapping[norm_artist] = {
            "singer_ids": [],
            "representative": singer_id,  # choose the first encountered singer_id as representative
            "artist": artist,  # preserve the case from the first occurrence
        }
    merged_mapping[norm_artist]["singer_ids"].append(singer_id)

# Compute deduplication statistics
original_count = len(mapping_df)
merged_count = len(merged_mapping)
dedup_percentage = (original_count - merged_count) / original_count * 100

# Write the merged mapping CSV: one line per unique artist
merged_mapping_rows = []
for norm_artist, data in merged_mapping.items():
    merged_mapping_rows.append([data["artist"], data["representative"]])
merged_mapping_df = pd.DataFrame(merged_mapping_rows, columns=["artist", "singer_id"])
merged_mapping_df.to_csv(
    "artist_to_singer_id_mapping_merged.csv", index=False, header=False
)

# Print deduplication summary
print(f"Original number of singer entries: {original_count}")
print(f"Merged (unique) singer entries: {merged_count}")
print(f"Percentage deduplicated: {dedup_percentage:.2f}%")


# ---------------------------
# Step 2. Define a helper function to parse song folder names
# ---------------------------
def parse_song_info(folder_name):
    """
    Expected folder format:
       tmp3gu_pld_a-10c3eeld07162b0bd5217456113dba2be___All_Of_My_Heart-Abc
    This function splits on "___" then uses the last part (e.g., "All_Of_My_Heart-Abc")
    to extract the song name and the artist name by splitting on the last hyphen.
    """
    if "___" in folder_name:
        parts = folder_name.split("___")
        # Use the last part
        song_artist_part = parts[-1]
        if "-" in song_artist_part:
            # Split on the last hyphen so that song names with hyphens are handled correctly
            song_name, artist_name = song_artist_part.rsplit("-", 1)
            return song_name, artist_name
    return None, None


# ---------------------------
# Step 3. Process the audio directories
# ---------------------------
# The dataset is under "audio-16k" with "train" and "test" subfolders.
input_dirs = [os.path.join("audio_16k", "train"), os.path.join("audio_16k", "test")]
output_dir = "dedup_singing"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# To record which songs (by lower-case song name) have been copied per merged singer ID
singer_songs = {}

# For writing merged_singers.csv, we keep rows: singer_id, artist, song_name, source_path, dest_path.
merged_singers = []

# Iterate over both train and test folders (ignoring the split in the merged output)
for input_dir in input_dirs:
    if not os.path.isdir(input_dir):
        continue  # Skip if directory doesn't exist
    for singer_dir in tqdm(os.listdir(input_dir)):
        singer_path = os.path.join(input_dir, singer_dir)
        if not os.path.isdir(singer_path):
            continue

        # Look up the artist name from the mapping CSV using the singer id
        row = mapping_df[mapping_df["singer_id"] == singer_dir]
        if row.empty:
            # Skip if singer id is not in the mapping file
            continue
        artist_name = row.iloc[0]["artist"]
        norm_artist = artist_name.lower()
        merged_singer_id = merged_mapping[norm_artist]["representative"]

        # Create the destination singer directory if it doesn't exist
        dest_singer_dir = os.path.join(output_dir, merged_singer_id)
        if not os.path.exists(dest_singer_dir):
            os.makedirs(dest_singer_dir)
            singer_songs[merged_singer_id] = set()

        # Process each song folder in the current singer directory
        for song_folder in os.listdir(singer_path):
            song_folder_path = os.path.join(singer_path, song_folder)
            if not os.path.isdir(song_folder_path):
                continue

            song_name, song_artist = parse_song_info(song_folder)
            if song_name is None:
                continue

            norm_song_name = song_name.lower()
            # If a song with the same name (case-insensitive) already exists for this merged singer, skip it
            audio_file_path = os.path.join(song_folder_path, "00001.wav")
            if (
                norm_song_name in singer_songs[merged_singer_id]
                and os.path.exists(audio_file_path)
                and os.path.getsize(audio_file_path) > 1000
            ):
                continue

            # Copy the entire song folder into the merged directory under the representative singer id
            dest_song_folder = os.path.join(dest_singer_dir, song_folder)
            try:
                shutil.copytree(song_folder_path, dest_song_folder)
                singer_songs[merged_singer_id].add(norm_song_name)

                # Record the copied song information
                merged_singers.append(
                    [
                        merged_singer_id,
                        artist_name,
                        song_name,
                        song_folder_path,
                        dest_song_folder,
                    ]
                )
            except Exception as e:
                print(f"Error copying {song_folder_path}: {str(e)}")
                continue

# ---------------------------
# Step 4. Write the merged singers CSV and print statistics
# ---------------------------
merged_singers_df = pd.DataFrame(
    merged_singers,
    columns=["singer_id", "artist", "song_name", "source_path", "dest_path"],
)
merged_singers_df.to_csv("merged_singers.csv", index=False)

# Print final statistics
total_copied = len(merged_singers)
total_singers = len(singer_songs)
total_songs = sum(len(songs) for songs in singer_songs.values())
print(f"\nFinal Statistics:")
print(f"Total singers processed: {total_singers}")
print(f"Total songs copied: {total_copied}")
print(f"Average songs per singer: {total_songs/total_singers:.2f}")
