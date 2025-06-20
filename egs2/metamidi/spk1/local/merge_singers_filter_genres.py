import os
import shutil
import json
import pandas as pd
from tqdm import tqdm
from collections import Counter

# ---------------------------
# Step 1. Define highlighted genres to filter out
# ---------------------------
highlighted_genres = [
    "ancient",
    "romantic",
    "modern",
    "baroque",
    "renaissance",
    "non-western_classical",
    "classical music",
    "early 20th century",
    "early_20th_century",
    "classical",
    "medieval",
    "piano",
    "instrumental",
    "instrumentals",
    "composer",
    "waltz",
    "big-band",
    "swing",
    "national anthems",
    "ballroom and standards",
    "opera",
]

# ---------------------------
# Step 2. Extract MD5 hashes associated with highlighted genres
# ---------------------------
print("Building list of MD5 hashes for highlighted genres...")
jsonl_file = "./MMD_scraped_genre.jsonl"
highlighted_md5_hashes = set()


# Function to extract MD5 from folder name
def extract_md5(folder_name):
    """
    Expected folder format:
       tmp3gu_pld_a-10c3eeld07162b0bd5217456113dba2be___All_Of_My_Heart-Abc
    """
    if "___" in folder_name:
        parts = folder_name.split("___")
        md5 = parts[-2].split("-")[-1]
        if len(md5) < 10:
            md5 = parts[-3].split("-")[-1]
            if len(md5) < 10:
                return None
        return md5
    return None


# Open and process the JSONL file line by line to find MD5s related to highlighted genres
with open(jsonl_file, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():  # skip empty lines
            data = json.loads(line)
            md5_val = data.get("md5")

            # Check if this MD5 has any of the highlighted genres
            for genre_list in data.get("genre", []):
                for genre in genre_list:
                    if genre in highlighted_genres and md5_val:
                        highlighted_md5_hashes.add(md5_val)
                        break

print(
    f"Found {len(highlighted_md5_hashes)} MD5 hashes associated with highlighted genres"
)

# ---------------------------
# Step 3. Read the mapping CSV and build merged mapping
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
# Step 4. Define a helper function to parse song folder names
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
# Step 5. Process the audio directories
# ---------------------------
# The dataset is under "audio_16k" with "train" and "test" subfolders.
input_dirs = [os.path.join("audio_16k", "train"), os.path.join("audio_16k", "test")]
output_dir = "dedup_singing"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# To record which songs (by lower-case song name) have been copied per merged singer ID
singer_songs = {}

# For writing merged_singers.csv, we keep rows: singer_id, artist, song_name, source_path, dest_path.
merged_singers = []

# Track statistics
skipped_genre_count = 0
skipped_by_genre = Counter()
total_processed = 0

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
        # Initialize the set for this singer if it doesn't exist
        if merged_singer_id not in singer_songs:
            singer_songs[merged_singer_id] = set()

        # Process each song folder in the current singer directory
        for song_folder in os.listdir(singer_path):
            total_processed += 1
            song_folder_path = os.path.join(singer_path, song_folder)
            if not os.path.isdir(song_folder_path):
                continue

            # Check if the song folder contains an MD5 associated with highlighted genres
            md5 = extract_md5(song_folder)
            if md5 in highlighted_md5_hashes:
                skipped_genre_count += 1
                skipped_by_genre["highlighted_genre"] += 1
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
                and os.path.getsize(audio_file_path) > 1024 * 1024  # 1MB threshold
            ):
                skipped_by_genre["duplicate"] += 1
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
            except FileExistsError:
                skipped_by_genre["file_exists"] += 1
                continue
            except Exception as e:
                print(f"Error copying {song_folder_path}: {str(e)}")
                skipped_by_genre["error"] += 1
                continue

# ---------------------------
# Step 6. Write the merged singers CSV and print statistics
# ---------------------------
merged_singers_df = pd.DataFrame(
    merged_singers,
    columns=["singer_id", "artist", "song_name", "source_path", "dest_path"],
)
merged_singers_df.to_csv("merged_singers_filtered.csv", index=False)

# remove all empty singer_id directories in dedup_singing
print("\nRemoving empty singer directories...")
empty_dirs = []
for singer_dir in os.listdir(output_dir):
    singer_path = os.path.join(output_dir, singer_dir)
    if os.path.isdir(singer_path) and not os.listdir(singer_path):
        empty_dirs.append(singer_path)
        os.rmdir(singer_path)
print(f"Removed {len(empty_dirs)} empty singer directories")


# Print final statistics
total_copied = len(merged_singers)
total_singers = len(singer_songs)
total_songs = sum(len(songs) for songs in singer_songs.values())

print(f"\nFinal Statistics:")
print(f"Total files processed: {total_processed}")
print(f"Total singers processed: {total_singers}")
print(f"Total songs copied: {total_copied}")
print(f"Songs skipped due to highlighted genres: {skipped_genre_count}")
print(f"Breakdown of skipped files: {dict(skipped_by_genre)}")
print(
    f"Average songs per singer: {total_songs / total_singers:.2f if total_singers > 0 else 0}"
)
