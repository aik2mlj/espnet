import os
import json


def parse_song_info(folder_name):
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
                print(folder_name)
                return None
        return md5
    print(folder_name)
    return None


# Change these paths to match your dataset folder and JSONL file path
dataset_folder = "./deduplicated/"  # Folder containing subfolders named as md5 values
jsonl_file = "./MMD_scraped_genre.jsonl"  # JSONL file containing the md5 field

# Read the JSONL file and collect MD5 values into a set for fast lookup
jsonl_md5_set = set()
with open(jsonl_file, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():  # skip empty lines
            data = json.loads(line)
            md5_val = data.get("md5")
            if md5_val:
                jsonl_md5_set.add(md5_val)

# Get a list of subfolder names in the dataset folder (assuming each subfolder is an md5)
dataset_subfolders = [
    parse_song_info(name)
    for id in os.listdir(dataset_folder)
    for name in os.listdir(os.path.join(dataset_folder, id))
]

total_folders = len(dataset_subfolders)
if total_folders == 0:
    print("No subfolders found in the dataset folder.")
else:
    # Count how many subfolders have a matching md5 in the JSONL file
    matching_folders = [
        folder for folder in dataset_subfolders if folder in jsonl_md5_set
    ]
    num_matching = len(matching_folders)

    # Calculate the percentage
    percentage_covered = (num_matching / total_folders) * 100

    print(f"Total subfolders in dataset: {total_folders}")
    print(f"Subfolders covered by JSONL file: {num_matching}")
    print(f"Coverage: {percentage_covered:.2f}%")
