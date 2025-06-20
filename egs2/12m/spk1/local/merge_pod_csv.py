import os
import glob
import pandas as pd

# Directory containing the CSV files
input_folder = "pod_csv"
output_file = "merged_output.csv"

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through the CSV files and read them
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged {len(csv_files)} files into {output_file}")
