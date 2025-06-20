import json
import collections
import matplotlib.pyplot as plt

# Change this to your JSONL file path
filename = "./MMD_scraped_genre.jsonl"

# Counter to keep track of genre counts
genre_counter = collections.Counter()

# Open and process the JSONL file line by line
with open(filename, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():  # skip empty lines
            data = json.loads(line)
            # Iterate through each inner list in the 'genre' field
            for genre_list in data.get("genre", []):
                for genre in genre_list:
                    genre_counter[genre] += 1

# Print the counted genres
print("Genre distribution:")
for genre, count in genre_counter.items():
    print(f"{genre}: {count}")

# Print all genres in an array
all_genres = sorted(list(genre_counter.keys()))
print("\nAll genres array:")
print(all_genres)

# Define the genres related to classical music and non-singing music
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

# Calculate and print counts
total_songs = sum(genre_counter.values())
highlighted_count = sum(
    count for genre, count in genre_counter.items() if genre in highlighted_genres
)
print(f"\nTotal songs in all genres: {total_songs}")
print(f"Total songs in highlighted genres: {highlighted_count}")

# Plot the genre distribution with highlighted genres in a different color
plt.figure(figsize=(12, 6))

# Iterate over the genres to plot each bar with an appropriate color
for genre, count in genre_counter.items():
    # Use 'red' for highlighted genres and 'skyblue' for others
    color = "red" if genre in highlighted_genres else "skyblue"
    plt.bar(genre, count, color=color)

plt.xlabel("Genre")
plt.ylabel("Count")
plt.title("Distribution of Genres (Classical & Non-Singing Highlighted)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
