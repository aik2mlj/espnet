import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detect voice segments using Silero VAD and write segments to disk."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to input directory with audio files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dir = args.dir

    # remove all empty singer_id directories in dedup_singing
    print("\nRemoving empty singer/song directories...")
    empty_singers = []
    empty_songs = []
    for singer_dir in os.listdir(dir):
        singer_path = os.path.join(dir, singer_dir)
        for song_dir in os.listdir(singer_path):
            song_path = os.path.join(singer_path, song_dir)
            if os.path.isdir(song_path) and not os.listdir(song_path):
                empty_songs.append(song_path)
                os.rmdir(song_path)
        if os.path.isdir(singer_path) and not os.listdir(singer_path):
            empty_singers.append(singer_path)
            os.rmdir(singer_path)
    print(f"Removed {len(empty_singers)} empty singer directories")
    print(f"Removed {len(empty_songs)} empty song directories")
