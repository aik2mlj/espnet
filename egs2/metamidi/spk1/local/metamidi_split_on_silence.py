#!/usr/bin/env python3
"""
This script processes a directory of WAV files organized as:
    idxxxxx/<md5-hash>/00001.wav
It uses Silero VAD to detect voice activity and writes the detected voice segments
to an output directory preserving the hierarchy:
    idxxxxx/<md5-hash>/*.wav

Each output file corresponds to one detected voice segment.
The input audios are already resampled to 16kHz.
"""

import argparse
import logging
import multiprocessing
import os
from functools import partial

from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("metamidi_split_on_silence.log"),
        # Uncomment below if you want to keep console logging too
        # logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


def process_single_file(input_filepath, input_dir, output_dir):
    """Process a single audio file with VAD and save segments."""
    if not input_filepath.lower().endswith(".wav"):
        return

    # Compute relative path components
    rel_path = os.path.relpath(input_filepath, input_dir)
    parts = rel_path.split(os.sep)
    if len(parts) < 3:
        logger.warning(f"Skipping file with unexpected structure: {input_filepath}")
        return

    # Create output directory
    id_dir, hash_dir = parts[0], parts[1]
    out_folder = os.path.join(output_dir, id_dir, hash_dir)
    os.makedirs(out_folder, exist_ok=True)

    logger.info(f"Processing file: {input_filepath}")
    try:
        audio = AudioSegment.from_wav(input_filepath)
        segments = split_on_silence(
            audio,
            min_silence_len=2000,
            silence_thresh=-40,
            keep_silence=100,
        )

        # Filter segments shorter than 3 seconds (3000 ms)
        valid_segments = [seg for seg in segments if len(seg) >= 3000]
        logger.info(f"Extracted {len(valid_segments)} valid segments (>=3s) from {input_filepath}")

        for idx, segment in enumerate(valid_segments, start=1):
            out_filename = f"{idx:05d}.wav"
            out_filepath = os.path.join(out_folder, out_filename)
            segment.export(out_filepath, format="wav")
    except Exception as e:
        logger.error(f"Error processing {input_filepath}: {e}")


def process_audio_files(input_dir: str, output_dir: str, num_threads):
    """Process audio files using multithreading."""
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_paths.append(os.path.join(root, file))

    # Process files in parallel
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     process_func = partial(process_single_file, input_dir=input_dir, output_dir=output_dir)
    #     # Using list() to force evaluation and catch exceptions immediately
    #     list(
    #         executor.map(process_func, file_paths, chunksize=10)
    #     )  # chunksize for better load balancing

    # Create a partial function with fixed input_dir and output_dir
    worker_func = partial(process_single_file, input_dir=input_dir, output_dir=output_dir)

    # Determine the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Using {num_cores} CPU cores for processing.")

    # Create pool and process with tqdm progress bar
    with multiprocessing.Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap(worker_func, file_paths),
                total=len(file_paths),
                desc="Processing files",
                unit="file",
            )
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detect voice segments using Silero VAD and write segments to disk."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to input directory with audio files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for segmented audio.",
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="Number of threads to use for processing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    process_audio_files(args.input_dir, args.output_dir, num_threads=int(args.num_threads))
