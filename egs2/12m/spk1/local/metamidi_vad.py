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

import os
import argparse
import logging
import torch
import torchaudio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(self):
        # Load the Silero VAD model and utils from torch.hub.
        # Note: force_reload can be set to False if you already have the model cached.
        logger.info("Loading Silero VAD model...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )
        # get_timestamps function is usually the first in the vad_utils list.
        self.get_timestamps = self.vad_utils[0]
        logger.info("Silero VAD model loaded.")

    def _get_voice_segments(self, waveform: torch.Tensor, sample_rate: int = 16000) -> list:
        """
        Extract voice segments using Silero VAD.
        Returns a list of torch.Tensor segments.
        """
        # Get timestamps using the silero VAD utility
        timestamps = self.get_timestamps(waveform, self.vad_model, return_seconds=True)

        segments = []

        # If no voice segments detected, return nothing
        if not timestamps:
            logger.warning("No voice segments detected, return nothing.")
            # segment_length = 8 * sample_rate  # 8 second segments
            # for start in range(0, waveform.shape[1], segment_length):
            #     end = min(start + segment_length, waveform.shape[1])
            #     segments.append(waveform[:, start:end])
            return []

        logger.info(f"Found {len(timestamps)} voice segments, first timestamp: {timestamps[0]}")

        # Process timestamps to merge close segments
        merged_timestamps = []
        current_segment = timestamps[0]

        for i in range(1, len(timestamps)):
            silence_duration = timestamps[i]["start"] - current_segment["end"]
            segment_duration = current_segment["end"] - current_segment["start"]

            # If silence < 3s and current segment < 5s, merge segments
            if silence_duration < 3 and segment_duration < 5:
                current_segment["end"] = timestamps[i]["end"]
            else:
                merged_timestamps.append(current_segment)
                current_segment = timestamps[i]
        merged_timestamps.append(current_segment)

        logger.info(f"Found {len(merged_timestamps)} merged voice segments")
        # Create segments from merged timestamps
        for ts in merged_timestamps:
            start_sample = int(ts["start"] * sample_rate)
            end_sample = int(ts["end"] * sample_rate)
            duration = end_sample - start_sample

            # Keep segments >= 5 seconds, and truncate to 10 seconds if longer.
            if duration >= 5 * sample_rate:
                end_sample = min(end_sample, start_sample + 10 * sample_rate)
                segments.append(waveform[:, start_sample:end_sample])

        # If no valid voice segments found, return nothing
        if not segments:
            logger.warning("No valid voice segments found, using full audio segments")
            # segment_length = 8 * sample_rate  # 8 second segments
            # for start in range(0, waveform.shape[1], segment_length):
            #     end = min(start + segment_length, waveform.shape[1])
            #     segments.append(waveform[:, start:end])
            return []

        return segments


def process_audio_files(input_dir: str, output_dir: str):
    """
    Traverse the input directory, process each WAV file using Silero VAD,
    and save the voice segments to the output directory.
    """
    vad_processor = VoiceActivityDetector()

    # Walk through the input directory recursively
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            input_filepath = os.path.join(root, file)
            # Compute relative path components to rebuild the idxxxxx/<md5-hash> structure.
            rel_path = os.path.relpath(input_filepath, input_dir)
            parts = rel_path.split(os.sep)
            if len(parts) < 3:
                logger.warning(f"Skipping file with unexpected structure: {input_filepath}")
                continue

            # Extract the id and hash parts (assumed to be the first two directories)
            id_dir, hash_dir = parts[0], parts[1]
            out_folder = os.path.join(output_dir, id_dir, hash_dir)
            os.makedirs(out_folder, exist_ok=True)

            logger.info(f"Processing file: {input_filepath}")
            try:
                waveform, sr = torchaudio.load(input_filepath)
            except Exception as e:
                logger.error(f"Failed to load {input_filepath}: {e}")
                continue

            # Check sample rate (should be 16kHz as per instructions)
            if sr != 16000:
                logger.warning(f"Unexpected sample rate {sr} for file {input_filepath}")

            # Get voice segments
            segments = vad_processor._get_voice_segments(waveform, sample_rate=sr)
            logger.info(f"Extracted {len(segments)} segments from {input_filepath}")

            # Save each segment to the output directory
            for idx, segment in enumerate(segments, start=1):
                out_filename = f"{idx:05d}.wav"
                out_filepath = os.path.join(out_folder, out_filename)
                try:
                    torchaudio.save(out_filepath, segment, sample_rate=sr)
                    logger.info(f"Saved segment {idx} to {out_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save segment {idx} for file {input_filepath}: {e}")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    process_audio_files(args.input_dir, args.output_dir)
