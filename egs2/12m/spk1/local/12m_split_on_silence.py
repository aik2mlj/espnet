#!/usr/bin/env python3
import argparse
import csv
import logging
import multiprocessing
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from google.cloud import storage
from google.oauth2.credentials import Credentials
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("split_upload.log")],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GCS helper
# -----------------------------------------------------------------------------
def make_storage_client():
    # try:
    #     token = subprocess.check_output(
    #         ["gcloud", "auth", "print-access-token"], text=True
    #     ).strip()
    #     creds = Credentials(token)
    #     return storage.Client(credentials=creds)
    # except Exception:
    return storage.Client()


STORAGE = make_storage_client()


def parse_gs(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("Expected gs:// URI")
    bucket, *rest = uri[5:].split("/", 1)
    return bucket, rest[0] if rest else ""


def download_blob_to(uri: str, local: Path):
    b, p = parse_gs(uri)
    STORAGE.bucket(b).blob(p).download_to_filename(str(local))


def upload_blob_from(local: Path, uri: str):
    b, p = parse_gs(uri)
    STORAGE.bucket(b).blob(p).upload_from_filename(str(local))


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def process_and_upload(
    vocals_uri: str,
    segments_gs_prefix: str,
    output_dir: str,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_segment_len: int,
) -> None:
    """
    - downloads `vocals_uri` into a temp dir
    - splits on silence
    - uploads each valid segment to GCS under segments_gs_prefix/<stem_id>/<#####.wav>
    - cleans up temp
    """
    song_name = Path(vocals_uri).parent.name
    out_base = Path(output_dir) / song_name
    out_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav_path = td / f"{song_name}.wav"

        # 1️⃣ download
        try:
            download_blob_to(vocals_uri, wav_path)
        except Exception as e:
            logger.error(f"[{song_name}] download failed: {e}")
            return

        # 2️⃣ load & split
        audio = AudioSegment.from_wav(str(wav_path))
        segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )
        valid = [seg for seg in segments if len(seg) >= min_segment_len]
        logger.info(f"[{song_name}] found {len(valid)} ≥{min_segment_len} ms segments")

        # 3️⃣ export & upload
        for idx, seg in enumerate(valid, start=1):
            fname = f"{idx:05d}.wav"
            seg.export(str(out_base / fname), format="wav")
            # local_seg = td / fname
            # gs_target = segments_gs_prefix.rstrip("/") + f"/{song_name}/{fname}"
            # try:
            #     upload_blob_from(local_seg, gs_target)
            # except Exception as e:
            #     logger.error(f"[{song_name}] upload {fname} failed: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    csv_path: str,
    segments_gs_prefix: str,
    output_dir: str,
    workers: int,
    min_silence_len: int,
    silence_thresh: int,
    keep_silence: int,
    min_segment_len: int,
):
    # load URIs
    uris = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "vocals_uri" not in rdr.fieldnames:
            logger.error("CSV missing 'vocals_uri'")
            return
        for row in rdr:
            uri = row["vocals_uri"].strip()
            if uri:
                uris.append(uri)

    logger.info(f"{len(uris)} URIs to process")

    # use all CPU cores
    num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} parallel processes")

    # process in a pool
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {
            exe.submit(
                process_and_upload,
                uri,
                segments_gs_prefix,
                output_dir,
                min_silence_len,
                silence_thresh,
                keep_silence,
                min_segment_len,
            ): uri
            for uri in uris
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Overall"):
            pass

    logger.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download vocals stems, split on silence, upload segments—no persistent local files"
    )
    p.add_argument("--csv_file", required=True, help="CSV with vocals_uri")
    p.add_argument(
        "--segments_gs_prefix",
        # required=True,
        help="GCS folder, e.g. gs://my-bucket/output_dir",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="local output dir",
    )
    p.add_argument(
        "--workers", type=int, default=4, help="Number of parallel processes"
    )
    p.add_argument(
        "--min_silence_len", type=int, default=2000, help="ms of silence to split on"
    )
    p.add_argument(
        "--silence_thresh", type=int, default=-40, help="dBFS threshold for silence"
    )
    p.add_argument(
        "--keep_silence", type=int, default=100, help="ms of silence to leave at edges"
    )
    p.add_argument(
        "--min_segment_len", type=int, default=3000, help="ms minimum segment length"
    )
    args = p.parse_args()

    main(
        csv_path=args.csv_file,
        segments_gs_prefix=args.segments_gs_prefix,
        output_dir=args.output_dir,
        workers=args.workers,
        min_silence_len=args.min_silence_len,
        silence_thresh=args.silence_thresh,
        keep_silence=args.keep_silence,
        min_segment_len=args.min_segment_len,
    )
