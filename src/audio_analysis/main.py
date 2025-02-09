# src/audio_analysis/main.py
import json
import logging
import os
from typing import Any, Dict

from tqdm import tqdm

from ..utils.audio import load_audio
from .config import PROCESSED_DIR, RAW_DIR
from .features import process_audio_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_collection(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Processes all audio files in the given directory.
    Returns a dict mapping file paths to their extracted features.
    """
    audio_files = load_all_audio_files(root_dir, extensions=(".mp3", ".wav"))

    all_results = {}
    for file_path in tqdm(audio_files, desc="Processing audio files"):
        all_results[file_path] = process_audio_file(file_path)
    return all_results


def main():
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    results = process_collection(str(RAW_DIR))
    output_file = os.path.join(PROCESSED_DIR, "musav_features.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Features for all files saved to {output_file}")


if __name__ == "__main__":
    main()
