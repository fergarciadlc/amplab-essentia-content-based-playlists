# src/audio_analysis/config.py
from pathlib import Path

from audio_analysis.extractors import KeyExtractor, TempoCNNExtractor

# DATA_DIR = Path("data")
# RAW_DIR = DATA_DIR / "raw" / "MusAV"
# PROCESSED_DIR = DATA_DIR / "processed"


EXTRACTORS = [
    TempoCNNExtractor(
        model_weights="audio_analysis/model_weights/deepsquare-k16-3.pb",
        model_metadata="audio_analysis/model_metadata/deepsquare-k16-3.json",
    ),
    KeyExtractor(profiles=["temperley", "krumhansl", "edma"]),
]
