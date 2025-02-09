# src/audio_analysis/config.py
from pathlib import Path

from audio_analysis.extractors import (KeyExtractor, LoudnessEBUR128Extractor,
                                       TempoCNNExtractor)

DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw" / "MusAV"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_WEIGHTS_DIR = Path("audio_analysis/model_weights")
MODELS_METADATA_DIR = Path("audio_analysis/model_metadata")


EXTRACTORS = [
    TempoCNNExtractor(
        model_weights=str(MODELS_WEIGHTS_DIR / "deepsquare-k16-3.pb"),
        model_metadata=str(MODELS_METADATA_DIR / "deepsquare-k16-3.json"),
    ),
    KeyExtractor(profiles=["temperley", "krumhansl", "edma"]),
    LoudnessEBUR128Extractor(),
]
