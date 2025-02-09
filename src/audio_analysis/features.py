# src/audio_analysis/features.py
from datetime import datetime, timezone
from typing import Any, Dict, List

from tqdm import tqdm

from audio_analysis.config import EXTRACTORS
from audio_analysis.extractors.base_extractor import FeatureExtractor
from utils.audio import AudioData, load_audio


class AudioFeaturesExtractor:
    def __init__(self, extractors: List[FeatureExtractor] = EXTRACTORS):
        self.extractors = extractors
        self.features: Dict[str, Any] = None

    def _initiate_features(self, audio_data: AudioData) -> Dict[str, Any]:
        utc_now = datetime.now(timezone.utc)
        return {
            "extracted_time_utc": utc_now.isoformat(),
            "filepath": audio_data.filepath,
            "duration": audio_data.duration,
            "sample_rate": audio_data.sample_rate,
            "num_channels": audio_data.num_channels,
            "md5": audio_data.md5,
            "bit_rate": audio_data.bit_rate,
            "codec": audio_data.codec,
        }

    def extract(self, audio_data: AudioData) -> Dict[str, Any]:
        if not self.features:
            self.features = self._initiate_features(audio_data)
        for extractor in tqdm(self.extractors, desc="Extracting features", leave=False):
            self.features.update(extractor.extract(audio_data=audio_data))

        return self.features


if __name__ == "__main__":
    print(
        """
Usage:
    cd src
    python -m audio_analysis.features [--audio_file path/to/audio/file]
"""
    )
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Audio analysis script")
    parser.add_argument("--audio_file", type=str, help="Path to audio file")
    args = parser.parse_args()

    audio_file_path = args.audio_file
    if not audio_file_path:
        audio_file_path = str(Path("../audio/recorded/techno_loop.mp3").resolve())

    audio_features = AudioFeaturesExtractor(extractors=EXTRACTORS)
    features = audio_features.extract(load_audio(audio_file_path))
    print(json.dumps(features, indent=4))
