# src/audio_analysis/features.py
from datetime import datetime, timezone
from typing import Any, Dict, List

import audio_analysis.extractors as feature_extractors
from audio_analysis.extractors.base_extractor import FeatureExtractor
from utils.audio import AudioData, load_audio

EXTRACTORS = [
    feature_extractors.KeyExtractor,
    feature_extractors.TempoCNNExtractor,
]


class FeaturesExtractor:
    def __init__(
        self, audio_data: AudioData, extractors: List[FeatureExtractor] = EXTRACTORS
    ):
        self.audio_data = audio_data
        self.extractors = extractors
        self.features: Dict[str, Any] = self._initiate_features()

        print(f"Init features: {self.features}")

    def _initiate_features(self) -> Dict[str, Any]:
        utc_now = datetime.now(timezone.utc)
        return {
            "extracted_time_utc": utc_now.isoformat(),
            "filepath": self.audio_data.filepath,
            "duration": self.audio_data.duration,
            "sample_rate": self.audio_data.sample_rate,
            "num_channels": self.audio_data.num_channels,
            "md5": self.audio_data.md5,
            "bit_rate": self.audio_data.bit_rate,
            "codec": self.audio_data.codec,
        }

    def extract(self):
        for extractor in self.extractors:
            extractor = extractor(self.audio_data)
            self.features.update(extractor.extract())

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

    audio_features = FeaturesExtractor(
        audio_data=load_audio(audio_file_path), extractors=EXTRACTORS
    )
    features = audio_features.extract()
    print(json.dumps(features, indent=4))
