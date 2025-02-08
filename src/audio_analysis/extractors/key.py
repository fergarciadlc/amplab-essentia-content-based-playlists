from typing import Dict, Union

import essentia.standard as es

from audio_analysis.extractors.base_extractor import FeatureExtractor
from utils.audio import AudioData


class KeyExtractor(FeatureExtractor):
    def __init__(self, audio_data: AudioData):
        super().__init__(audio_data)
        self.key_extractors = {
            "temperley": es.KeyExtractor(profileType="temperley"),
            "krumhansl": es.KeyExtractor(profileType="krumhansl"),
            "edma": es.KeyExtractor(profileType="edma"),
        }

    def extract(self) -> Dict[str, Union[str, float]]:
        key_results = {}
        for key_type, key_extractor in self.key_extractors.items():
            key, mode, probability = key_extractor(self.audio_data.audio_mono)
            key_results[f"key_{key_type}_predict"] = f"{key} {mode}"
            key_results[f"key_{key_type}_probability"] = probability

        return key_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio analysis script")
    parser.add_argument("--audio_file", type=str, help="Path to audio file")
    args = parser.parse_args()

    audio_file_path = args.audio_file
    if not audio_file_path:
        audio_file_path = str(Path("../audio/recorded/techno_loop.mp3").resolve())

    print(
        """
Usage:
    cd src
    python -m audio_analysis.extractors.key [--audio_file path/to/audio/file]
    """
    )
    # define files as paths
    import json
    from pathlib import Path

    from utils.audio import load_audio

    audio_data: AudioData = load_audio(audio_file_path)
    print(f"Audio file {audio_data.filepath} loaded.")

    key_extractor = KeyExtractor(audio_data)
    key_results = key_extractor.extract()
    print(json.dumps(key_results, indent=4))
