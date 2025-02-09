from typing import Dict, List, Optional, Union

import essentia.standard as es

from audio_analysis.extractors.base_extractor import FeatureExtractor
from utils.audio import AudioData


class KeyExtractor(FeatureExtractor):

    PROFILE_TYPES = (
        "diatonic",
        "krumhansl",
        "temperley",
        # "weichai", # RuntimeError: In KeyExtractor.compute: Key: error in Wei Chai algorithm. Wei Chai algorithm does not support minor scales.
        "tonictriad",
        "temperley2005",
        "thpcp",
        "shaath",
        "gomez",
        "noland",
        # "faraldo", # Is in documentation but not working in Essentia 2.1
        # "pentatonic", # Is in documentation but not working in Essentia 2.1
        "edmm",
        "edma",
        "bgate",
        "braw",
    )

    def __init__(self, profiles: Optional[List[str]] = None):
        if profiles is None:
            profiles = self.PROFILE_TYPES

        invalid_profiles = set(profiles) - set(self.PROFILE_TYPES)
        if invalid_profiles:
            raise ValueError(f"Invalid profile type(s): {', '.join(invalid_profiles)}")

        self.key_extractors = {
            profile: es.KeyExtractor(profileType=profile) for profile in profiles
        }

    def extract(self, audio_data: AudioData) -> Dict[str, Union[str, float]]:
        key_results = {}
        for key_type, key_extractor in self.key_extractors.items():
            key, mode, probability = key_extractor(audio_data.audio_mono)
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

    key_extractor = KeyExtractor(profiles=["temperley", "krumhansl", "edma"])
    key_results = key_extractor.extract(audio_data)
    print(json.dumps(key_results, indent=4))
