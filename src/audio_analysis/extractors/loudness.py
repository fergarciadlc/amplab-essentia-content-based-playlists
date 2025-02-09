from typing import Dict

import essentia.standard as es

from audio_analysis.extractors.base_extractor import FeatureExtractor
from utils.audio import AudioData


class LoudnessEBUR128Extractor(FeatureExtractor):
    def extract(self, audio_data: AudioData) -> Dict[str, float]:
        loudness_extractor = es.LoudnessEBUR128(sampleRate=audio_data.sample_rate)
        momentary_loudness, short_term_loudness, integrated_loudness, loudness_range = (
            loudness_extractor(audio_data.audio_stereo)
        )
        return {
            "loudness_ebur128_integrated_lufs": integrated_loudness,
            "loudness_ebur128_range_lu": loudness_range,
        }


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
    python -m audio_analysis.extractors.loudness [--audio_file path/to/audio/file]
    """
    )
    import json
    from pathlib import Path

    from utils.audio import load_audio

    audio_data: AudioData = load_audio(audio_file_path)
    print(f"Audio file {audio_data.filepath} loaded.")

    loudness_extractor = LoudnessEBUR128Extractor()
    loudness_results = loudness_extractor.extract(audio_data)
    print(json.dumps(loudness_results, indent=4))
