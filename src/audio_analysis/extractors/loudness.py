import essentia.standard as es

from utils.audio import AudioData


class LoudnessEBUR128Extractor:
    def __init__(self, audio_data: AudioData):
        self.audio_data = audio_data
        self.loudness_extractor = es.LoudnessEBUR128(
            sampleRate=self.audio_data.sample_rate
        )

    def extract(self):
        momentary_loudness, short_term_loudness, integrated_loudness, loudness_range = (
            self.loudness_extractor(self.audio_data.audio_stereo)
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

    loudness_extractor = LoudnessEBUR128Extractor(audio_data)
    loudness_results = loudness_extractor.extract()
    print(json.dumps(loudness_results, indent=4))
