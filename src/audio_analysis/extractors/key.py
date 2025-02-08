import essentia.standard as es

from utils.audio import AudioData


class KeyExtractor:
    def __init__(self, audio_data: AudioData):
        self.audio_data = audio_data
        self.key_extractors = {
            "temperley": es.KeyExtractor(profileType="temperley"),
            "krumhansl": es.KeyExtractor(profileType="krumhansl"),
            "edma": es.KeyExtractor(profileType="edma"),
        }

    def extract(self):
        key_results = {}
        for key_type, key_extractor in self.key_extractors.items():
            key, mode, probability = key_extractor(self.audio_data.audio_mono)
            key_results[f"key_{key_type}_predict"] = f"{key} {mode}"
            key_results[f"key_{key_type}_probability"] = probability

        return key_results


if __name__ == "__main__":
    print(
        """
Usage:
    cd src
    python -m audio_analysis.extractors.key
    """
    )
    # define files as paths
    import json
    from pathlib import Path

    from utils.audio import load_audio

    audio_file_path = str(Path("../audio/recorded/mozart_c_major_30sec.wav").resolve())

    audio_data: AudioData = load_audio(audio_file_path)
    print(f"Audio file {audio_data.filepath} loaded.")

    key_extractor = KeyExtractor(audio_data)
    key_results = key_extractor.extract()
    print(json.dumps(key_results, indent=4))
