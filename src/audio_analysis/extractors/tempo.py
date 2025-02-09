import json
import os
from typing import Dict

import essentia.standard as es

from audio_analysis.extractors.base_extractor import FeatureExtractor
from utils.audio import AudioData


class TempoCNNExtractor(FeatureExtractor):
    def __init__(
        self,
        model_weights: str = "audio_analysis/model_weights/deepsquare-k16-3.pb",
        model_metadata: str = "audio_analysis/model_metadata/deepsquare-k16-3.json",
    ):
        self.model_weights = model_weights
        self.model_metadata = model_metadata
        self._check_if_file_exists(self.model_weights)
        self._check_if_file_exists(self.model_metadata)

        self.model_inference_sample_rate = self._extract_inference_sample_rate()
        self.resampler = es.Resample
        self.model = es.TempoCNN(graphFilename=self.model_weights)

    def _check_if_file_exists(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def _extract_inference_sample_rate(self):
        with open(self.model_metadata) as f:
            metadata = json.load(f)
        return metadata["inference"]["sample_rate"]

    # Override the extract method to include the model inference
    def extract(self, audio_data: AudioData) -> Dict[str, float]:
        audio_mono = audio_data.audio_mono
        if audio_data.sample_rate != self.model_inference_sample_rate:
            audio_mono = self.resampler(
                inputSampleRate=audio_data.sample_rate,
                outputSampleRate=self.model_inference_sample_rate,
            )(audio_mono)

        predicted_bpm, local_tempo, local_tempo_probabilities = self.model(audio_mono)

        return {
            "tempocnn_bpm": predicted_bpm,
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
    python -m audio_analysis.extractors.tempo [--audio_file path/to/audio/file]
"""
    )
    # define files as paths
    from pathlib import Path

    from utils.audio import load_audio

    model_weights_path = str(
        Path("audio_analysis/model_weights/deepsquare-k16-3.pb").resolve()
    )
    model_metadata_path = str(
        Path("audio_analysis/model_metadata/deepsquare-k16-3.json").resolve()
    )

    audio_data = load_audio(audio_file_path)

    print(f"Audio file {audio_data.filepath} loaded.")

    tempo_extractor = TempoCNNExtractor(
        model_weights=model_weights_path,
        model_metadata=model_metadata_path,
    )
    tempo_features = tempo_extractor.extract(audio_data=audio_data)
    print(tempo_features)
