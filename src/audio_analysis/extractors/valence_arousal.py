import json
import os
from typing import Dict

import essentia.standard as es
import numpy as np

from audio_analysis.extractors.base_extractor import FeatureExtractor
from audio_analysis.extractors.embeddings import EmbeddingModel, MusicCNNModel
from utils.audio import AudioData


class ValenceArousalExtractor(FeatureExtractor):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        model_weights: str = "audio_analysis/model_weights/emomusic-msd-musicnn-2.pb",
        model_metadata: str = "audio_analysis/model_metadata/emomusic-msd-musicnn-2.json",
    ):

        self._check_if_file_exists(model_weights)
        self._check_if_file_exists(model_metadata)

        self.model_weights = model_weights
        self.model_metadata = model_metadata
        self.embedding_model = embedding_model
        self.metadata = self.get_metadata()
        self.model = es.TensorflowPredict2D(
            graphFilename=self.model_weights, output="model/Identity"
        )
        self.classes = self.metadata["classes"]

    def get_metadata(self):
        with open(self.model_metadata) as f:
            metadata = json.load(f)
        return metadata

    def _check_if_file_exists(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def extract(self, audio_data: AudioData) -> Dict[str, str]:
        embeddings = self.embedding_model.get_audio_embedings(audio_data)
        predictions = self.model(embeddings)
        valence, arousal = np.mean(predictions, axis=0)
        return {
            "emomusic_valence": valence,
            "emomusic_arousal": arousal,
            "musicnn_embeddings_mean": np.mean(embeddings, axis=0).tolist(),
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
    python -m audio_analysis.extractors.valence_arousal [--audio_file path/to/audio/file]
    """
    )
    import json
    from pathlib import Path

    from utils.audio import load_audio

    audio_data: AudioData = load_audio(audio_file_path)
    print(f"Audio file {audio_data.filepath} loaded.")
    genre_extractor = ValenceArousalExtractor(
        embedding_model=MusicCNNModel(),
    )
    genre_results = genre_extractor.extract(audio_data)
    print(json.dumps(genre_results, indent=4))
