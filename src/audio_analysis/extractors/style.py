import json
import os
from typing import Dict

import essentia.standard as es
import numpy as np

from audio_analysis.extractors.base_extractor import FeatureExtractor
from audio_analysis.extractors.embeddings import (EffnetDiscogsModel,
                                                  EmbeddingModel)
from utils.audio import AudioData


def get_probabilities(predictions, classes):

    # Average probabilities across time
    avg_probs = np.mean(predictions, axis=0)

    # Normalize so they sum to 1
    overall_probs = avg_probs / np.sum(avg_probs)

    # Build a dictionary: {class_name: probability}
    prob_dict = {classes[i]: overall_probs[i] for i in range(len(classes))}
    return prob_dict


def get_highest_probability(prob_dict):
    # Find the class with the highest probability
    best_class = max(prob_dict, key=prob_dict.get)
    return best_class, prob_dict[best_class]


class GenreDiscogs400Extractor(FeatureExtractor):
    def __init__(
        self, embedding_model: EmbeddingModel, model_weights: str, model_metadata: str
    ):

        self._check_if_file_exists(model_weights)
        self._check_if_file_exists(model_metadata)

        self.model_weights = model_weights
        self.model_metadata = model_metadata
        self.embedding_model = embedding_model
        self.metadata = self.get_metadata()
        self.model = es.TensorflowPredict2D(
            graphFilename=self.model_weights,
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
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
        predictions_by_class = get_probabilities(predictions, self.classes)
        best_class, best_prob = get_highest_probability(predictions_by_class)
        return {
            "style_genre_discogs400": best_class,
            "style_genre_discogs400_probability": float(best_prob),
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
    python -m audio_analysis.extractors.stylec [--audio_file path/to/audio/file]
    """
    )
    import json
    from pathlib import Path

    from utils.audio import load_audio

    audio_data: AudioData = load_audio(audio_file_path)
    print(f"Audio file {audio_data.filepath} loaded.")
    genre_extractor = GenreDiscogs400Extractor(
        embedding_model=EffnetDiscogsModel(
            model_weights="audio_analysis/model_weights/discogs-effnet-bs64-1.pb",
            model_metadata="audio_analysis/model_metadata/discogs-effnet-bs64-1.json",
        ),
        model_weights="audio_analysis/model_weights/genre_discogs400-discogs-effnet-1.pb",
        model_metadata="audio_analysis/model_metadata/genre_discogs400-discogs-effnet-1.json",
    )
    genre_results = genre_extractor.extract(audio_data)
    print(json.dumps(genre_results, indent=4))
