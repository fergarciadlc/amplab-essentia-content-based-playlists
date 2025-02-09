import json
import os
from typing import Dict, Union

import essentia.standard as es
import numpy as np

from audio_analysis.extractors.base_extractor import FeatureExtractor
from utils.audio import AudioData


class EmbeddingModel:
    def __init__(
        self,
        model: "es.AlgorithmComposite",
        model_weights: str,
        model_metadata: str,
        output: str,
    ):
        self.model_weights = model_weights
        self.model_metadata = model_metadata
        self._check_if_file_exists(self.model_weights)
        self._check_if_file_exists(self.model_metadata)

        self.model_inference_sample_rate = self._extract_inference_sample_rate()
        self.resampler = es.Resample

        self.model = model(graphFilename=self.model_weights, output=output)

    def _check_if_file_exists(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def _extract_inference_sample_rate(self):
        with open(self.model_metadata) as f:
            metadata = json.load(f)
        return metadata["inference"]["sample_rate"]

    def get_audio_embedings(self, audio_data: AudioData) -> np.ndarray:
        audio_mono = audio_data.audio_mono
        if audio_data.sample_rate != self.model_inference_sample_rate:
            audio_mono = self.resampler(
                inputSampleRate=audio_data.sample_rate,
                outputSampleRate=self.model_inference_sample_rate,
            )(audio_mono)

        return self.model(audio_mono)


class EffnetDiscogsModel(EmbeddingModel):
    def __init__(
        self,
        model_weights: str = "audio_analysis/model_weights/discogs-effnet-bs64-1.pb",
        model_metadata: str = "audio_analysis/model_metadata/discogs-effnet-bs64-1.json",
    ):
        super().__init__(
            model=es.TensorflowPredictEffnetDiscogs,
            model_weights=model_weights,
            model_metadata=model_metadata,
            output="PartitionedCall:1",
        )


class MusicCNNModel(EmbeddingModel):
    def __init__(
        self,
        model_weights: str = "audio_analysis/model_weights/msd-musicnn-1.pb",
        model_metadata: str = "audio_analysis/model_metadata/msd-musicnn-1.json",
    ):
        super().__init__(
            model=es.TensorflowPredictMusiCNN,
            model_weights=model_weights,
            model_metadata=model_metadata,
            output="model/dense/BiasAdd",
        )


class VGGishModel(EmbeddingModel):
    def __init__(
        self,
        model_weights: str = "audio_analysis/model_weights/audioset-vggish-3.pb",
        model_metadata: str = "audio_analysis/model_metadata/voice_instrumental-audioset-vggish-1.json",
    ):
        super().__init__(
            model=es.TensorflowPredictVGGish,
            model_weights=model_weights,
            model_metadata=model_metadata,
            output="model/vggish/embeddings",
        )


class EffnetDiscogsEmbeddingExtractor(FeatureExtractor):
    def __init__(
        self,
        model_weights: str = "audio_analysis/model_weights/discogs-effnet-bs64-1.pb",
        model_metadata: str = "audio_analysis/model_metadata/discogs-effnet-bs64-1.json",
    ):
        self.model = EffnetDiscogsModel(
            model_weights=model_weights,
            model_metadata=model_metadata,
        )

    def extract(self, audio_data: AudioData) -> Dict[str, Union[str, float]]:
        audio_embeddings = self.model.get_audio_embedings(audio_data)

        return {
            "effnet_discogs_embeddings_mean": float(audio_embeddings.mean()),
            "effnet_discogs_embeddings_std": float(audio_embeddings.std()),
            "effnet_discogs_embeddings_shape": str(audio_embeddings.shape),
        }


class MusicNNEmbeddingExtractor(FeatureExtractor):
    def __init__(
        self,
        model_weights: str = "audio_analysis/model_weights/msd-musicnn-1.pb",
        model_metadata: str = "audio_analysis/model_metadata/msd-musicnn-1.json",
    ):
        self.model = MusicCNNModel(
            model_weights=model_weights,
            model_metadata=model_metadata,
        )

    def extract(self, audio_data: AudioData) -> Dict[str, Union[str, float]]:
        audio_embeddings = self.model.get_audio_embedings(audio_data)

        return {
            "music_cnn_embeddings_mean": float(audio_embeddings.mean()),
            "music_cnn_embeddings_std": float(audio_embeddings.std()),
            "music_cnn_embeddings_shape": str(audio_embeddings.shape),
        }


if __name__ == "__main__":
    print(
        """
Usage:
    cd src
    python -m audio_analysis.extractors.embeddings [--audio_file path/to/audio/file]
"""
    )
    import argparse
    import json
    from pathlib import Path

    from utils.audio import load_audio

    parser = argparse.ArgumentParser(description="Audio analysis script")
    parser.add_argument("--audio_file", type=str, help="Path to audio file")
    args = parser.parse_args()

    audio_file_path = args.audio_file
    if not audio_file_path:
        audio_file_path = str(Path("../audio/recorded/techno_loop.mp3").resolve())

    audio_data = load_audio(audio_file_path)
    extractor = EffnetDiscogsEmbeddingExtractor()
    embeddings = extractor.extract(audio_data)
    print(json.dumps(embeddings, indent=4))

    print("*" * 80)
    extractor = MusicNNEmbeddingExtractor()
    embeddings = extractor.extract(audio_data)
    print(json.dumps(embeddings, indent=4))


# class EffnetDiscogsModel(EmbeddingModel):
#     def __init__(
#         self,
#         model_weights: str = "audio_analysis/model_weights/discogs-effnet-bs64-1.pb",
#         model_metadata: str = "audio_analysis/model_metadata/discogs-effnet-bs64-1.json",
#     ):
#         self.model_weights = model_weights
#         self.model_metadata = model_metadata
#         self._check_if_file_exists(self.model_weights)
#         self._check_if_file_exists(self.model_metadata)

#         self.model_inference_sample_rate = self._extract_inference_sample_rate()
#         self.resampler = es.Resample

#         self.model_weights = model_weights
#         self.model = es.TensorflowPredictEffnetDiscogs(
#             graphFilename=self.model_weights,
#             output="PartitionedCall:1"
#         )

#     def _check_if_file_exists(self, file_path: str):
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")

#     def _extract_inference_sample_rate(self):
#         with open(self.model_metadata) as f:
#             metadata = json.load(f)
#         return metadata["inference"]["sample_rate"]

#     def get_audio_embedings(self, audio_data: AudioData) -> np.ndarray:
#         audio_mono = audio_data.audio_mono
#         if audio_data.sample_rate != self.model_inference_sample_rate:
#             audio_mono = self.resampler(
#                 inputSampleRate=audio_data.sample_rate,
#                 outputSampleRate=self.model_inference_sample_rate,
#             )(audio_mono)

#         return self.model(audio_mono)
