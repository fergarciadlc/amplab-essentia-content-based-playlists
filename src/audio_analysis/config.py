# src/audio_analysis/config.py
from pathlib import Path

from audio_analysis.extractors import (EffnetDiscogsEmbeddingExtractor,
                                       GenreDiscogs400Extractor, KeyExtractor,
                                       LoudnessEBUR128Extractor,
                                       MusicNNEmbeddingExtractor,
                                       TempoCNNExtractor,
                                       VGGishVoiceInstrumentalExtractor)
from audio_analysis.extractors.embeddings import (EffnetDiscogsModel,
                                                  VGGishModel)

DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw" / "MusAV"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_WEIGHTS_DIR = Path("audio_analysis/model_weights")
MODELS_METADATA_DIR = Path("audio_analysis/model_metadata")


EXTRACTORS = [
    TempoCNNExtractor(
        model_weights=str(MODELS_WEIGHTS_DIR / "deepsquare-k16-3.pb"),
        model_metadata=str(MODELS_METADATA_DIR / "deepsquare-k16-3.json"),
    ),
    KeyExtractor(profiles=["temperley", "krumhansl", "edma"]),
    LoudnessEBUR128Extractor(),
    EffnetDiscogsEmbeddingExtractor(
        model_weights=str(MODELS_WEIGHTS_DIR / "discogs-effnet-bs64-1.pb"),
        model_metadata=str(MODELS_METADATA_DIR / "discogs-effnet-bs64-1.json"),
    ),
    MusicNNEmbeddingExtractor(
        model_weights=str(MODELS_WEIGHTS_DIR / "msd-musicnn-1.pb"),
        model_metadata=str(MODELS_METADATA_DIR / "msd-musicnn-1.json"),
    ),
    GenreDiscogs400Extractor(
        embedding_model=EffnetDiscogsModel(
            model_weights=str(MODELS_WEIGHTS_DIR / "discogs-effnet-bs64-1.pb"),
            model_metadata=str(MODELS_METADATA_DIR / "discogs-effnet-bs64-1.json"),
        ),
        model_weights=str(MODELS_WEIGHTS_DIR / "genre_discogs400-discogs-effnet-1.pb"),
        model_metadata=str(
            MODELS_METADATA_DIR / "genre_discogs400-discogs-effnet-1.json"
        ),
    ),
    VGGishVoiceInstrumentalExtractor(
        embedding_model=VGGishModel(
            model_weights=str(MODELS_WEIGHTS_DIR / "audioset-vggish-3.pb"),
            model_metadata=str(
                MODELS_METADATA_DIR / "voice_instrumental-audioset-vggish-1.json"
            ),
        ),
        model_weights=str(
            MODELS_WEIGHTS_DIR / "voice_instrumental-audioset-vggish-1.pb"
        ),
        model_metadata=str(
            MODELS_METADATA_DIR / "voice_instrumental-audioset-vggish-1.json"
        ),
    ),
]
