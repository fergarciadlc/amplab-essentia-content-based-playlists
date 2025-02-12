# src/audio_analysis/config.py
from pathlib import Path

from audio_analysis.extractors import (DanceabilityExtractor,
                                       GenreDiscogs400Extractor, KeyExtractor,
                                       LoudnessEBUR128Extractor,
                                       TempoCNNExtractor,
                                       ValenceArousalExtractor,
                                       VGGishVoiceInstrumentalExtractor)
from audio_analysis.extractors.embeddings import (EffnetDiscogsAllExtractors,
                                                  MusicCNNModel)

DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw" / "MusAV"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_WEIGHTS_DIR = Path("audio_analysis/model_weights")
MODELS_METADATA_DIR = Path("audio_analysis/model_metadata")

# fmt: off
EXTRACTORS = [
    TempoCNNExtractor(
        model_weights=str(MODELS_WEIGHTS_DIR / "deepsquare-k16-3.pb"),
        model_metadata=str(MODELS_METADATA_DIR / "deepsquare-k16-3.json"),
    ),
    KeyExtractor(profiles=["temperley", "krumhansl", "edma"]),
    LoudnessEBUR128Extractor(),
    ValenceArousalExtractor(
        embedding_model=MusicCNNModel(
            model_weights=str(MODELS_WEIGHTS_DIR / "msd-musicnn-1.pb"),
            model_metadata=str(MODELS_METADATA_DIR / "msd-musicnn-1.json"),
        ),
        model_weights=str(MODELS_WEIGHTS_DIR / "emomusic-msd-musicnn-2.pb"),
        model_metadata=str(MODELS_METADATA_DIR / "emomusic-msd-musicnn-2.json"),
    ),
    EffnetDiscogsAllExtractors(
        extractors=[
            GenreDiscogs400Extractor(
                embedding_model=None,
                model_weights=str(MODELS_WEIGHTS_DIR / "genre_discogs400-discogs-effnet-1.pb"),
                model_metadata=str(MODELS_METADATA_DIR / "genre_discogs400-discogs-effnet-1.json"),
            ),
            VGGishVoiceInstrumentalExtractor(
                embedding_model=None,
                model_weights=str(MODELS_WEIGHTS_DIR / "voice_instrumental-discogs-effnet-1.pb"),
                model_metadata=str(
                    MODELS_METADATA_DIR / "voice_instrumental-discogs-effnet-1.json"
                ),
            ),
            DanceabilityExtractor(
                embedding_model=None,
                model_weights=str(MODELS_WEIGHTS_DIR / "danceability-discogs-effnet-1.pb"),
                model_metadata=str(MODELS_METADATA_DIR / "danceability-discogs-effnet-1.json"),
            ),
        ]
    ),
]
# fmt: on
