# src/audio_analysis/config.py
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "MusAV"
PROCESSED_DIR = DATA_DIR / "processed"
SEGMENT_LENGTH = 30  # seconds
SAMPLE_RATE = 44100
RESAMPLE_RATE = 16000

# Paths to model files (example)
DISCOGS_EFFNET_MODEL = "discogs_effnet_model.pb"
MSD_MUSICNN_MODEL = "msd_musicnn_model.pb"
DISCOGS_EFFNET_GENRE_MODEL = "discogs_effnet_genre_model.pb"
VOICE_INSTRUMENTAL_MODEL = "voice_instrumental_model.pb"
DANCEABILITY_MODEL = "danceability_model.pb"
AROUSAL_VALENCE_MODEL = "arousal_valence_model.pb"