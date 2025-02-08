# src/audio_analysis/features.py
from typing import List

from utils.audio import AudioData, load_audio


class FeaturesExtractor:
    def __init__(self, audio_data: AudioData, extractors: List):
        self.features = {}
        self.audio_data = audio_data
