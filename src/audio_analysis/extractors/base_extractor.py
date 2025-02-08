from abc import ABC, abstractmethod
from typing import Any, Dict

from utils.audio import AudioData


class FeatureExtractor(ABC):
    def __init__(self, audio_data: AudioData):
        self.audio_data = audio_data

    @abstractmethod
    def extract(self) -> Dict[str, Any]:
        """Extract features from the audio data."""
        pass
