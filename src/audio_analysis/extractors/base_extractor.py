from abc import ABC, abstractmethod
from typing import Any, Dict

from utils.audio import AudioData


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, audio_data: AudioData) -> Dict[str, Any]:
        """Extract features from the audio data."""
        pass
