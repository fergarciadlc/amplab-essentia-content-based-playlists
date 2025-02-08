# audio_analysis/base_extractor.py
from typing import Dict, Any
import numpy as np

class AudioFeatureExtractor:
    """
    Base class for audio feature extractors that share a common interface.
    """
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def extract(self, audio_mono: np.ndarray, audio_stereo: np.ndarray) -> Dict[str, Any]:
        """
        Subclasses must implement how they extract features 
        from mono/stereo arrays.
        """
        raise NotImplementedError("Subclasses must implement this method.")