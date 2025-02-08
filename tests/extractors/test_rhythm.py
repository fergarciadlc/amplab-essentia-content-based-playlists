# tests/extractors/test_rhythm.py
import pytest
import numpy as np
import essentia.standard as es
from src.audio_analysis.extractors.rhythm import extract_rhythm_features

@pytest.fixture
def sample_audio_mono():
    # Create or load a small dummy audio signal (e.g., 1 second of silence or random noise)
    # For real testing, you might store small .wav files in a dedicated test data directory
    # and load them. Here we just create an array of zeros.
    return np.zeros(44100, dtype=np.float32)  # 1 second @ 44100 Hz

def test_extract_rhythm_features(sample_audio_mono):
    # Run the function
    result = extract_rhythm_features(sample_audio_mono)
    
    # Check the structure of the result
    assert "bpm" in result
    assert "beats" in result
    # For this dummy audio, you might expect certain default or edge-case values
    # e.g., BPM might default to something or be 0.0 for silence, but we mainly
    # check the function doesn't crash and returns the right keys
    assert isinstance(result["bpm"], float)
    assert isinstance(result["beats"], list)