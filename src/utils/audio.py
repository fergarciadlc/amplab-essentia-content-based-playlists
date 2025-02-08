import os
from dataclasses import dataclass

import essentia.standard as es
import numpy as np


@dataclass
class AudioData:
    filepath: str
    audio_stereo: np.ndarray
    sample_rate: int
    num_channels: int
    md5: str
    bit_rate: int
    codec: str

    @property
    def audio_mono(self) -> np.ndarray:
        """Return a mono signal from the loaded audio."""
        if self.num_channels == 1:
            # In Essentia, stereo arrays come as shape=(frames, channels), so for mono:
            # Basically transpose and return the first channel
            # equivalent to: return self.audio_stereo.T[0]
            return self.audio_stereo[:, 0]
        else:
            # Average across channels to downmix to mono
            return self.audio_stereo.mean(axis=1)


def load_audio(filepath: str) -> AudioData:
    """Load audio (potentially stereo) and return a dataclass with stereo + metadata."""
    audio_stereo, sample_rate, num_channels, md5, bit_rate, codec = es.AudioLoader(
        filename=filepath
    )()

    return AudioData(
        filepath=filepath,
        audio_stereo=audio_stereo,
        sample_rate=sample_rate,
        num_channels=num_channels,
        md5=md5,
        bit_rate=bit_rate,
        codec=codec,
    )


if __name__ == "__main__":
    # Usage:
    audio_data = load_audio("../audio/recorded/mozart_c_major_30sec.wav")
    print("Channels:", audio_data.num_channels)
    print("Stereo shape:", audio_data.audio_stereo.shape)
    print("Mono shape:", audio_data.audio_mono.shape)
