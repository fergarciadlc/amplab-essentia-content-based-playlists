import essentia.standard as es

from utils.audio import AudioData


class LoudnessEBUR128Extractor:
    def __init__(self, audio_data: AudioData):
        self.audio_data = audio_data
        self.loudness_extractor = es.LoudnessEBUR128(
            sampleRate=self.audio_data.sample_rate
        )

    def extract(self):
        # momentaryLoudness, shortTermLoudness, integratedLoudness, loudnessRange = loundsEBUR128(audio_stereo)
        momentary_loudness, shortterm_loudness, integrated_loudness, loudness_range = (
            self.loudness_extractor(self.audio_data.audio_stereo)
        )


if __name__ == "__main__":
    print(
        """
Usage:
    cd src
    python -m audio_analysis.extractors.key
    """
    )
    # define files as paths
    import json
    from pathlib import Path

    from utils.audio import load_audio

    audio_file_path = str(Path("../audio/recorded/mozart_c_major_30sec.wav").resolve())

    audio_data: AudioData = load_audio(audio_file_path)
    print(f"Audio file {audio_data.filepath} loaded.")
