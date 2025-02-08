# src/audio_analysis/extractors/rhythm.py
from typing import Any, Dict, List

import essentia.standard as es


def extract_rhythm_features(audio: Any) -> Dict[str, Any]:
    """
    Extract BPM and beats from a mono audio signal.

    :param audio: Mono audio array loaded via Essentia.
    :return: A dictionary containing BPM and beats array.
    """
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)

    return {"bpm": bpm, "beats": beats.tolist()}
