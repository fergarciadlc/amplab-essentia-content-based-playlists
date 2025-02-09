# cd src
# python process_audio_collection.py
import logging
from typing import Any, Dict, List

import essentia
from tqdm import tqdm

from audio_analysis.config import EXTRACTORS, RAW_DIR
from audio_analysis.features import AudioFeaturesExtractor
from audio_analysis.loaders import load_all_audio_files
from utils.audio import load_audio

essentia.log.warningActive = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_audio_collection(root_dir: str):

    collection_features: List[Dict[str, Any]] = []
    audio_files = load_all_audio_files(root_dir)

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio_data = load_audio(audio_file)
        audio_features = AudioFeaturesExtractor(
            extractors=EXTRACTORS,
        ).extract(audio_data=audio_data)
        collection_features.append(audio_features)
        # logger.info(f"Extracted features for {audio_file}: {audio_features}")

    print(collection_features)


def main():
    process_audio_collection(str(RAW_DIR))


if __name__ == "__main__":
    main()
