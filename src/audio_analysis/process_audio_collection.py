# src/audio_analysis/process_audio_collection.py
## python process_audio_collection.py
import logging

import tqdm

from audio_analysis.config import EXTRACTORS, RAW_DIR
from audio_analysis.features import AudioFeaturesExtractor
from audio_analysis.loaders import load_all_audio_files
from utils.audio import load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_audio_collection(root_dir: str):

    audio_files = load_all_audio_files(root_dir)[:10]
    collection_features = []
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio_data = load_audio(audio_file)
        audio_features = AudioFeaturesExtractor(
            extractors=EXTRACTORS,
        ).extract()
        collection_features.append(audio_features)
        logger.info(f"Extracted features for {audio_file}: {audio_features}")

    print(collection_features)


if __name__ == "__main__":
    process_audio_collection(str(RAW_DIR.resolve()))
