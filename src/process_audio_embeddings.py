# cd src
# python process_audio_collection.py
import logging
from datetime import datetime

# from audio_analysis.config import RAW_DIR
# from audio_analysis.config import MODELS_METADATA_DIR, MODELS_WEIGHTS_DIR
from pathlib import Path
from typing import Any, Dict, List

import essentia
from tqdm import tqdm

import numpy as np
from audio_analysis.extractors.embeddings import EffnetDiscogsModel, MusicCNNModel

MODELS_WEIGHTS_DIR = Path("audio_analysis/model_weights")
MODELS_METADATA_DIR = Path("audio_analysis/model_metadata")
RAW_DIR = Path("../data") / "raw" / "MusAV"


from audio_analysis.loaders import load_all_audio_files
from utils.audio import load_audio
from utils.storage import store_csv, store_parquet, store_sqlite

essentia.log.warningActive = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_audio_embeddings_from_collection(root_dir: str):

    collection_embeddings: List[Dict[str, Any]] = []
    audio_files = load_all_audio_files(root_dir)

    discogs_model = EffnetDiscogsModel(
        model_weights=str(MODELS_WEIGHTS_DIR / "discogs-effnet-bs64-1.pb"),
        model_metadata=str(MODELS_METADATA_DIR / "discogs-effnet-bs64-1.json"),
    )
    musicnn_model = MusicCNNModel(
        model_weights=str(MODELS_WEIGHTS_DIR / "msd-musicnn-1.pb"),
        model_metadata=str(MODELS_METADATA_DIR / "msd-musicnn-1.json"),
    )

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio_data = load_audio(audio_file)

        discogs_embeddings = discogs_model.get_audio_embedings(audio_data)
        musicnn_embeddings = musicnn_model.get_audio_embedings(audio_data)

        # print(f"Effnet Discogs embeddings shape: {discogs_embeddings.shape}")
        # print(f"MusicNN embeddings shape: {musicnn_embeddings.shape}")

        collection_embeddings.append(
            {
                "filepath": audio_data.filepath,
                "discogs_embeddings": discogs_embeddings,
                "musicnn_embeddings": musicnn_embeddings,
            }
        )

        # break

    # store in npy/npz format
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    npy_file = f"audio_embeddings_{timestamp}.npy"
    npz_file = f"audio_embeddings_{timestamp}.npz"

    np.save(npy_file, collection_embeddings)
    np.savez_compressed(npz_file, collection_embeddings)


def main():
    extract_audio_embeddings_from_collection(str(RAW_DIR))


if __name__ == "__main__":
    main()
