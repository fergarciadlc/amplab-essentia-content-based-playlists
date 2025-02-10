# Essentia Playlists Generator

Assignment: [AMPLab 2025 Essentia playlists assignment.md](./AMPLab%202025%20Essentia%20playlists%20assignment.md)


Tested on Python 3.9.21, Apple Silicon M1

Install dependencies
```bash
pip install -r requirements.txt
```

## Audio Feature Analysis

**MusAV** audio collection dataset:

Make sure to download the dataset and store it in the `data/raw/MusAV` folder.

### Extracting audio features
Run the following command to extract audio features from the dataset:
```bash
cd src
python extract_audio_features.py
```

