# src/audio_analysis/features.py
import json
import os

import essentia
import essentia.standard as es
import numpy as np
from tqdm import tqdm

SEGMENT_LENGTH = 30  # in seconds


def process_audio_file(file_path):
    """
    Process a single audio file to compute various descriptors.

    Returns a dictionary with computed descriptors.
    """
    result = {}
    try:
        audio_mono = es.MonoLoader(filename=file_path)()
        audio_stereo, _, _, _, _, _ = es.AudioLoader(filename=file_path)()
        audio_stereo = es.StereoTrimmer(startTime=0, endTime=SEGMENT_LENGTH)(
            audio_stereo
        )

        # Get sample rate from the loader (default is 44100Hz if not set otherwise)
        sample_rate = 44100  # adjust if needed

        # For models requiring 16kHz, resample:
        audio_16k = es.Resample(inputSampleRate=sample_rate, outputSampleRate=16000)(
            audio_mono
        )

        # --------------------------
        # 1. Tempo Estimation
        # --------------------------
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio_mono)
        result["bpm"] = bpm
        result["beats"] = beats.tolist()

        # --------------------------
        # 2. Key Estimation with Three Profiles
        # --------------------------
        key_temperley = es.KeyExtractor(profileType="temperley")
        key_krumhansl = es.KeyExtractor(profileType="krumhansl")
        key_edma = es.KeyExtractor(profileType="edma")

        key_temp, scale_temp, strength_temp = key_temperley(audio_mono)
        key_krum, scale_krum, strength_krum = key_krumhansl(audio_mono)
        key_edm, scale_edm, strength_edm = key_edma(audio_mono)
        result["key_temperley"] = {
            "key": key_temp,
            "scale": scale_temp,
            "strength": strength_temp,
        }
        result["key_krumhansl"] = {
            "key": key_krum,
            "scale": scale_krum,
            "strength": strength_krum,
        }
        result["key_edma"] = {
            "key": key_edm,
            "scale": scale_edm,
            "strength": strength_edm,
        }

        # --------------------------
        # 3. Loudness (Integrated LUFS) using LoudnessEBUR128
        # --------------------------
        loudness = es.LoudnessEBUR128()
        integrated_loudness = loudness(audio_stereo)
        result["integrated_loudness"] = integrated_loudness

        # --------------------------
        # 4. Embeddings Extraction
        # --------------------------
        # (a) Discogs-Effnet Embedding (e.g., 1280-dimensional)
        discogs_effnet = es.TensorflowPredictEffnetDiscogs(
            graphFilename="discogs_effnet_model.pb"
        )
        embeddings_discogs = discogs_effnet(audio_mono)
        avg_embedding_discogs = np.mean(embeddings_discogs, axis=0)
        result["discogs_effnet_embedding"] = avg_embedding_discogs.tolist()

        # (b) MSD-MusicCNN Embedding (e.g., 200-dimensional)
        msd_musicnn = es.TensorflowPredictMusiCNN(
            graphFilename="msd_musicnn_model.pb", output="model/dense/BiasAdd"
        )
        embeddings_musicnn = msd_musicnn(audio_16k)
        avg_embedding_musicnn = np.mean(embeddings_musicnn, axis=0)
        result["msd_musicnn_embedding"] = avg_embedding_musicnn.tolist()

        # --------------------------
        # 5. Music Styles via Discogs-Effnet Genre Model
        # --------------------------
        discogs_effnet_genre = es.TensorflowPredictEffnetDiscogs(
            graphFilename="discogs_effnet_genre_model.pb"
        )
        genre_activations = discogs_effnet_genre(audio_mono)
        avg_genre_activations = np.mean(genre_activations, axis=0)
        result["genre_activations"] = avg_genre_activations.tolist()

        # --------------------------
        # 6. Voice/Instrumental Classification
        # --------------------------
        voice_instrumental = es.TensorflowPredictVoiceInstrumental(
            graphFilename="voice_instrumental_model.pb"
        )
        voice_inst_preds = voice_instrumental(audio_mono)
        avg_voice_inst = np.mean(voice_inst_preds, axis=0)
        result["voice_instrumental"] = avg_voice_inst.tolist()

        # --------------------------
        # 7. Danceability
        # --------------------------
        danceability = es.TensorflowPredictDanceability(
            graphFilename="danceability_model.pb"
        )
        dance_preds = danceability(audio_mono)
        avg_dance = float(np.mean(dance_preds))
        result["danceability"] = avg_dance

        # --------------------------
        # 8. Arousal & Valence (Emotion)
        # --------------------------
        arousal_valence = es.TensorflowPredictArousalValence(
            graphFilename="arousal_valence_model.pb"
        )
        emo_preds = arousal_valence(audio_16k)
        # Assuming output shape is (frames, 2) for arousal and valence
        avg_arousal = float(np.mean(emo_preds[:, 0]))
        avg_valence = float(np.mean(emo_preds[:, 1]))
        result["arousal"] = avg_arousal
        result["valence"] = avg_valence

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        result["error"] = str(e)

    return result


def process_collection(root_dir):
    """
    Processes all audio files in the given directory.
    Returns a dictionary mapping file paths to descriptor dictionaries.
    """
    # Import the loader function from our loaders module
    from loaders import load_all_audio_files  # Adjust path if necessary

    audio_files = load_all_audio_files(root_dir, extensions=(".mp3",))
    all_results = {}
    for file in tqdm(audio_files, desc="Processing audio files"):
        features = process_audio_file(file)
        all_results[file] = features
    return all_results


if __name__ == "__main__":
    # Define the root directory for the MusAV collection
    root_dir = os.path.join("data", "raw", "MusAV")
    results = process_collection(root_dir)
    # Save the results to a JSON file for persistence
    output_file = os.path.join("data", "processed", "musav_features.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Features for all files have been saved to {output_file}")
