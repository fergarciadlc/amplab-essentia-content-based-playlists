# src/audio_analysis/loaders.py
import os

from tqdm import tqdm


def find_audio_files(root_dir, extensions=(".mp3",)):
    """
    Recursively search for audio files within root_dir that have one of the specified extensions.

    Parameters:
        root_dir (str): The root directory to search.
        extensions (tuple): File extensions to include (default: ('.mp3',)).

    Returns:
        List[str]: A list of full file paths for the audio files found.
    """
    audio_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(subdir, file)
                audio_files.append(full_path)
    return audio_files


def load_all_audio_files(root_dir, extensions=(".mp3",)):
    """
    Scans the given directory and returns a list of audio file paths.

    Parameters:
        root_dir (str): The directory containing the audio collection.
        extensions (tuple): Allowed file extensions (default: ('.mp3',)).

    Returns:
        List[str]: A list of discovered audio file paths.
    """
    print(f"Scanning for audio files in {root_dir} ...")
    files = find_audio_files(root_dir, extensions)
    print(f"Found {len(files)} audio files.")
    return files


if __name__ == "__main__":
    # Example usage:
    # Assuming your MusAV collection is in data/raw/MusAV
    root_directory = os.path.join("data", "raw", "MusAV")
    audio_files = load_all_audio_files(root_directory)

    # Optionally, display the list with a progress bar
    for file in tqdm(audio_files, desc="Listing files"):
        print(file)
