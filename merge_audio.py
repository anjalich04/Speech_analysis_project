from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def get_project_paths():
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "data"
    output_file = output_dir / "sample.wav"
    return project_root, output_dir, output_file


def find_wav_files(project_root):
    # Read all WAV files from project root only.
    return sorted(project_root.glob("*.wav"))


def load_and_prepare_audio(wav_files):
    audio_parts = []
    target_sample_rate = None

    for wav_path in wav_files:
        try:
            audio_data, sample_rate = librosa.load(str(wav_path), sr=None)
        except Exception as error:
            print(f"Could not read file: {wav_path.name}")
            print(f"Reason: {error}")
            continue

        if target_sample_rate is None:
            # First valid file sets the sample rate used for all output.
            target_sample_rate = sample_rate
        elif sample_rate != target_sample_rate:
            # Resample so all parts match before merging.
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=target_sample_rate,
            )

        audio_parts.append(audio_data)

    return audio_parts, target_sample_rate


def merge_root_wav_files():
    project_root, output_dir, output_file = get_project_paths()

    print("Searching for WAV files in project root...")
    wav_files = find_wav_files(project_root)

    if not wav_files:
        print("No .wav files found in project root.")
        print("Place files like 61-70968-0000.wav next to merge_audio.py and run again.")
        return

    print(f"Found {len(wav_files)} file(s). Loading audio...")
    audio_parts, target_sample_rate = load_and_prepare_audio(wav_files)

    if not audio_parts:
        print("No valid audio files could be loaded.")
        return

    print("Merging audio clips...")
    combined_audio = np.concatenate(audio_parts)

    print("Saving merged file to data/sample.wav...")
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_file), combined_audio, target_sample_rate)

    print("Merge complete.")
    print(f"Input files used: {len(audio_parts)}")
    print(f"Output file: {output_file}")
    print(f"Sample rate: {target_sample_rate} Hz")


if __name__ == "__main__":
    merge_root_wav_files()
