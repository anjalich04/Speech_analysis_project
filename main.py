from pathlib import Path

from utils import (
    calculate_total_pause,
    detect_pauses,
    detect_repetition,
    load_audio,
    normalize_audio,
    reduce_noise,
    split_audio_chunks,
)


def get_input_audio_path():
    # Build path relative to this file so it works from any terminal location.
    project_root = Path(__file__).resolve().parent
    return project_root / "data" / "sample.wav"


def print_pause_segments(pause_segments):
    print("Pause Segments:")
    if not pause_segments:
        print("No pause segments detected")
        return

    # Print one pause per line in [start – end] format.
    for start_time, end_time in pause_segments:
        print(f"[{start_time:.2f}s – {end_time:.2f}s]")


def print_repetition_summary(repeated_indices):
    print("\nRepetitions:")
    if repeated_indices:
        print('Detected pattern: "similar segment repetition"')
        print(f"Repetition Count: {len(repeated_indices)}")
    else:
        print('Detected pattern: "No repetition detected"')
        print("Repetition Count: 0")


def main():
    audio_path = get_input_audio_path()

    if not audio_path.exists():
        print("data/sample.wav not found.")
        print("Run merge_audio.py first, or place a WAV file at data/sample.wav.")
        return

    print("Loading audio...")
    audio_data, sample_rate = load_audio(str(audio_path))

    print("Preprocessing audio (normalize + basic noise handling)...")
    normalized_audio = normalize_audio(audio_data)
    clean_audio = reduce_noise(normalized_audio)

    print("Detecting pauses and repetitions...")
    pause_segments = detect_pauses(clean_audio, sample_rate)
    total_pause_duration = calculate_total_pause(pause_segments)

    chunks = split_audio_chunks(clean_audio, sample_rate, chunk_duration=0.5)
    repeated_indices = detect_repetition(chunks)

    print()
    print_pause_segments(pause_segments)
    print(f"\nTotal Pause Duration: {total_pause_duration:.2f}s")
    print_repetition_summary(repeated_indices)


if __name__ == "__main__":
    main()
