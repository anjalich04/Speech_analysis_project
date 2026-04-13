import librosa
import numpy as np
from scipy.signal import medfilt


def load_audio(file_path):
    # sr=None keeps the original sample rate of the file.
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate


def normalize_audio(audio_data):
    # Keep audio values in a stable range near [-1, 1].
    max_value = np.max(np.abs(audio_data))
    if max_value == 0:
        return audio_data
    return audio_data / max_value


def reduce_noise(audio_data, kernel_size=5):
    # Basic noise handling using median filter.
    # Small kernel keeps speech shape mostly intact.
    if len(audio_data) == 0:
        return audio_data

    # Kernel size must be odd for medfilt.
    if kernel_size % 2 == 0:
        kernel_size += 1

    return medfilt(audio_data, kernel_size=kernel_size)


def detect_pauses(audio_data, sample_rate):
    # Find non-silent intervals first.
    non_silent_intervals = librosa.effects.split(audio_data, top_db=25)

    pause_segments = []
    total_samples = len(audio_data)

    # If whole audio is silent, return full duration as one pause.
    if len(non_silent_intervals) == 0:
        if total_samples > 0:
            return [(0.0, total_samples / sample_rate)]
        return []

    # Pause before first non-silent part.
    first_start = non_silent_intervals[0][0]
    if first_start > 0:
        pause_segments.append((0.0, first_start / sample_rate))

    # Pauses between non-silent parts.
    for i in range(1, len(non_silent_intervals)):
        previous_end = non_silent_intervals[i - 1][1]
        current_start = non_silent_intervals[i][0]
        if current_start > previous_end:
            pause_segments.append((previous_end / sample_rate, current_start / sample_rate))

    # Pause after last non-silent part.
    last_end = non_silent_intervals[-1][1]
    if last_end < total_samples:
        pause_segments.append((last_end / sample_rate, total_samples / sample_rate))

    return pause_segments


def calculate_total_pause(pause_segments):
    total_pause_duration = 0.0
    for start_time, end_time in pause_segments:
        total_pause_duration += end_time - start_time
    return total_pause_duration


def split_audio_chunks(audio_data, sample_rate, chunk_duration=0.5):
    # Convert chunk duration from seconds to samples.
    samples_per_chunk = int(chunk_duration * sample_rate)
    if samples_per_chunk <= 0:
        return []

    chunks = []
    for start_sample in range(0, len(audio_data), samples_per_chunk):
        end_sample = start_sample + samples_per_chunk
        chunk = audio_data[start_sample:end_sample]

        # Keep full chunks only for consistent comparison.
        if len(chunk) == samples_per_chunk:
            chunks.append(chunk)

    return chunks


def extract_mfcc(audio_data, sample_rate, n_mfcc=13):
    # MFCCs are common features for simple speech analysis.
    return librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)


def _cosine_similarity(vector_a, vector_b):
    numerator = np.dot(vector_a, vector_b)
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def detect_repetition(chunks):
    # Higher threshold reduces false positives.
    similarity_threshold = 0.995
    min_chunk_energy = 0.01

    repeated_indices = []
    in_repetition_run = False

    for i in range(1, len(chunks)):
        previous_chunk = chunks[i - 1]
        current_chunk = chunks[i]

        # Skip very low-energy chunks to avoid false repeats from silence.
        previous_energy = np.mean(np.abs(previous_chunk))
        current_energy = np.mean(np.abs(current_chunk))
        if previous_energy < min_chunk_energy or current_energy < min_chunk_energy:
            in_repetition_run = False
            continue

        # Use MFCC features for chunk comparison.
        previous_mfcc = extract_mfcc(previous_chunk, sample_rate=22050, n_mfcc=13)
        current_mfcc = extract_mfcc(current_chunk, sample_rate=22050, n_mfcc=13)

        # Reduce MFCC matrices to simple 1D vectors.
        previous_vector = np.mean(previous_mfcc, axis=1)
        current_vector = np.mean(current_mfcc, axis=1)

        similarity = _cosine_similarity(previous_vector, current_vector)

        if similarity >= similarity_threshold:
            # Count only once for a streak of similar neighbors.
            if not in_repetition_run:
                repeated_indices.append(i)
                in_repetition_run = True
        else:
            in_repetition_run = False

    return repeated_indices
