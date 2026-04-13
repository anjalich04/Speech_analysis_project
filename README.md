# Speech Analysis: Pause & Repetition Detection

## Objective
Build a beginner-friendly speech analysis project that detects:
- Pause segments (start time, end time, total pause duration)
- Repetition patterns in speech (for example: stutter-like repeated segments)

## Features
- Pause detection using silence detection
- Total pause duration calculation
- Repetition detection using MFCC + cosine similarity
- False-positive reduction with stricter similarity threshold and run-based counting
- Audio preprocessing with normalization and basic noise handling
- Combine multiple short `.wav` clips into one analysis file

## Technologies Used
- Python
- librosa
- numpy
- scipy
- soundfile

## Approach
1. Audio preprocessing
- Load `data/sample.wav`
- Normalize waveform values
- Apply a basic median filter for simple noise handling

2. Feature extraction (MFCC)
- Split audio into small chunks
- Extract MFCC features from each chunk
- Convert MFCC to compact vectors using mean values

3. Pause detection (silence detection)
- Use `librosa.effects.split` to detect non-silent intervals
- Convert gaps between non-silent intervals into pause segments

4. Repetition detection (segment similarity)
- Compare MFCC vectors of consecutive chunks
- Use cosine similarity to detect repeated patterns
- Avoid over-counting repeated streaks

## Project Structure
- `data/sample.wav`
- `main.py`
- `utils.py`
- `merge_audio.py`
- `README.md`
- `requirements.txt`

## How to Run (Step-by-Step)
1. Open terminal in project folder:
```bash
cd /Users/macbookair/Desktop/speech_analysis
```
2. Create virtual environment:
```bash
python -m venv .venv
```
3. Activate environment:
```bash
source .venv/bin/activate
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Put short LibriSpeech `.wav` files in project root (same folder as `main.py`).
6. Merge clips into one file:
```bash
python merge_audio.py
```
7. Run analysis:
```bash
python main.py
```

## Example Output
```text
Pause Segments:
[0.50s – 1.20s], [2.10s – 2.80s]

Total Pause Duration: 1.40s

Repetitions:
Detected pattern: "ba-ba-ball"
Repetition Count: 2
```

## Challenges Faced
- Input clips can have different sample rates
- Silence threshold selection affects pause quality
- Similar neighboring chunks can cause over-counting
- Short chunks can increase false repetition matches

## Note About LibriSpeech Clips
This project supports combining multiple short clips from the LibriSpeech dataset.
Use `merge_audio.py` to merge root-level `.wav` clips into one file: `data/sample.wav`.
Sample `.wav` clips used for testing can be included in the repository for submission.

## Submission Note
This is a basic implementation for learning and internship-level demonstration.
