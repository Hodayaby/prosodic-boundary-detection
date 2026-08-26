"""Audio preprocessing for the boundary-detection pipeline.

Resamples validated input audio to 16kHz mono - the exact convention
already used at feature-extraction time in train_whisper_binary.py and
evaluate_boundary_final2.py (librosa.load(sr=16000, mono=True)). Keeping
preprocessing identical to what the model was trained on avoids a
train/inference mismatch.

Deliberately out of scope: silence trimming (would desync the
transcript's start_s/end_s from the audio) and noise cleaning (no
precedent in the existing codebase, and untested against the trained
model).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import librosa
import numpy as np

TARGET_SAMPLE_RATE = 16000


@dataclass
class PreprocessedAudio:
    """Audio ready for the rest of the pipeline: mono, resampled, with its own metadata."""

    samples: np.ndarray  # mono float32 waveform at TARGET_SAMPLE_RATE
    sample_rate: int
    duration_s: float
    source_path: Path


def preprocess_audio(audio_path: Union[str, Path]) -> PreprocessedAudio:
    """Load audio_path and resample it to 16kHz mono.

    Expects audio_path to have already passed validate_audio() -
    this function does not re-check format or duration.

    Input: audio_path - path to the audio file to load.
    Output: PreprocessedAudio with the resampled mono waveform and its metadata.
    """
    path = Path(audio_path)
    # sr=TARGET_SAMPLE_RATE makes librosa resample during load itself, so
    # duration_s below is already in terms of the resampled audio, not the source file's original rate
    samples, sample_rate = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    duration_s = len(samples) / sample_rate if sample_rate else 0.0

    return PreprocessedAudio(
        samples=samples,
        sample_rate=sample_rate,
        duration_s=duration_s,
        source_path=path,
    )
