"""Input validation for the boundary-detection pipeline (KAN-46).

Validates the two things a job can receive before the pipeline runs:
  1. an audio file (exists, loadable, non-empty, within duration bounds)
  2. an optional transcript/word-timestamp CSV (has the required columns)

This stage only checks structure, so it can fail fast before any
GPU/model work starts. Semantic checks against the audio itself
(timestamps within duration, monotonicity, chunk alignment) belong to
transcript alignment (KAN-57), which runs later once the audio has been
preprocessed and chunked.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import pandas as pd

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
REQUIRED_TRANSCRIPT_COLUMNS = ("word", "start_s", "end_s")

MIN_DURATION_S = 0.1
MAX_DURATION_S = 4 * 60 * 60  # sanity ceiling, not a hard product limit


class InputValidationError(ValueError):
    """Raised when uploaded input fails validation. Message is user-facing."""


@dataclass
class AudioInfo:
    path: Path
    duration_s: float
    sample_rate: int
    num_channels: int


def validate_audio(audio_path: Union[str, Path]) -> AudioInfo:
    path = Path(audio_path)

    if not path.exists():
        raise InputValidationError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise InputValidationError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    try:
        speech_array, sample_rate = librosa.load(str(path), sr=None, mono=False)
    except Exception as exc:
        raise InputValidationError(f"Could not read audio file '{path.name}': {exc}") from exc

    num_channels = 1 if speech_array.ndim == 1 else speech_array.shape[0]
    num_samples = speech_array.shape[-1]
    duration_s = num_samples / sample_rate if sample_rate else 0.0

    if duration_s < MIN_DURATION_S:
        raise InputValidationError(
            f"Audio file '{path.name}' is too short ({duration_s:.3f}s). "
            f"Minimum duration is {MIN_DURATION_S}s."
        )

    if duration_s > MAX_DURATION_S:
        raise InputValidationError(
            f"Audio file '{path.name}' is too long ({duration_s / 60:.1f} min). "
            f"Maximum supported duration is {MAX_DURATION_S / 3600:.0f} hours."
        )

    return AudioInfo(path=path, duration_s=duration_s, sample_rate=sample_rate, num_channels=num_channels)


def validate_transcript_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise InputValidationError(f"Transcript CSV not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise InputValidationError(f"Could not read transcript CSV '{path.name}': {exc}") from exc

    missing = [c for c in REQUIRED_TRANSCRIPT_COLUMNS if c not in df.columns]
    if missing:
        raise InputValidationError(
            f"Transcript CSV '{path.name}' is missing required columns: {missing}. "
            f"Required columns: {list(REQUIRED_TRANSCRIPT_COLUMNS)}"
        )

    if df.empty:
        raise InputValidationError(f"Transcript CSV '{path.name}' has no rows")

    if df["word"].isna().any():
        raise InputValidationError(f"Transcript CSV '{path.name}' has empty 'word' values")

    if not pd.api.types.is_numeric_dtype(df["start_s"]) or not pd.api.types.is_numeric_dtype(df["end_s"]):
        raise InputValidationError(f"Transcript CSV '{path.name}': 'start_s' and 'end_s' must be numeric")

    if (df["end_s"] < df["start_s"]).any():
        raise InputValidationError(f"Transcript CSV '{path.name}' has rows where end_s < start_s")

    return df


def validate_input(
    audio_path: Union[str, Path],
    transcript_csv_path: Optional[Union[str, Path]] = None,
) -> AudioInfo:
    """Validate a job's input before the pipeline runs. Raises InputValidationError on failure."""
    audio_info = validate_audio(audio_path)

    if transcript_csv_path is not None:
        validate_transcript_csv(transcript_csv_path)

    return audio_info
