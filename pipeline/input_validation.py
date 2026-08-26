"""Input validation for the boundary-detection pipeline.

Validates the two things every job must receive before the pipeline runs:
  1. an audio file (exists, loadable, non-empty, within duration bounds)
  2. a transcript/word-timestamp CSV (has word/start/end columns)

Both are required - the pipeline does not run ASR to generate its own
transcript, so a caller must supply one.

This stage only checks structure, so it can fail fast before any
GPU/model work starts. Semantic checks against the audio itself
(timestamps within duration, monotonicity, chunk alignment) belong to
the transcript_alignment module, which runs later once the audio has
been preprocessed and chunked.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import librosa
import pandas as pd

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# canonical column name -> substring used to recognize it if the exact
# canonical name isn't present (case-insensitive)
TRANSCRIPT_COLUMN_KEYWORDS = {
    "word": "word",
    "start_s": "start",
    "end_s": "end",
}

MIN_DURATION_S = 0.1
MAX_DURATION_S = 4 * 60 * 60  # sanity ceiling, not a hard product limit


class InputValidationError(ValueError):
    """Raised when uploaded input fails validation. Message is user-facing."""


@dataclass
class AudioInfo:
    """Basic facts about an audio file, collected while validating it."""

    path: Path
    duration_s: float
    sample_rate: int
    num_channels: int


# ============================================================
# Audio validation
# ============================================================

def validate_audio(audio_path: Union[str, Path]) -> AudioInfo:
    """Check that audio_path exists, is a supported format, and is a reasonable length.

    Input: audio_path - path to the audio file to check.
    Output: AudioInfo with the file's duration, sample rate, and channel count.
    Raises InputValidationError with a user-facing message on any problem.
    """
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
        # Some decode failures (e.g. audioread.exceptions.NoBackendError on a
        # corrupted/unrecognized file) carry no message text at all - str(exc)
        # would be "", leaving a dangling "...: " with no actual information.
        reason = str(exc) or f"{type(exc).__name__} - the file may be corrupted or not a valid audio file"
        raise InputValidationError(f"Could not read audio file '{path.name}': {reason}") from exc

    num_channels = 1 if speech_array.ndim == 1 else speech_array.shape[0]
    num_samples = speech_array.shape[-1]
    duration_s = num_samples / sample_rate if sample_rate else 0.0  # guards a stray sample_rate=0 rather than trusting librosa never returns it

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


# ============================================================
# Transcript CSV validation
# ============================================================

def _resolve_transcript_columns(df: pd.DataFrame, filename: str) -> Dict[str, str]:
    """Map each canonical column name to the actual column in df that represents it.

    Tries an exact (case-insensitive) name match first - this is what real
    exports like data/splits/*.csv use, and it avoids ambiguity with columns
    like 'word_index' or 'global_word_index' that also contain "word". Only
    falls back to substring matching for files that use different naming
    (e.g. 'Start_Time', 'End'), and refuses to guess if that's ambiguous.

    Checks all three canonical columns before raising, so a CSV that's
    missing/ambiguous on more than one of them gets reported together in one
    error - not one at a time across repeated fix-reupload-fail cycles.

    Input: df - the loaded transcript CSV; filename - used only in error messages.
    Output: dict mapping canonical name ("word"/"start_s"/"end_s") to the actual column name in df.
    """
    resolved: Dict[str, str] = {}
    claimed: Dict[str, str] = {}  # actual column -> canonical name that claimed it
    problems: List[str] = []

    for canonical, keyword in TRANSCRIPT_COLUMN_KEYWORDS.items():
        # exact name match wins outright; substring is only a fallback, so a
        # file with both "start_s" and some unrelated "restart" column still
        # resolves cleanly instead of tripping the "more than one column" case
        exact = [c for c in df.columns if c.lower() == canonical.lower()]
        candidates = exact or [c for c in df.columns if keyword in c.lower()]

        if not candidates:
            problems.append(
                f"no column for '{canonical}' (expected a column named '{canonical}', or containing '{keyword}')"
            )
            continue
        if len(candidates) > 1:
            problems.append(f"more than one column could be '{canonical}': {candidates} - rename so only one contains '{keyword}'")
            continue

        col = candidates[0]
        if col in claimed:
            problems.append(f"column '{col}' matches both '{claimed[col]}' and '{canonical}' - rename columns so each is unambiguous")
            continue
        claimed[col] = canonical
        resolved[canonical] = col

    if problems:
        required = ", ".join(TRANSCRIPT_COLUMN_KEYWORDS.keys())
        raise InputValidationError(
            f"Transcript CSV '{filename}' needs columns for: {required}. "
            f"Found columns: {list(df.columns)}. Problems: " + "; ".join(problems)
        )

    return resolved


# ============================================================
# Entry points
# ============================================================

def validate_transcript_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load csv_path and check it has usable word/start_s/end_s columns and sane values.

    Input: csv_path - path to the transcript CSV to check.
    Output: DataFrame with columns renamed to the canonical word/start_s/end_s names.
    Raises InputValidationError on any problem.
    """
    path = Path(csv_path)

    if not path.exists():
        raise InputValidationError(f"Transcript CSV not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise InputValidationError(f"Could not read transcript CSV '{path.name}': {exc}") from exc

    column_map = _resolve_transcript_columns(df, path.name)
    df = df.rename(columns={actual: canonical for canonical, actual in column_map.items()})

    if df.empty:
        raise InputValidationError(f"Transcript CSV '{path.name}' has no rows")

    if df["word"].isna().any():
        raise InputValidationError(f"Transcript CSV '{path.name}' has empty 'word' values")

    if not pd.api.types.is_numeric_dtype(df["start_s"]) or not pd.api.types.is_numeric_dtype(df["end_s"]):
        raise InputValidationError(f"Transcript CSV '{path.name}': 'start_s' and 'end_s' must be numeric")

    if (df["end_s"] < df["start_s"]).any():
        raise InputValidationError(f"Transcript CSV '{path.name}' has rows where end_s < start_s")

    return df


def validate_input(audio_path: Union[str, Path], transcript_csv_path: Union[str, Path]) -> AudioInfo:
    """Validate a job's input before the pipeline runs. Raises InputValidationError on failure.

    Both audio and transcript are required - see module docstring.

    Input: audio_path, transcript_csv_path - paths to the two required files.
    Output: AudioInfo for the audio file (the transcript is checked but not returned here).
    """
    audio_info = validate_audio(audio_path)
    validate_transcript_csv(transcript_csv_path)
    return audio_info
