"""Transcript validation and alignment to audio.

Runs on the orchestrator, right after chunking and before the chunks
get uploaded to the remote server. Catches problems the earlier,
structure-only input validation can't see: timestamps that exceed the
actual audio duration, timestamps that aren't monotonic, and chunks
whose word list doesn't line up with the chunk's own declared time
span - exactly the kind of mismatch that chunk_audio() would otherwise
silently truncate instead of erroring on.
"""

from typing import List

import pandas as pd

from pipeline.audio_preprocessing import PreprocessedAudio
from pipeline.chunker import AudioChunk


class AlignmentError(ValueError):
    """Raised when the transcript doesn't align with the audio. Message is user-facing."""


def validate_alignment(audio: PreprocessedAudio, transcript: pd.DataFrame, chunks: List[AudioChunk]) -> None:
    """Validate transcript-to-audio and transcript-to-chunk alignment. Raises AlignmentError on failure."""
    _validate_within_duration(audio, transcript)
    _validate_monotonic(transcript)
    _validate_chunk_alignment(audio, chunks)


def _validate_within_duration(audio: PreprocessedAudio, transcript: pd.DataFrame) -> None:
    """Reject a transcript whose last word ends after the audio actually finishes."""
    if transcript.empty:
        return

    max_end_s = transcript["end_s"].max()
    if max_end_s > audio.duration_s + 1e-6:
        raise AlignmentError(
            f"Transcript has a word ending at {max_end_s:.3f}s, but the audio is only "
            f"{audio.duration_s:.3f}s long. Check that the transcript matches this audio file."
        )


def _validate_monotonic(transcript: pd.DataFrame) -> None:
    """Reject a transcript where a word starts before the previous word did."""
    if len(transcript) < 2:
        return

    starts = transcript["start_s"].reset_index(drop=True)
    diffs = starts.diff()
    non_monotonic = diffs.iloc[1:] < 0
    if non_monotonic.any():
        bad_index = non_monotonic.idxmax()
        raise AlignmentError(
            f"Transcript word start times are not monotonic: word at row {bad_index} "
            f"starts at {starts[bad_index]:.3f}s, before the previous word "
            f"({starts[bad_index - 1]:.3f}s)."
        )


def _validate_chunk_alignment(audio: PreprocessedAudio, chunks: List[AudioChunk]) -> None:
    """Reject a chunk whose words fall outside its own time span, or whose audio
    slice length doesn't match that time span - both signs the chunker's math
    and the transcript's timestamps disagree with each other."""
    for chunk in chunks:
        if chunk.words.empty:
            continue

        word_min_start = chunk.words["start_s"].min()
        word_max_end = chunk.words["end_s"].max()
        if word_min_start < chunk.chunk_offset_s - 1e-6 or word_max_end > chunk.end_s + 1e-6:
            raise AlignmentError(
                f"{chunk.chunk_id}: word timestamps [{word_min_start:.3f}, {word_max_end:.3f}]s "
                f"fall outside the chunk's own time span [{chunk.chunk_offset_s:.3f}, {chunk.end_s:.3f}]s."
            )

        expected_samples = round((chunk.end_s - chunk.chunk_offset_s) * audio.sample_rate)
        if abs(len(chunk.samples) - expected_samples) > 1:
            raise AlignmentError(
                f"{chunk.chunk_id}: audio slice has {len(chunk.samples)} samples, expected "
                f"~{expected_samples} for a {chunk.end_s - chunk.chunk_offset_s:.3f}s span - "
                f"the transcript's timestamps likely extend past the actual audio length."
            )
