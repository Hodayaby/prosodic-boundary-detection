"""Long-audio chunker for the boundary-detection pipeline (KAN-50).

Splits preprocessed audio into <=MAX_CHUNK_DURATION_S chunks, cutting
only between words (never mid-word) using the validated transcript's
start_s/end_s. Deliberately no overlap between chunks - see KAN-50:
past error analysis didn't find a meaningful concentration of errors
at chunk edges, and non-overlapping chunks keep the merge step
(KAN-51) simple, since each word belongs to exactly one chunk.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from pipeline.audio_preprocessing import PreprocessedAudio

MAX_CHUNK_DURATION_S = 30.0


@dataclass
class AudioChunk:
    chunk_id: str
    chunk_offset_s: float  # this chunk's start time in the original, global timeline
    end_s: float  # this chunk's end time in the global timeline
    samples: np.ndarray  # audio slice, at the source sample rate
    words: pd.DataFrame  # transcript rows in this chunk (global timestamps, unmodified)


def chunk_audio(
    audio: PreprocessedAudio,
    transcript: pd.DataFrame,
    max_duration_s: float = MAX_CHUNK_DURATION_S,
) -> List[AudioChunk]:
    """Split audio + transcript into <=max_duration_s chunks, splitting only between words.

    transcript must have word/start_s/end_s columns, e.g. as returned by
    validate_transcript_csv (KAN-46). Every word ends up in exactly one chunk.
    """
    if transcript.empty:
        return []

    words = transcript.sort_values("start_s").reset_index(drop=True)
    sample_rate = audio.sample_rate

    chunks: List[AudioChunk] = []
    chunk_start_idx = 0
    chunk_start_s = words.loc[0, "start_s"]

    def close_chunk(end_idx: int, chunk_end_s: float) -> None:
        chunk_words = words.loc[chunk_start_idx:end_idx].reset_index(drop=True)
        start_sample = int(chunk_start_s * sample_rate)
        end_sample = int(chunk_end_s * sample_rate)
        chunks.append(
            AudioChunk(
                chunk_id=f"chunk_{len(chunks)}",
                chunk_offset_s=chunk_start_s,
                end_s=chunk_end_s,
                samples=audio.samples[start_sample:end_sample],
                words=chunk_words,
            )
        )

    for i in range(len(words)):
        word_end_s = words.loc[i, "end_s"]

        # only close early if this chunk already has at least one word -
        # a single word longer than max_duration_s still gets its own chunk
        if word_end_s - chunk_start_s > max_duration_s and i > chunk_start_idx:
            prev_end_s = words.loc[i - 1, "end_s"]
            close_chunk(i - 1, prev_end_s)
            chunk_start_idx = i
            chunk_start_s = words.loc[i, "start_s"]

    close_chunk(len(words) - 1, words.loc[len(words) - 1, "end_s"])

    return chunks
