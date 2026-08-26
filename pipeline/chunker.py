"""Long-audio chunker for the boundary-detection pipeline.

Splits preprocessed audio into <=MAX_CHUNK_DURATION_S chunks, cutting
only between words (never mid-word) using the validated transcript's
start_s/end_s. Deliberately no overlap between chunks: past error
analysis didn't find a meaningful concentration of errors at chunk
edges, and non-overlapping chunks keep the later merge-predictions
step simple, since each word belongs to exactly one chunk.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from pipeline.audio_preprocessing import PreprocessedAudio

MAX_CHUNK_DURATION_S = 30.0


@dataclass
class AudioChunk:
    """One <=30s slice of audio plus the transcript words that fall inside it."""

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
    validate_transcript_csv. Every word ends up in exactly one chunk.

    Input: audio - preprocessed audio; transcript - its word-level transcript;
        max_duration_s - longest a single chunk is allowed to be.
    Output: list of AudioChunk, covering the whole transcript in order.
    """
    if transcript.empty:
        return []

    words = transcript.sort_values("start_s").reset_index(drop=True)  # defensive - callers should already be sorted, but chunking assumes it
    sample_rate = audio.sample_rate

    chunks: List[AudioChunk] = []
    chunk_start_idx = 0
    chunk_start_s = words.loc[0, "start_s"]

    def close_chunk(end_idx: int, chunk_end_s: float) -> None:
        """Slice out the words and audio samples for the chunk in progress and save it.

        Input: end_idx - index (into words) of the last word in this chunk;
            chunk_end_s - this chunk's end time in seconds.
        Output: none (appends the new AudioChunk to the outer chunks list).
        """
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

    # walks word-by-word rather than slicing by a fixed time window, so the
    # cut always lands on a word boundary - closing on prev_end_s (not the
    # current word's start) means the chunk boundary sits exactly where the
    # previous word finished, with no silence gap or overlap either side
    for i in range(len(words)):
        word_end_s = words.loc[i, "end_s"]

        # only close early if this chunk already has at least one word -
        # a single word longer than max_duration_s still gets its own chunk
        if word_end_s - chunk_start_s > max_duration_s and i > chunk_start_idx:
            prev_end_s = words.loc[i - 1, "end_s"]
            close_chunk(i - 1, prev_end_s)
            chunk_start_idx = i
            chunk_start_s = words.loc[i, "start_s"]

    close_chunk(len(words) - 1, words.loc[len(words) - 1, "end_s"])  # always close the final, still-open chunk

    return chunks
