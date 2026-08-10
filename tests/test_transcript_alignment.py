from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.audio_preprocessing import PreprocessedAudio
from pipeline.chunker import chunk_audio
from pipeline.transcript_alignment import AlignmentError, validate_alignment


def _make_audio(duration_s, sr=16000):
    n = int(duration_s * sr)
    return PreprocessedAudio(samples=np.zeros(n, dtype=np.float32), sample_rate=sr, duration_s=duration_s, source_path=Path("fake.wav"))


def _make_transcript(total_duration_s, word_dur=0.3, gap=0.1):
    rows = []
    t = 0.0
    i = 0
    while t < total_duration_s:
        rows.append({"word": f"w{i}", "start_s": t, "end_s": t + word_dur})
        t += word_dur + gap
        i += 1
    return pd.DataFrame(rows)


def test_accepts_well_formed_transcript_and_chunks():
    transcript = _make_transcript(70.0)
    audio = _make_audio(transcript["end_s"].iloc[-1] + 1.0)
    chunks = chunk_audio(audio, transcript)

    validate_alignment(audio, transcript, chunks)  # should not raise


def test_rejects_word_timestamp_past_audio_duration():
    """Regression test: this is the exact scenario that made chunk_audio()
    silently truncate a chunk's audio during development (a word's end_s
    exceeded the audio's actual duration). This must be caught with a
    clear error, not silently produce a short chunk."""
    transcript = _make_transcript(70.0)  # last word's end_s can exceed 70.0
    audio = _make_audio(70.0)  # audio cut off at exactly 70.0s, no padding
    chunks = chunk_audio(audio, transcript)

    with pytest.raises(AlignmentError):
        validate_alignment(audio, transcript, chunks)


def test_rejects_non_monotonic_start_times():
    transcript = pd.DataFrame([
        {"word": "a", "start_s": 0.0, "end_s": 0.3},
        {"word": "b", "start_s": 1.0, "end_s": 1.3},
        {"word": "c", "start_s": 0.5, "end_s": 0.8},  # goes backwards
    ])
    audio = _make_audio(2.0)

    with pytest.raises(AlignmentError):
        validate_alignment(audio, transcript, [])


def test_accepts_empty_transcript_and_chunks():
    audio = _make_audio(5.0)
    validate_alignment(audio, pd.DataFrame(columns=["word", "start_s", "end_s"]), [])  # should not raise


def test_real_transcript_slice_passes(real_words_csv):
    real_df = pd.read_csv(real_words_csv)
    one_file = real_df[real_df["audio_file"] == real_df["audio_file"].iloc[0]].sort_values("start_s").reset_index(drop=True)

    offset = one_file["start_s"].iloc[0]
    one_file = one_file.assign(start_s=one_file["start_s"] - offset, end_s=one_file["end_s"] - offset)
    transcript = one_file[["word", "start_s", "end_s"]]

    audio = _make_audio(transcript["end_s"].iloc[-1] + 1.0)
    chunks = chunk_audio(audio, transcript)

    validate_alignment(audio, transcript, chunks)  # should not raise
