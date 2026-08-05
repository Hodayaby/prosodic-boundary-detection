from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.audio_preprocessing import PreprocessedAudio
from pipeline.chunker import MAX_CHUNK_DURATION_S, chunk_audio


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


@pytest.fixture
def long_case():
    transcript = _make_transcript(70.0)
    audio = _make_audio(transcript["end_s"].iloc[-1] + 1.0)  # pad past the last word
    chunks = chunk_audio(audio, transcript)
    return audio, transcript, chunks


def test_produces_multiple_chunks_for_long_audio(long_case):
    _, _, chunks = long_case
    assert len(chunks) >= 3


def test_chunk_spans_within_limit(long_case):
    _, _, chunks = long_case
    for c in chunks:
        assert c.end_s - c.chunk_offset_s <= MAX_CHUNK_DURATION_S + 1e-6


def test_no_overlap_only_natural_inter_word_gap(long_case):
    _, _, chunks = long_case
    for i in range(len(chunks) - 1):
        gap = chunks[i + 1].chunk_offset_s - chunks[i].end_s
        assert -1e-9 <= gap <= 0.1 + 1e-9


def test_every_word_appears_exactly_once(long_case):
    _, transcript, chunks = long_case
    total_words_in_chunks = sum(len(c.words) for c in chunks)
    assert total_words_in_chunks == len(transcript)


def test_word_order_preserved(long_case):
    _, transcript, chunks = long_case
    all_words = pd.concat([c.words for c in chunks])["word"].tolist()
    assert all_words == transcript["word"].tolist()


def test_sample_counts_match_time_spans(long_case):
    audio, _, chunks = long_case
    for c in chunks:
        expected = round((c.end_s - c.chunk_offset_s) * audio.sample_rate)
        assert abs(len(c.samples) - expected) <= 1


def test_empty_transcript_returns_no_chunks():
    audio = _make_audio(10.0)
    assert chunk_audio(audio, pd.DataFrame(columns=["word", "start_s", "end_s"])) == []


def test_oversized_single_word_gets_its_own_chunk():
    transcript = pd.DataFrame([{"word": "looong", "start_s": 0.0, "end_s": 45.0}])
    audio = _make_audio(45.0)
    chunks = chunk_audio(audio, transcript, max_duration_s=30.0)

    assert len(chunks) == 1
    assert len(chunks[0].words) == 1
    assert chunks[0].end_s == 45.0


def test_short_audio_produces_one_chunk():
    transcript = _make_transcript(5.0)
    audio = _make_audio(5.0)
    assert len(chunk_audio(audio, transcript)) == 1


def test_real_transcript_slice_preserves_all_words(real_words_csv):
    real_df = pd.read_csv(real_words_csv)
    one_file = real_df[real_df["audio_file"] == real_df["audio_file"].iloc[0]].sort_values("start_s").reset_index(drop=True)

    offset = one_file["start_s"].iloc[0]
    one_file = one_file.assign(start_s=one_file["start_s"] - offset, end_s=one_file["end_s"] - offset)

    audio = _make_audio(one_file["end_s"].iloc[-1] + 1.0)
    chunks = chunk_audio(audio, one_file[["word", "start_s", "end_s"]])

    total_words = sum(len(c.words) for c in chunks)
    assert total_words == len(one_file)
