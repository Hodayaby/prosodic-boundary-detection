import inspect

import pandas as pd
import pytest

from pipeline.input_validation import (
    InputValidationError,
    validate_audio,
    validate_input,
    validate_transcript_csv,
)


# --- validate_audio ---

def test_validate_audio_accepts_valid_wav(make_wav):
    path = make_wav(duration_s=2.0, sr=16000)
    info = validate_audio(path)
    assert info.duration_s == pytest.approx(2.0, abs=0.05)
    assert info.sample_rate == 16000


def test_validate_audio_rejects_too_short_audio(make_wav):
    path = make_wav(duration_s=0.01)
    with pytest.raises(InputValidationError):
        validate_audio(path)


def test_validate_audio_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not audio")
    with pytest.raises(InputValidationError):
        validate_audio(path)


def test_validate_audio_rejects_missing_file(tmp_path):
    with pytest.raises(InputValidationError):
        validate_audio(tmp_path / "nope.wav")


def test_validate_audio_rejects_corrupt_file_with_a_non_empty_message(tmp_path):
    """Regression test: a corrupted/unrecognized .wav makes librosa raise
    audioread.exceptions.NoBackendError, whose str() is "" - the old code
    produced a dangling "Could not read audio file 'x.wav': " with no actual
    information. The message must say something even when the underlying
    exception has no text of its own."""
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"this is not a real wav file, just garbage bytes")

    with pytest.raises(InputValidationError) as exc_info:
        validate_audio(path)

    message = str(exc_info.value)
    assert message.strip() != f"Could not read audio file '{path.name}':"
    assert len(message) > len(f"Could not read audio file '{path.name}': ")


# --- validate_transcript_csv: basic structure ---

def test_validate_transcript_csv_accepts_real_data(real_words_csv):
    df = validate_transcript_csv(real_words_csv)
    assert len(df) > 0
    assert {"word", "start_s", "end_s"}.issubset(df.columns)


def test_validate_transcript_csv_rejects_missing_end_s_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"word": ["a"], "start_s": [0.0]}).to_csv(path, index=False)
    with pytest.raises(InputValidationError):
        validate_transcript_csv(path)


def test_validate_transcript_csv_rejects_end_before_start(tmp_path):
    path = tmp_path / "bad_order.csv"
    pd.DataFrame({"word": ["a"], "start_s": [1.0], "end_s": [0.5]}).to_csv(path, index=False)
    with pytest.raises(InputValidationError):
        validate_transcript_csv(path)


def test_validate_transcript_csv_rejects_missing_file(tmp_path):
    with pytest.raises(InputValidationError):
        validate_transcript_csv(tmp_path / "missing.csv")


# --- validate_transcript_csv: flexible column-name matching ---

def test_resolves_renamed_columns_via_keyword_fallback(tmp_path):
    path = tmp_path / "renamed.csv"
    pd.DataFrame({
        "Word_Text": ["hi", "there"],
        "Start_Time": [0.0, 0.5],
        "End_Time": [0.4, 0.9],
    }).to_csv(path, index=False)

    df = validate_transcript_csv(path)
    assert {"word", "start_s", "end_s"}.issubset(df.columns)


def test_resolves_word_exactly_even_when_word_id_also_exists(tmp_path):
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({
        "word": ["hi"], "word_id": [1], "start_s": [0.0], "end_s": [0.4],
    }).to_csv(path, index=False)

    df = validate_transcript_csv(path)  # should not raise
    assert df["word"].tolist() == ["hi"]


def test_rejects_genuinely_ambiguous_word_columns(tmp_path):
    path = tmp_path / "truly_ambiguous.csv"
    pd.DataFrame({
        "Word_Start": ["hi"], "Word_End": ["there"], "start_time": [0.0], "end_time": [0.4],
    }).to_csv(path, index=False)

    with pytest.raises(InputValidationError):
        validate_transcript_csv(path)


def test_rejects_csv_with_no_recognizable_columns(tmp_path):
    path = tmp_path / "no_kw.csv"
    pd.DataFrame({"token": ["hi"], "from": [0.0], "to": [0.4]}).to_csv(path, index=False)
    with pytest.raises(InputValidationError):
        validate_transcript_csv(path)


# --- validate_input ---

def test_validate_input_requires_transcript_csv_path():
    sig = inspect.signature(validate_input)
    assert sig.parameters["transcript_csv_path"].default is inspect.Parameter.empty


def test_validate_input_accepts_audio_and_real_transcript(make_wav, real_words_csv):
    wav = make_wav(duration_s=1.0)
    info = validate_input(wav, real_words_csv)
    assert info.sample_rate is not None
