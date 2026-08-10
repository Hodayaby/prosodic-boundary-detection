import pytest

from pipeline.audio_preprocessing import TARGET_SAMPLE_RATE, preprocess_audio
from pipeline.input_validation import validate_audio


def test_resamples_44k_stereo_to_16k_mono(make_wav):
    path = make_wav(duration_s=2.0, sr=44100, n_channels=2)
    result = preprocess_audio(path)

    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.samples.ndim == 1
    assert result.duration_s == pytest.approx(2.0, abs=0.05)
    assert len(result.samples) == pytest.approx(TARGET_SAMPLE_RATE * 2, abs=100)


def test_16k_mono_input_stays_16k_mono(make_wav):
    path = make_wav(duration_s=1.5, sr=16000, n_channels=1)
    result = preprocess_audio(path)

    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.samples.ndim == 1


def test_composes_with_validate_audio(make_wav):
    path = make_wav(duration_s=2.0, sr=44100, n_channels=2)
    validate_audio(path)  # structural validation gate, should not raise
    result = preprocess_audio(path)  # resampling, consumes the same path
    assert result.sample_rate == TARGET_SAMPLE_RATE
