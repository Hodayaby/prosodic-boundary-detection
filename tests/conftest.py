import math
import struct
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def make_wav(tmp_path):
    """Factory fixture: make_wav(name="a.wav", duration_s=1.0, sr=16000, n_channels=1) -> Path."""

    def _make(name="audio.wav", duration_s=1.0, sr=16000, n_channels=1, freq=440.0):
        path = tmp_path / name
        n_samples = int(duration_s * sr)
        with wave.open(str(path), "w") as f:
            f.setnchannels(n_channels)
            f.setsampwidth(2)
            f.setframerate(sr)
            frames = []
            for i in range(n_samples):
                val = int(3000 * math.sin(2 * math.pi * freq * i / sr))
                for _ in range(n_channels):
                    frames.append(val)
            f.writeframes(struct.pack("<%dh" % len(frames), *frames))
        return path

    return _make


@pytest.fixture
def real_words_csv():
    """Path to a real training-data CSV. data/ is gitignored, so this file
    only exists on machines that already have the project data - skip
    rather than fail when it's missing (e.g. a fresh clone or CI)."""
    path = REPO_ROOT / "data" / "splits" / "test_words.csv"
    if not path.exists():
        pytest.skip(f"real data file not found (data/ is gitignored): {path}")
    return path
