import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import orchestrator
from pipeline.audio_preprocessing import PreprocessedAudio
from pipeline.biu_sync import BIUCredentials, BIUJobError
from pipeline.chunker import AudioChunk
from pipeline.input_validation import InputValidationError
from pipeline.orchestrator import PipelineError, run_pipeline_job
from pipeline.transcript_alignment import AlignmentError


def _fake_audio():
    return PreprocessedAudio(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000, duration_s=1.0, source_path=Path("a.wav"))


def _fake_chunks():
    words = pd.DataFrame([{"word": "hi", "start_s": 0.0, "end_s": 1.0}])
    return [AudioChunk(chunk_id="chunk_0", chunk_offset_s=0.0, end_s=1.0, samples=np.zeros(16000, dtype=np.float32), words=words)]


def _patch_happy_path(monkeypatch, calls):
    monkeypatch.setattr(orchestrator, "validate_audio", lambda p: calls.append("validate_audio") or None)
    monkeypatch.setattr(orchestrator, "validate_transcript_csv", lambda p: calls.append("validate_transcript_csv") or pd.DataFrame())
    monkeypatch.setattr(orchestrator, "preprocess_audio", lambda p: calls.append("preprocess_audio") or _fake_audio())
    monkeypatch.setattr(orchestrator, "chunk_audio", lambda audio, transcript: calls.append("chunk_audio") or _fake_chunks())
    monkeypatch.setattr(orchestrator, "validate_alignment", lambda audio, transcript, chunks: calls.append("validate_alignment"))
    monkeypatch.setattr(
        orchestrator, "run_biu_job",
        lambda creds, chunks, local_result_path, sample_rate, email, local_log_dir: calls.append("run_biu_job") or pd.DataFrame({"word": ["hi"]}),
    )


def _creds():
    return BIUCredentials(host="biu.example.edu", username="shira", password="x")


def test_happy_path_runs_all_stages_in_order(monkeypatch, tmp_path):
    calls = []
    _patch_happy_path(monkeypatch, calls)

    df = run_pipeline_job("a.wav", "t.csv", _creds(), tmp_path / "result.csv")

    assert calls == [
        "validate_audio", "validate_transcript_csv",
        "preprocess_audio", "chunk_audio", "validate_alignment", "run_biu_job",
    ]
    assert list(df["word"]) == ["hi"]


def test_input_validation_failure_reports_correct_stage_and_stops_early(monkeypatch, tmp_path):
    calls = []
    _patch_happy_path(monkeypatch, calls)
    monkeypatch.setattr(orchestrator, "validate_audio", lambda p: (_ for _ in ()).throw(InputValidationError("bad audio")))

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline_job("a.wav", "t.csv", _creds(), tmp_path / "result.csv")

    assert exc_info.value.stage == "input_validation"
    assert calls == []  # nothing downstream ran


def test_audio_preprocessing_failure_reports_correct_stage_and_stops_early(monkeypatch, tmp_path):
    calls = []
    _patch_happy_path(monkeypatch, calls)
    monkeypatch.setattr(orchestrator, "preprocess_audio", lambda p: (_ for _ in ()).throw(RuntimeError("corrupt file")))

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline_job("a.wav", "t.csv", _creds(), tmp_path / "result.csv")

    assert exc_info.value.stage == "audio_preprocessing"
    assert calls == ["validate_audio", "validate_transcript_csv"]  # ran up to here, no further


def test_chunking_failure_reports_correct_stage(monkeypatch, tmp_path):
    calls = []
    _patch_happy_path(monkeypatch, calls)
    monkeypatch.setattr(orchestrator, "chunk_audio", lambda audio, transcript: (_ for _ in ()).throw(ValueError("bad chunk math")))

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline_job("a.wav", "t.csv", _creds(), tmp_path / "result.csv")

    assert exc_info.value.stage == "chunking"
    assert "validate_alignment" not in calls
    assert "run_biu_job" not in calls


def test_transcript_alignment_failure_reports_correct_stage(monkeypatch, tmp_path):
    calls = []
    _patch_happy_path(monkeypatch, calls)
    monkeypatch.setattr(
        orchestrator, "validate_alignment",
        lambda audio, transcript, chunks: (_ for _ in ()).throw(AlignmentError("timestamps past audio end")),
    )

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline_job("a.wav", "t.csv", _creds(), tmp_path / "result.csv")

    assert exc_info.value.stage == "transcript_alignment"
    assert "run_biu_job" not in calls


def test_biu_sync_failure_reports_correct_stage_and_propagates_log_paths(monkeypatch, tmp_path):
    calls = []
    _patch_happy_path(monkeypatch, calls)
    fake_logs = [tmp_path / "job_1.out", tmp_path / "job_1.err"]

    def fake_run_biu_job(creds, chunks, local_result_path, sample_rate, email, local_log_dir):
        calls.append("run_biu_job")
        raise BIUJobError("SLURM job 1 ended with state FAILED", log_paths=fake_logs)

    monkeypatch.setattr(orchestrator, "run_biu_job", fake_run_biu_job)

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline_job("a.wav", "t.csv", _creds(), tmp_path / "result.csv")

    assert exc_info.value.stage == "biu_sync"
    assert exc_info.value.log_paths == fake_logs


def test_pipeline_error_rejects_unknown_stage():
    with pytest.raises(ValueError):
        PipelineError("not_a_real_stage", "oops")


def test_happy_path_writes_per_job_log_with_all_stages(monkeypatch, tmp_path):
    calls = []
    _patch_happy_path(monkeypatch, calls)

    run_pipeline_job("a.wav", "t.csv", _creds(), tmp_path / "result.csv")

    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) == 1
    records = [json.loads(line) for line in log_files[0].read_text(encoding="utf-8").splitlines()]
    stages_completed = [r["stage"] for r in records if r["event"] == "completed"]
    assert stages_completed == [
        "input_validation", "audio_preprocessing", "chunking", "transcript_alignment", "biu_sync",
    ]


def test_stage_failure_writes_failed_event_to_job_log(monkeypatch, tmp_path):
    calls = []
    _patch_happy_path(monkeypatch, calls)
    monkeypatch.setattr(orchestrator, "chunk_audio", lambda audio, transcript: (_ for _ in ()).throw(ValueError("bad chunk math")))

    with pytest.raises(PipelineError):
        run_pipeline_job("a.wav", "t.csv", _creds(), tmp_path / "result.csv")

    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) == 1
    records = [json.loads(line) for line in log_files[0].read_text(encoding="utf-8").splitlines()]
    failed = [r for r in records if r["event"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["stage"] == "chunking"
