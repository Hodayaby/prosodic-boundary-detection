import json

import pytest

from pipeline.job_logging import get_job_logger, log_stage, new_job_id


def _read_log_lines(log_path):
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_new_job_id_is_unique():
    assert new_job_id() != new_job_id()


def test_log_stage_records_start_and_completed_with_metrics(tmp_path):
    job_id = new_job_id()
    logger = get_job_logger(job_id, tmp_path)

    with log_stage(logger, "audio_preprocessing") as metrics:
        metrics["duration_s"] = 12.5

    records = _read_log_lines(tmp_path / f"{job_id}.log")
    assert [r["event"] for r in records] == ["start", "completed"]
    assert all(r["stage"] == "audio_preprocessing" for r in records)
    assert records[1]["duration_s"] == 12.5
    assert records[1]["elapsed_s"] >= 0


def test_log_stage_records_failure_and_reraises(tmp_path):
    job_id = new_job_id()
    logger = get_job_logger(job_id, tmp_path)

    with pytest.raises(ValueError, match="boom"):
        with log_stage(logger, "chunking"):
            raise ValueError("boom")

    records = _read_log_lines(tmp_path / f"{job_id}.log")
    assert [r["event"] for r in records] == ["start", "failed"]
    assert records[1]["error"] == "boom"


def test_log_stage_invokes_on_event_callback(tmp_path):
    job_id = new_job_id()
    logger = get_job_logger(job_id, tmp_path)
    events = []

    with log_stage(logger, "chunking", on_event=lambda stage, event, fields: events.append((stage, event, fields))) as metrics:
        metrics["num_chunks"] = 3

    assert events[0] == ("chunking", "start", {})
    assert events[1][:2] == ("chunking", "completed")
    assert events[1][2]["num_chunks"] == 3


def test_log_stage_invokes_on_event_callback_on_failure(tmp_path):
    job_id = new_job_id()
    logger = get_job_logger(job_id, tmp_path)
    events = []

    with pytest.raises(ValueError):
        with log_stage(logger, "chunking", on_event=lambda stage, event, fields: events.append((stage, event, fields))):
            raise ValueError("boom")

    assert events[0] == ("chunking", "start", {})
    assert events[1][1] == "failed"
    assert events[1][2]["error"] == "boom"


def test_get_job_logger_does_not_duplicate_handlers_on_repeat_calls(tmp_path):
    job_id = new_job_id()
    logger_a = get_job_logger(job_id, tmp_path)
    logger_b = get_job_logger(job_id, tmp_path)

    assert logger_a is logger_b
    assert len(logger_a.handlers) == 1
