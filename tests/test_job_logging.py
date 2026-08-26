import pytest

from pipeline.job_logging import log_stage


def test_log_stage_notifies_start_and_completed_with_metrics():
    events = []

    with log_stage("audio_preprocessing", on_event=lambda stage, event, fields: events.append((stage, event, fields))) as metrics:
        metrics["duration_s"] = 12.5

    assert [(e[0], e[1]) for e in events] == [
        ("audio_preprocessing", "start"),
        ("audio_preprocessing", "completed"),
    ]
    assert events[1][2]["duration_s"] == 12.5
    assert events[1][2]["elapsed_s"] >= 0


def test_log_stage_notifies_failure_and_reraises():
    events = []

    with pytest.raises(ValueError, match="boom"):
        with log_stage("chunking", on_event=lambda stage, event, fields: events.append((stage, event, fields))):
            raise ValueError("boom")

    assert [(e[0], e[1]) for e in events] == [("chunking", "start"), ("chunking", "failed")]
    assert events[1][2]["error"] == "boom"


def test_log_stage_works_with_no_callback():
    with log_stage("chunking") as metrics:
        metrics["num_chunks"] = 3
