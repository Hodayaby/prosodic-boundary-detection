"""Structured per-job logging.

Each pipeline run gets a JSON-lines log file recording every stage's
start/end time, status, and any metrics the stage records - so a
failure (especially one against BIU, which can't be reproduced without
a live VPN session) can be debugged from the log alone.
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator


def new_job_id() -> str:
    """A short unique id to tag one pipeline run's log file and log lines."""
    return uuid.uuid4().hex[:12]


def get_job_logger(job_id: str, log_dir: Path) -> logging.Logger:
    """A logger, unique to this job, that appends JSON lines to log_dir/<job_id>.log.

    Input: job_id - this run's unique id; log_dir - directory for the log
        file (created if missing).
    Output: a configured logging.Logger. Safe to call more than once for
        the same job_id - a handler is only attached the first time.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pipeline.job.{job_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / f"{job_id}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def close_job_logger(logger: logging.Logger) -> None:
    """Close and detach this job's log handler(s), releasing the log file.

    logging.getLogger caches loggers by name, so without this the
    FileHandler from get_job_logger stays open for the rest of the
    process - harmless by itself, but a caller that deletes the log
    directory once the job's done (e.g. a tempfile.TemporaryDirectory)
    hits a PermissionError on Windows, which refuses to delete a file a
    process still has open.

    Input: logger - the job's logger, as returned by get_job_logger.
    Output: none.
    """
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _emit(logger: logging.Logger, stage: str, event: str, **fields: Any) -> None:
    record: Dict[str, Any] = {"time": time.time(), "stage": stage, "event": event, **fields}
    logger.info(json.dumps(record))


@contextmanager
def log_stage(logger: logging.Logger, stage: str) -> Iterator[Dict[str, Any]]:
    """Log one pipeline stage's start, then its completion or failure, with elapsed time.

    Yields a dict the caller can fill in with stage-specific metrics
    (e.g. metrics["num_chunks"] = len(chunks)) - whatever's in it by the
    end gets logged alongside the completed/failed event.

    Input: logger - this job's logger (see get_job_logger); stage - the stage's name.
    Output: none. Re-raises whatever the wrapped code raised, after logging it as a failure.
    """
    start = time.monotonic()
    _emit(logger, stage, "start")
    metrics: Dict[str, Any] = {}
    try:
        yield metrics
    except Exception as exc:
        _emit(logger, stage, "failed", elapsed_s=round(time.monotonic() - start, 3), error=str(exc), **metrics)
        raise
    else:
        _emit(logger, stage, "completed", elapsed_s=round(time.monotonic() - start, 3), **metrics)
