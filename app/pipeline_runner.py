"""Adapts the UI's raw text inputs into the real pipeline and runs it."""

import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the `pipeline` package

from pipeline.biu_sync import (
    BIUCredentials,
    BIUJobError,
    JobStatus,
    check_pending_job,
    connect,
    discard_pending_job,
    find_pending_jobs,
    recover_pending_job,
)
from pipeline.orchestrator import PipelineError, run_pipeline_job

__all__ = [
    "run_pipeline",
    "test_connection",
    "PipelineError",
    "BIUJobError",
    "JobStatus",
    "find_pending_job",
    "check_pending_job_status",
    "recover_pending_job_result",
    "discard_pending_job_by_dir",
]


def test_connection(host: str, username: str, password: str) -> None:
    """Open and immediately close an SSH connection, to check the login details
    before asking the user to upload anything.

    Input: host, username, password - the BIU login details typed into the UI.
    Output: none. Raises BIUJobError (with a user-facing message) if the
    connection or authentication fails.
    """
    credentials = BIUCredentials(host=host, username=username, password=password)
    with connect(credentials):
        pass


def run_pipeline(
    audio_path: Union[str, Path],
    transcript_csv_path: Union[str, Path],
    host: str,
    username: str,
    password: str,
    local_result_path: Union[str, Path],
    notify_email: Optional[str] = None,
    local_log_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build BIU login credentials from the raw UI fields and run the real pipeline.

    Raises PipelineError on any stage's failure - the error carries
    which stage failed (.stage) and, for a BIU-side failure, local
    copies of the job's log files (.log_paths).

    Input: audio_path, transcript_csv_path - the uploaded files' local paths;
        host, username, password - BIU login details typed into the UI;
        local_result_path - where to save the final result table;
        notify_email - optional address for SLURM's start/end/fail emails
        (separate from `username`, which is a login name, not an email address);
        local_log_dir - where to save logs if the BIU step fails.
    Output: the final result table as a DataFrame.
    """
    credentials = BIUCredentials(host=host, username=username, password=password)
    return run_pipeline_job(
        audio_path,
        transcript_csv_path,
        credentials,
        Path(local_result_path),
        email=notify_email,
        local_log_dir=Path(local_log_dir) if local_log_dir else None,
    )


def find_pending_job(host: str, username: str) -> Optional[dict]:
    """The most recent job recorded for this server/account that a previous
    session lost track of (e.g. a dropped connection), if any.

    Input: host, username - the BIU login details typed into the UI.
    Output: a {job_dir, slurm_job_id, created_at, ...} dict, or None if there isn't one.
    """
    jobs = find_pending_jobs(host, username)
    return jobs[-1] if jobs else None


def check_pending_job_status(host: str, username: str, password: str, slurm_job_id: str) -> JobStatus:
    """Check a pending job's current SLURM status.

    Input: host, username, password - BIU login details; slurm_job_id - from find_pending_job.
    Output: its current JobStatus. Raises BIUJobError if the check itself fails.
    """
    credentials = BIUCredentials(host=host, username=username, password=password)
    return check_pending_job(credentials, slurm_job_id)


def recover_pending_job_result(
    host: str, username: str, password: str, job_dir: str, local_result_path: Union[str, Path]
) -> pd.DataFrame:
    """Fetch a finished pending job's result table and clean up its remote directory.

    Input: host, username, password - BIU login details; job_dir - from find_pending_job;
        local_result_path - where to save the result.
    Output: the result table as a DataFrame. Raises BIUJobError on failure.
    """
    credentials = BIUCredentials(host=host, username=username, password=password)
    return recover_pending_job(credentials, job_dir, Path(local_result_path))


def discard_pending_job_by_dir(host: str, username: str, password: str, job_dir: str) -> None:
    """Delete a pending job's remote directory without fetching its result.

    Input: host, username, password - BIU login details; job_dir - from find_pending_job.
    Output: none. Raises BIUJobError on failure.
    """
    credentials = BIUCredentials(host=host, username=username, password=password)
    discard_pending_job(credentials, job_dir)
