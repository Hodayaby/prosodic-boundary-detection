"""Adapts the UI's raw text inputs into the real pipeline and runs it."""

import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the `pipeline` package

from pipeline.biu_sync import BIUCredentials
from pipeline.orchestrator import PipelineError, run_pipeline_job

__all__ = ["run_pipeline", "PipelineError"]


def run_pipeline(
    audio_path: Union[str, Path],
    transcript_csv_path: Union[str, Path],
    host: str,
    username: str,
    password: str,
    local_result_path: Union[str, Path],
    local_log_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build BIU login credentials from the raw UI fields and run the real pipeline.

    Raises PipelineError on any stage's failure - the error carries
    which stage failed (.stage) and, for a BIU-side failure, local
    copies of the job's log files (.log_paths).

    Input: audio_path, transcript_csv_path - the uploaded files' local paths;
        host, username, password - BIU login details typed into the UI;
        local_result_path - where to save the final result table;
        local_log_dir - where to save logs if the BIU step fails.
    Output: the final result table as a DataFrame.
    """
    credentials = BIUCredentials(host=host, username=username, password=password)
    return run_pipeline_job(
        audio_path,
        transcript_csv_path,
        credentials,
        Path(local_result_path),
        email=username,
        local_log_dir=Path(local_log_dir) if local_log_dir else None,
    )
