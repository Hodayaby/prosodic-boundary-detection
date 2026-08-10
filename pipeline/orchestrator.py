"""Pipeline controller / orchestrator (KAN-61).

Runs every stage for one job in order, on two machines:

  input validation (KAN-46) -> audio preprocessing (KAN-56)
  -> chunking (KAN-50) -> transcript alignment (KAN-57)
  -> [SSH/SFTP to BIU] -> classify + merge + threshold + QC (KAN-68/51/58/59)

Each stage's own exception is caught and re-raised as PipelineError,
tagged with which stage failed, so callers (the job API - KAN-60, the
UI - KAN-63/64) don't need to know every stage's specific exception
type to report a clear error. A stage that fails stops the job - later
stages never run on bad input from an earlier one.
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from pipeline.audio_preprocessing import preprocess_audio
from pipeline.biu_sync import BIUCredentials, BIUJobError, run_biu_job
from pipeline.chunker import chunk_audio
from pipeline.input_validation import InputValidationError, validate_audio, validate_transcript_csv
from pipeline.transcript_alignment import AlignmentError, validate_alignment

PIPELINE_STAGES = (
    "input_validation",
    "audio_preprocessing",
    "chunking",
    "transcript_alignment",
    "biu_sync",
)


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails. Message is user-facing.

    stage is one of PIPELINE_STAGES. log_paths carries local SLURM log
    files when the failure happened in biu_sync - see BIUJobError.
    """

    def __init__(self, stage: str, message: str, log_paths: Optional[List[Path]] = None):
        if stage not in PIPELINE_STAGES:
            raise ValueError(f"Unknown pipeline stage: {stage!r}. Expected one of {PIPELINE_STAGES}.")
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.log_paths: List[Path] = log_paths or []


def run_pipeline_job(
    audio_path: Union[str, Path],
    transcript_csv_path: Union[str, Path],
    biu_credentials: BIUCredentials,
    local_result_path: Path,
    email: Optional[str] = None,
    local_log_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run one job end to end. Returns the final result table (KAN-52 schema).

    Raises PipelineError on any stage's failure, with .stage identifying
    which one and .log_paths populated if it failed during biu_sync.
    """
    try:
        validate_audio(audio_path)
        transcript = validate_transcript_csv(transcript_csv_path)
    except InputValidationError as exc:
        raise PipelineError("input_validation", str(exc)) from exc

    try:
        audio = preprocess_audio(audio_path)
    except Exception as exc:
        raise PipelineError("audio_preprocessing", str(exc)) from exc

    try:
        chunks = chunk_audio(audio, transcript)
    except Exception as exc:
        raise PipelineError("chunking", str(exc)) from exc

    try:
        validate_alignment(audio, transcript, chunks)
    except AlignmentError as exc:
        raise PipelineError("transcript_alignment", str(exc)) from exc

    try:
        return run_biu_job(
            biu_credentials,
            chunks,
            local_result_path,
            sample_rate=audio.sample_rate,
            email=email,
            local_log_dir=local_log_dir,
        )
    except BIUJobError as exc:
        raise PipelineError("biu_sync", str(exc), log_paths=exc.log_paths) from exc
