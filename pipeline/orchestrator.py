"""Pipeline controller / orchestrator.

Runs every stage for one job in order, on two machines:

  input validation -> audio preprocessing
  -> chunking -> transcript alignment
  -> [SSH/SFTP to BIU] -> classify + merge + threshold + QC

Each stage's own exception is caught and re-raised as PipelineError,
tagged with which stage failed, so callers (the job API, the UI)
don't need to know every stage's specific exception type to report a
clear error. A stage that fails stops the job - later stages never
run on bad input from an earlier one.
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from pipeline.audio_preprocessing import preprocess_audio
from pipeline.biu_sync import BIUCredentials, BIUJobError, run_biu_job
from pipeline.chunker import chunk_audio
from pipeline.input_validation import InputValidationError, validate_audio, validate_transcript_csv
from pipeline.job_logging import StageEventCallback, get_job_logger, log_stage, new_job_id
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
    on_stage_event: Optional[StageEventCallback] = None,
) -> pd.DataFrame:
    """Run one job end to end. Returns the final result table (see pipeline/schema.py).

    Raises PipelineError on any stage's failure, with .stage identifying
    which one and .log_paths populated if it failed during biu_sync.

    Input: audio_path, transcript_csv_path - the job's two input files;
        biu_credentials - BIU login details; local_result_path - where to
        save the final result table; email - optional SLURM notification
        address; local_log_dir - where to save logs if biu_sync fails
        (also where this job's structured log file is written, if given -
        otherwise it's written next to local_result_path); on_stage_event -
        optional callback fired on each stage's start/completed/failed (and,
        during biu_sync, on each SLURM poll) - e.g. to drive a live status
        display. See pipeline.job_logging.StageEventCallback.
    Output: the final result table as a DataFrame.
    """
    local_result_path = Path(local_result_path)
    job_id = new_job_id()
    logger = get_job_logger(job_id, Path(local_log_dir) if local_log_dir else local_result_path.parent)

    with log_stage(logger, "input_validation", on_event=on_stage_event):
        try:
            validate_audio(audio_path)
            transcript = validate_transcript_csv(transcript_csv_path)
        except InputValidationError as exc:
            raise PipelineError("input_validation", str(exc)) from exc

    with log_stage(logger, "audio_preprocessing", on_event=on_stage_event) as metrics:
        try:
            audio = preprocess_audio(audio_path)
        except Exception as exc:
            raise PipelineError("audio_preprocessing", str(exc)) from exc
        metrics["duration_s"] = audio.duration_s

    with log_stage(logger, "chunking", on_event=on_stage_event) as metrics:
        try:
            chunks = chunk_audio(audio, transcript)
        except Exception as exc:
            raise PipelineError("chunking", str(exc)) from exc
        metrics["num_chunks"] = len(chunks)

    with log_stage(logger, "transcript_alignment", on_event=on_stage_event):
        try:
            validate_alignment(audio, transcript, chunks)
        except AlignmentError as exc:
            raise PipelineError("transcript_alignment", str(exc)) from exc

    with log_stage(logger, "biu_sync", on_event=on_stage_event):
        try:
            on_slurm_status = None
            if on_stage_event:
                on_slurm_status = lambda status: on_stage_event(
                    "biu_sync", "polling", {"slurm_job_id": status.slurm_job_id, "slurm_state": status.state}
                )
            result = run_biu_job(
                biu_credentials,
                chunks,
                local_result_path,
                sample_rate=audio.sample_rate,
                email=email,
                local_log_dir=local_log_dir,
                on_status=on_slurm_status,
            )
        except BIUJobError as exc:
            raise PipelineError("biu_sync", str(exc), log_paths=exc.log_paths) from exc

    return result
