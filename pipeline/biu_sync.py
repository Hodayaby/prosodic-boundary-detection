"""SSH/SFTP sync to BIU + SLURM job submission and polling.

Runs on the orchestrator after audio has been validated, preprocessed,
chunked, and aligned to the transcript. Uploads the chunks to a
per-job directory on BIU, submits a SLURM job that runs classification,
merges predictions, applies the threshold, and runs quality checks
(REMOTE_ENTRY_POINT is that remote script), polls until it finishes,
downloads the final result table, and cleans up.

Credentials are held only for the duration of run_biu_job() and are
never written to disk or logged.

Pure/testable logic (time estimate, script rendering, output parsing)
is separated from the paramiko network calls so it can be unit tested
without a live connection. The network calls themselves are only
verified here against mocks - integration testing against the real
BIU server has to happen separately, with real credentials.
"""

import io
import re
import shlex
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
import paramiko
import soundfile as sf

from pipeline.chunker import AudioChunk

REMOTE_REPO_DIR = "ShiraAndHodaya/prosodic-boundary-detection"  # relative to $HOME on BIU - confirmed via SSH
REMOTE_JOBS_DIR = f"{REMOTE_REPO_DIR}/pipeline_jobs"
REMOTE_ENTRY_POINT = "run_pipeline_job.py"  # BIU-side script: classify, merge, threshold, QC; takes --job-dir
SSH_CONNECT_TIMEOUT_S = 15

MIN_JOB_TIME_S = 15 * 60
MAX_JOB_TIME_S = 2 * 60 * 60
OVERHEAD_S = 10 * 60
PER_CHUNK_S = 90

POLL_MAX_RETRIES = 3
POLL_RETRY_DELAY_S = 5.0

TERMINAL_FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}


class BIUJobError(RuntimeError):
    """Raised when connecting to BIU, or a step of the remote job, fails. Message is user-facing.

    log_paths (if any) points to local copies of the job's SLURM stdout/stderr,
    downloaded before the remote job directory was cleaned up - see
    download_logs() and run_biu_job()'s local_log_dir parameter.
    """

    def __init__(self, message: str, log_paths: Optional[List[Path]] = None):
        super().__init__(message)
        self.log_paths: List[Path] = log_paths or []


@dataclass
class BIUCredentials:
    """Login details for the BIU lab account. password is hidden from repr/logs."""

    host: str
    username: str
    password: str = field(repr=False)  # keep out of repr/logs


@dataclass
class JobStatus:
    """The SLURM job's id and its current state (e.g. PENDING, RUNNING, COMPLETED, FAILED, UNKNOWN)."""

    slurm_job_id: str
    state: str  # e.g. PENDING, RUNNING, COMPLETED, FAILED, UNKNOWN


# --- pure logic: no network calls, fully unit-testable ---

def estimate_slurm_time(num_chunks: int) -> str:
    """SLURM --time, scaled by job size: short for a single-file job, up to the
    existing 2h ceiling (matching evaluate_test_model2.slurm) for large batches.

    Input: num_chunks - how many audio chunks this job will process.
    Output: a SLURM-formatted time limit string, e.g. "00:25:00".
    """
    seconds = OVERHEAD_S + num_chunks * PER_CHUNK_S
    seconds = max(MIN_JOB_TIME_S, min(seconds, MAX_JOB_TIME_S))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_slurm_script(job_id: str, job_dir: str, num_chunks: int, email: Optional[str] = None) -> str:
    """Render the SLURM script for this job, following the same directive
    pattern as the existing evaluate_test_model2.slurm.

    job_dir is where sbatch gets submitted from (submit_slurm_job() cd's
    there first), so --output/--error use bare filenames - SLURM resolves
    them relative to $SLURM_SUBMIT_DIR automatically. The code itself
    (run_pipeline_job.py, venv/) lives in REMOTE_REPO_DIR, not job_dir, so
    the script cd's there explicitly via $HOME rather than assuming
    $SLURM_SUBMIT_DIR is the repo - job_dir is a subdirectory of the repo,
    not the repo itself.

    Input: job_id - unique id for this job; job_dir - its remote directory
        (relative to $HOME, includes REMOTE_REPO_DIR); num_chunks - how many
        chunks it will process; email - optional address for SLURM's own
        start/end/fail notifications.
    Output: the full contents of the job.slurm script, as a string.
    """
    mail_lines = ""
    if email:
        mail_lines = f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=BEGIN,END,FAIL\n"

    return f"""#!/bin/bash
#SBATCH --job-name=pipeline_{job_id}
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={estimate_slurm_time(num_chunks)}
{mail_lines}set -e

cd "$HOME/{REMOTE_REPO_DIR}"
source venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Chunks: {num_chunks}"

python {REMOTE_ENTRY_POINT} --job-dir "$HOME/{job_dir}"
"""


def _parse_sbatch_job_id(sbatch_output: str) -> str:
    """Pull the numeric job ID out of sbatch's "Submitted batch job 12345" output.

    Input: sbatch_output - the raw text sbatch printed after submitting the job.
    Output: the job ID as a string. Raises BIUJobError if it can't be found.
    """
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if not match:
        raise BIUJobError(f"Could not parse SLURM job ID from sbatch output: {sbatch_output!r}")
    return match.group(1)


def _parse_sacct_state(sacct_output: str) -> str:
    """sacct --format=State --noheader --parsable2 prints one line per job
    step; states like 'CANCELLED by 12345' get the trailing detail stripped.

    Input: sacct_output - the raw text sacct printed for this job.
    Output: the plain state string (e.g. "RUNNING", "COMPLETED"), or "UNKNOWN" if empty.
    """
    stripped = sacct_output.strip()
    if not stripped:
        return "UNKNOWN"
    first_line = stripped.splitlines()[0].strip()
    if not first_line:
        return "UNKNOWN"
    return first_line.split()[0]


def _chunk_to_wav_bytes(chunk: AudioChunk, sample_rate: int) -> bytes:
    """Encode one audio chunk as an in-memory WAV file, ready to upload.

    Input: chunk - the audio chunk to encode; sample_rate - its sample rate.
    Output: the WAV file's raw bytes.
    """
    buffer = io.BytesIO()
    sf.write(buffer, chunk.samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# --- network calls: exercised via mocks in tests, need real credentials to integration-test ---

@contextmanager
def connect(credentials: BIUCredentials):
    """Open an SSH connection to BIU for the duration of the `with` block.

    Uses AutoAddPolicy (trust-on-first-connect) for simplicity. If BIU's
    host key should be pinned instead, load it via
    ssh.load_system_host_keys() and use RejectPolicy - a deliberate
    choice to revisit, not an oversight.

    Input: credentials - BIU login details (host, username, password).
    Output: an open paramiko SSHClient, via `with connect(creds) as ssh: ...`.
    Raises BIUJobError if the connection or authentication fails.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            hostname=credentials.host,
            username=credentials.username,
            password=credentials.password,
            timeout=SSH_CONNECT_TIMEOUT_S,
        )
    except paramiko.AuthenticationException as exc:
        raise BIUJobError("Authentication to BIU failed - check username/password.") from exc
    except Exception as exc:
        raise BIUJobError(f"Could not connect to BIU ({credentials.host}): {exc}") from exc

    try:
        yield ssh
    finally:
        ssh.close()


def _run_command(ssh: paramiko.SSHClient, command: str) -> str:
    """Run one shell command over the SSH connection and return its output.

    Raises BIUJobError if the command exits with a non-zero status.

    Input: ssh - an open SSH connection; command - the shell command to run.
    Output: the command's stdout text.
    """
    stdin, stdout, stderr = ssh.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()

    if exit_status != 0:
        raise BIUJobError(f"Remote command failed (exit {exit_status}): {command!r}: {err.strip() or out.strip()}")

    return out


def upload_job(ssh: paramiko.SSHClient, chunks: List[AudioChunk], sample_rate: int, email: Optional[str] = None) -> str:
    """Create a per-job remote directory and upload chunk audio + the SLURM script.

    Input: ssh - an open SSH connection; chunks - audio chunks to upload;
        sample_rate - their sample rate; email - optional SLURM notification address.
    Output: the new remote job directory's path.
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = f"{REMOTE_JOBS_DIR}/{job_id}"

    _run_command(ssh, f"mkdir -p {shlex.quote(job_dir)}")

    sftp = ssh.open_sftp()
    try:
        for chunk in chunks:
            wav_bytes = _chunk_to_wav_bytes(chunk, sample_rate)
            with sftp.open(f"{job_dir}/{chunk.chunk_id}.wav", "wb") as f:
                f.write(wav_bytes)

            words_csv = chunk.words.to_csv(index=False)
            with sftp.open(f"{job_dir}/{chunk.chunk_id}_words.csv", "w") as f:
                f.write(words_csv)

        script = build_slurm_script(job_id, job_dir, len(chunks), email=email)
        with sftp.open(f"{job_dir}/job.slurm", "w") as f:
            f.write(script)
    finally:
        sftp.close()

    return job_dir


def submit_slurm_job(ssh: paramiko.SSHClient, job_dir: str) -> str:
    """Submit the already-uploaded job.slurm script and return the new SLURM job ID.

    Input: ssh - an open SSH connection; job_dir - the remote job directory.
    Output: the new SLURM job ID as a string.
    """
    output = _run_command(ssh, f"cd {shlex.quote(job_dir)} && sbatch job.slurm")
    return _parse_sbatch_job_id(output)


def poll_job_status(ssh: paramiko.SSHClient, slurm_job_id: str) -> JobStatus:
    """Ask SLURM for this job's current state, one single check (no retry).

    Input: ssh - an open SSH connection; slurm_job_id - the job to check.
    Output: a JobStatus with the job's id and current state.
    """
    command = f"sacct -j {shlex.quote(slurm_job_id)} --format=State --noheader --parsable2"
    output = _run_command(ssh, command)
    return JobStatus(slurm_job_id=slurm_job_id, state=_parse_sacct_state(output))


def poll_job_status_with_retry(
    ssh: paramiko.SSHClient,
    slurm_job_id: str,
    max_retries: int = POLL_MAX_RETRIES,
    retry_delay_s: float = POLL_RETRY_DELAY_S,
) -> JobStatus:
    """poll_job_status(), retrying on transient failures (a dropped VPN/network
    blip during one sacct call) instead of giving up on tracking the whole job.

    Only retries exceptions raised while trying to poll. A job that legitimately
    reached a terminal FAILED state is a normal JobStatus return, not an
    exception - that's handled by the caller in run_biu_job(), unaffected by this.

    Input: ssh - an open SSH connection; slurm_job_id - the job to check;
        max_retries - how many extra attempts before giving up; retry_delay_s -
        pause between attempts.
    Output: a JobStatus. Raises BIUJobError if every attempt failed.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return poll_job_status(ssh, slurm_job_id)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_delay_s)

    raise BIUJobError(
        f"Lost track of SLURM job {slurm_job_id} after {max_retries} retries "
        f"(likely a dropped connection, not necessarily a failed job): {last_exc}"
    ) from last_exc


def download_result(ssh: paramiko.SSHClient, job_dir: str, local_path: Path) -> None:
    """Copy the finished job's result.csv from BIU down to local_path.

    Input: ssh - an open SSH connection; job_dir - the remote job directory;
        local_path - where to save the file locally.
    Output: none (the file is written to local_path). Raises BIUJobError if missing.
    """
    sftp = ssh.open_sftp()
    try:
        remote_path = f"{job_dir}/result.csv"
        try:
            sftp.get(remote_path, str(local_path))
        except FileNotFoundError as exc:
            raise BIUJobError(
                f"Result file not found at {remote_path} - the job may have failed before producing output."
            ) from exc
    finally:
        sftp.close()


def download_logs(ssh: paramiko.SSHClient, job_dir: str, local_dir: Path) -> List[Path]:
    """Best-effort download of the SLURM stdout/stderr (job_*.out/.err) from job_dir.

    Called on failure/timeout, before cleanup_job() deletes the remote
    directory, so the user has something to look at instead of just
    "job failed". Swallows its own errors - a logs-download problem
    should never mask the original failure that triggered it.

    Input: ssh - an open SSH connection; job_dir - the remote job directory;
        local_dir - where to save the downloaded log files.
    Output: list of local paths actually downloaded (may be empty).
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []

    try:
        sftp = ssh.open_sftp()
        try:
            for remote_name in sftp.listdir(job_dir):
                if remote_name.endswith(".out") or remote_name.endswith(".err"):
                    local_path = local_dir / remote_name
                    sftp.get(f"{job_dir}/{remote_name}", str(local_path))
                    downloaded.append(local_path)
        finally:
            sftp.close()
    except Exception:
        pass

    return downloaded


def cleanup_job(ssh: paramiko.SSHClient, job_dir: str) -> None:
    """Delete the job's remote directory once we're done with it.

    Deliberately runs even after failure/timeout - BIU is a shared
    filesystem, and orphaned job directories should never be left
    behind. Call download_logs() first (as run_biu_job() does) if the
    logs need to survive past this point.

    Input: ssh - an open SSH connection; job_dir - the remote job directory to delete.
    Output: none.
    """
    _run_command(ssh, f"rm -rf {shlex.quote(job_dir)}")


def run_biu_job(
    credentials: BIUCredentials,
    chunks: List[AudioChunk],
    local_result_path: Path,
    sample_rate: int = 16000,
    email: Optional[str] = None,
    poll_interval_s: float = 10.0,
    timeout_s: float = MAX_JOB_TIME_S + 300,
    local_log_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Upload chunks, run the SLURM job on BIU, wait for it, and return the final result table.

    On failure or timeout, downloads job_*.out/.err into local_log_dir (if
    given) before the remote job directory is cleaned up, and attaches
    their paths to the raised BIUJobError.log_paths so the orchestrator
    can offer them to the user instead of just an error string.

    Raises BIUJobError on connection failure, job failure, or timeout.

    Input: credentials - BIU login details; chunks - audio chunks to process;
        local_result_path - where to save the final result table; sample_rate -
        the chunks' sample rate; email - optional SLURM notification address;
        poll_interval_s - seconds between status checks; timeout_s - give up
        after this long; local_log_dir - where to save logs on failure.
    Output: the final result table as a DataFrame.
    """
    with connect(credentials) as ssh:
        job_dir = upload_job(ssh, chunks, sample_rate=sample_rate, email=email)
        try:
            slurm_job_id = submit_slurm_job(ssh, job_dir)

            start = time.monotonic()
            while True:
                status = poll_job_status_with_retry(ssh, slurm_job_id)

                if status.state == "COMPLETED":
                    break
                if status.state in TERMINAL_FAILURE_STATES:
                    log_paths = download_logs(ssh, job_dir, local_log_dir) if local_log_dir else []
                    raise BIUJobError(f"SLURM job {slurm_job_id} ended with state {status.state}", log_paths=log_paths)
                if time.monotonic() - start > timeout_s:
                    log_paths = download_logs(ssh, job_dir, local_log_dir) if local_log_dir else []
                    raise BIUJobError(
                        f"Timed out waiting for SLURM job {slurm_job_id} (last state: {status.state})",
                        log_paths=log_paths,
                    )

                time.sleep(poll_interval_s)

            download_result(ssh, job_dir, local_result_path)
        finally:
            cleanup_job(ssh, job_dir)

    return pd.read_csv(local_result_path)
