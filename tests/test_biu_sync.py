from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import paramiko
import pytest
import soundfile as sf

from pipeline import biu_sync
from pipeline.biu_sync import (
    BIUCredentials,
    BIUJobError,
    JobStatus,
    _chunk_to_wav_bytes,
    _parse_sacct_state,
    _parse_sbatch_job_id,
    _run_command,
    build_slurm_script,
    cleanup_job,
    connect,
    download_logs,
    download_result,
    estimate_slurm_time,
    poll_job_status,
    poll_job_status_with_retry,
    submit_slurm_job,
    upload_job,
)
from pipeline.chunker import AudioChunk


def _make_chunk(chunk_id="chunk_0", duration_s=2.0, sr=16000):
    n = int(duration_s * sr)
    words = pd.DataFrame([{"word": "hi", "start_s": 0.0, "end_s": duration_s}])
    return AudioChunk(chunk_id=chunk_id, chunk_offset_s=0.0, end_s=duration_s, samples=np.zeros(n, dtype=np.float32), words=words)


# --- pure logic ---

def test_estimate_slurm_time_floors_small_jobs():
    assert estimate_slurm_time(0) == "00:15:00"  # OVERHEAD_S alone (10min) is below the 15min floor


def test_estimate_slurm_time_scales_with_chunk_count():
    small = estimate_slurm_time(2)
    large = estimate_slurm_time(50)
    assert small < large  # string HH:MM:SS compares correctly lexicographically here since both start "00:"


def test_estimate_slurm_time_caps_at_two_hours():
    assert estimate_slurm_time(100_000) == "02:00:00"


def test_build_slurm_script_has_expected_directives():
    script = build_slurm_script("abc123", "pipeline_jobs/abc123", num_chunks=3)
    assert "--partition=L4-4h" in script
    assert "--gres=gpu:1" in script
    assert "--mem=32G" in script
    assert "--job-name=pipeline_abc123" in script
    assert 'python run_pipeline_job.py --job-dir "$HOME/pipeline_jobs/abc123"' in script


def test_build_slurm_script_cds_into_repo_dir_not_submit_dir():
    """Regression test: the script used to `cd $SLURM_SUBMIT_DIR` and assume
    run_pipeline_job.py/venv were there - but $SLURM_SUBMIT_DIR is the
    per-job directory, not the repo. It must cd into the repo explicitly."""
    script = build_slurm_script("abc123", "pipeline_jobs/abc123", num_chunks=3)
    assert "$SLURM_SUBMIT_DIR" not in script
    assert f'cd "$HOME/{biu_sync.REMOTE_REPO_DIR}"' in script


def test_build_slurm_script_uses_bare_output_filenames():
    """--output/--error must be bare filenames (resolved relative to
    $SLURM_SUBMIT_DIR, which submit_slurm_job() sets to job_dir by cd-ing
    there before sbatch) - not job_dir-prefixed, which would double up."""
    script = build_slurm_script("abc123", "pipeline_jobs/abc123", num_chunks=3)
    assert "--output=job_%j.out" in script
    assert "--error=job_%j.err" in script


def test_build_slurm_script_omits_mail_lines_without_email():
    script = build_slurm_script("abc123", "pipeline_jobs/abc123", num_chunks=3, email=None)
    assert "--mail-user" not in script


def test_build_slurm_script_includes_mail_lines_with_email():
    script = build_slurm_script("abc123", "pipeline_jobs/abc123", num_chunks=3, email="shira@example.com")
    assert "--mail-user=shira@example.com" in script
    assert "--mail-type=BEGIN,END,FAIL" in script


def test_parse_sbatch_job_id_success():
    assert _parse_sbatch_job_id("Submitted batch job 123456\n") == "123456"


def test_parse_sbatch_job_id_failure_raises():
    with pytest.raises(BIUJobError):
        _parse_sbatch_job_id("sbatch: error: something went wrong\n")


@pytest.mark.parametrize("output,expected", [
    ("COMPLETED\n", "COMPLETED"),
    ("RUNNING\n", "RUNNING"),
    ("CANCELLED by 999999\n", "CANCELLED"),
    ("", "UNKNOWN"),
    ("   \n", "UNKNOWN"),
])
def test_parse_sacct_state(output, expected):
    assert _parse_sacct_state(output) == expected


def test_chunk_to_wav_bytes_roundtrips_correctly():
    chunk = _make_chunk(duration_s=1.0, sr=16000)
    wav_bytes = _chunk_to_wav_bytes(chunk, sample_rate=16000)

    import io
    samples, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == 16000
    assert len(samples) == len(chunk.samples)


def test_biu_credentials_repr_hides_password():
    creds = BIUCredentials(host="biu.example.edu", username="shira", password="super-secret")
    assert "super-secret" not in repr(creds)
    assert "super-secret" not in str(creds)


# --- network calls, mocked ---

def _mock_exec_result(exit_status, stdout_text="", stderr_text=""):
    stdout = MagicMock()
    stdout.channel.recv_exit_status.return_value = exit_status
    stdout.read.return_value = stdout_text.encode()
    stderr = MagicMock()
    stderr.read.return_value = stderr_text.encode()
    return MagicMock(), stdout, stderr  # stdin, stdout, stderr


def test_submit_slurm_job_parses_id_from_mocked_ssh():
    ssh = MagicMock()
    ssh.exec_command.return_value = _mock_exec_result(0, stdout_text="Submitted batch job 42\n")

    job_id = submit_slurm_job(ssh, "pipeline_jobs/abc")
    assert job_id == "42"
    ssh.exec_command.assert_called_once()
    assert "sbatch job.slurm" in ssh.exec_command.call_args[0][0]


def test_submit_slurm_job_raises_on_nonzero_exit():
    ssh = MagicMock()
    ssh.exec_command.return_value = _mock_exec_result(1, stderr_text="sbatch: error: invalid partition")

    with pytest.raises(BIUJobError):
        submit_slurm_job(ssh, "pipeline_jobs/abc")


def test_poll_job_status_returns_parsed_state():
    ssh = MagicMock()
    ssh.exec_command.return_value = _mock_exec_result(0, stdout_text="RUNNING\n")

    status = poll_job_status(ssh, "42")
    assert status == JobStatus(slurm_job_id="42", state="RUNNING")


def test_upload_job_writes_wav_words_and_script_per_chunk():
    ssh = MagicMock()
    ssh.exec_command.return_value = _mock_exec_result(0)
    sftp = MagicMock()
    ssh.open_sftp.return_value = sftp

    chunks = [_make_chunk("chunk_0"), _make_chunk("chunk_1")]
    job_dir = upload_job(ssh, chunks, sample_rate=16000, email=None)

    assert job_dir.startswith(biu_sync.REMOTE_JOBS_DIR + "/")
    # one mkdir, and sftp.open called for: 2 wav + 2 words csv + 1 slurm script = 5 times
    assert sftp.open.call_count == 5
    sftp.close.assert_called_once()


def test_download_result_calls_sftp_get(tmp_path):
    ssh = MagicMock()
    sftp = MagicMock()
    ssh.open_sftp.return_value = sftp

    local_path = tmp_path / "result.csv"
    download_result(ssh, "pipeline_jobs/abc", local_path)

    sftp.get.assert_called_once_with("pipeline_jobs/abc/result.csv", str(local_path))
    sftp.close.assert_called_once()


def test_download_result_raises_biujoberror_when_missing(tmp_path):
    ssh = MagicMock()
    sftp = MagicMock()
    sftp.get.side_effect = FileNotFoundError()
    ssh.open_sftp.return_value = sftp

    with pytest.raises(BIUJobError):
        download_result(ssh, "pipeline_jobs/abc", tmp_path / "result.csv")


def test_download_logs_downloads_out_and_err_files(tmp_path):
    ssh = MagicMock()
    sftp = MagicMock()
    sftp.listdir.return_value = ["job_42.out", "job_42.err", "chunk_0.wav", "chunk_0_words.csv"]
    ssh.open_sftp.return_value = sftp

    downloaded = download_logs(ssh, "pipeline_jobs/abc", tmp_path)

    assert {p.name for p in downloaded} == {"job_42.out", "job_42.err"}
    assert sftp.get.call_count == 2  # only the log files, not the wav/csv
    sftp.close.assert_called_once()


def test_download_logs_returns_empty_list_on_error_instead_of_raising(tmp_path):
    ssh = MagicMock()
    ssh.open_sftp.side_effect = OSError("connection lost")

    downloaded = download_logs(ssh, "pipeline_jobs/abc", tmp_path)
    assert downloaded == []  # best-effort - must not raise and mask the real failure


def test_cleanup_job_runs_rm_rf():
    ssh = MagicMock()
    ssh.exec_command.return_value = _mock_exec_result(0)

    cleanup_job(ssh, "pipeline_jobs/abc")
    assert "rm -rf" in ssh.exec_command.call_args[0][0]
    assert "pipeline_jobs/abc" in ssh.exec_command.call_args[0][0]


# --- connection dropping mid-operation (not just a clean failure) ---
# These simulate the VPN/network dying partway through a step, not the
# remote command failing cleanly - paramiko/OSError raised directly from
# exec_command or an sftp call, rather than a non-zero exit status.

@pytest.mark.parametrize("dropped_exc", [OSError("Socket is closed"), paramiko.SSHException("EOF during negotiation")])
def test_run_command_wraps_connection_drop(dropped_exc):
    ssh = MagicMock()
    ssh.exec_command.side_effect = dropped_exc

    with pytest.raises(BIUJobError, match="Lost connection"):
        _run_command(ssh, "echo hi")


def test_upload_job_wraps_connection_drop_mid_transfer():
    ssh = MagicMock()
    ssh.exec_command.return_value = _mock_exec_result(0)  # mkdir succeeds
    sftp = MagicMock()
    ssh.open_sftp.return_value = sftp
    # First file write succeeds, the next one drops mid-transfer - doesn't
    # matter which exact file, just that a write failing partway through
    # the loop surfaces as a clear error instead of a raw traceback.
    sftp.open.return_value.__enter__.return_value.write.side_effect = [None, OSError("Connection reset by peer")]

    chunks = [_make_chunk("chunk_0"), _make_chunk("chunk_1")]
    with pytest.raises(BIUJobError, match="Connection to BIU was lost while uploading"):
        upload_job(ssh, chunks, sample_rate=16000, email=None)

    sftp.close.assert_called_once()  # still cleaned up despite the failure


def test_download_result_wraps_connection_drop(tmp_path):
    ssh = MagicMock()
    sftp = MagicMock()
    sftp.get.side_effect = paramiko.SSHException("Server connection dropped")
    ssh.open_sftp.return_value = sftp

    with pytest.raises(BIUJobError, match="Connection to BIU was lost while downloading"):
        download_result(ssh, "pipeline_jobs/abc", tmp_path / "result.csv")

    sftp.close.assert_called_once()


def test_download_result_still_distinguishes_missing_file_from_dropped_connection(tmp_path):
    """A missing result.csv (job failed before writing it) is a different,
    more specific message than a dropped connection - must not collapse
    into the same generic error."""
    ssh = MagicMock()
    sftp = MagicMock()
    sftp.get.side_effect = FileNotFoundError()
    ssh.open_sftp.return_value = sftp

    with pytest.raises(BIUJobError, match="Result file not found"):
        download_result(ssh, "pipeline_jobs/abc", tmp_path / "result.csv")


def test_poll_job_status_with_retry_succeeds_first_try(monkeypatch):
    calls = []
    monkeypatch.setattr(biu_sync, "poll_job_status", lambda ssh, job_id: calls.append(1) or JobStatus(job_id, "RUNNING"))

    status = poll_job_status_with_retry(MagicMock(), "42", retry_delay_s=0)
    assert status.state == "RUNNING"
    assert len(calls) == 1


def test_poll_job_status_with_retry_recovers_after_transient_failures(monkeypatch):
    attempts = iter([ConnectionError("blip"), ConnectionError("blip again"), JobStatus("42", "RUNNING")])

    def fake_poll(ssh, job_id):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(biu_sync, "poll_job_status", fake_poll)

    status = poll_job_status_with_retry(MagicMock(), "42", max_retries=3, retry_delay_s=0)
    assert status.state == "RUNNING"


def test_poll_job_status_with_retry_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(biu_sync, "poll_job_status", lambda ssh, job_id: (_ for _ in ()).throw(ConnectionError("dead")))

    with pytest.raises(BIUJobError):
        poll_job_status_with_retry(MagicMock(), "42", max_retries=2, retry_delay_s=0)


def test_poll_job_status_with_retry_does_not_swallow_legitimate_failed_state(monkeypatch):
    """A job that really failed returns a normal JobStatus, not an exception -
    retry logic must not mask or retry past that; it's the caller's job to act on it."""
    monkeypatch.setattr(biu_sync, "poll_job_status", lambda ssh, job_id: JobStatus(job_id, "FAILED"))

    status = poll_job_status_with_retry(MagicMock(), "42", retry_delay_s=0)
    assert status.state == "FAILED"


def test_connect_wraps_auth_failure(monkeypatch):
    import paramiko

    fake_client = MagicMock()
    fake_client.connect.side_effect = paramiko.AuthenticationException("bad password")
    monkeypatch.setattr(biu_sync.paramiko, "SSHClient", lambda: fake_client)

    creds = BIUCredentials(host="biu.example.edu", username="shira", password="wrong")
    with pytest.raises(BIUJobError):
        with connect(creds):
            pass


def test_connect_gives_a_human_message_for_a_bad_hostname(monkeypatch):
    """Regression test: a typo'd/unreachable hostname raises socket.gaierror,
    whose default message (e.g. "[Errno 11001] getaddrinfo failed" on Windows)
    is meaningless to someone who isn't reading a Python traceback - confirmed
    against the real, unmocked connect() against a nonexistent host."""
    import socket

    fake_client = MagicMock()
    fake_client.connect.side_effect = socket.gaierror(11001, "getaddrinfo failed")
    monkeypatch.setattr(biu_sync.paramiko, "SSHClient", lambda: fake_client)

    creds = BIUCredentials(host="typo-ed-hostname.invalid", username="shira", password="x")
    with pytest.raises(BIUJobError, match="Could not resolve server address"):
        with connect(creds):
            pass


def test_connect_gives_a_human_message_for_a_connection_timeout(monkeypatch):
    """A dead/unreachable host (e.g. not on the BIU VPN) times out rather than
    failing fast - the message should point at the VPN, not print a bare
    "timed out" with no next step."""
    fake_client = MagicMock()
    fake_client.connect.side_effect = TimeoutError("timed out")
    monkeypatch.setattr(biu_sync.paramiko, "SSHClient", lambda: fake_client)

    creds = BIUCredentials(host="slurm-login1.lnx.biu.ac.il", username="shira", password="x")
    with pytest.raises(BIUJobError, match="BIU VPN"):
        with connect(creds):
            pass


# --- run_biu_job orchestration (functions mocked, not paramiko directly) ---

def test_run_biu_job_happy_path(monkeypatch, tmp_path):
    fake_ssh = MagicMock()
    monkeypatch.setattr(biu_sync, "connect", lambda creds: _FakeConnectCtx(fake_ssh))
    monkeypatch.setattr(biu_sync, "upload_job", lambda ssh, chunks, sample_rate, email: "pipeline_jobs/job1")
    monkeypatch.setattr(biu_sync, "submit_slurm_job", lambda ssh, job_dir: "42")

    states = iter(["PENDING", "RUNNING", "COMPLETED"])
    monkeypatch.setattr(biu_sync, "poll_job_status", lambda ssh, job_id: JobStatus(job_id, next(states)))

    result_path = tmp_path / "result.csv"
    pd.DataFrame({"word": ["hi"]}).to_csv(result_path, index=False)
    monkeypatch.setattr(biu_sync, "download_result", lambda ssh, job_dir, local_path: None)

    cleanup_calls = []
    monkeypatch.setattr(biu_sync, "cleanup_job", lambda ssh, job_dir: cleanup_calls.append(job_dir))

    creds = BIUCredentials(host="biu.example.edu", username="shira", password="x")
    df = biu_sync.run_biu_job(creds, [_make_chunk()], result_path, poll_interval_s=0)

    assert list(df["word"]) == ["hi"]
    assert cleanup_calls == ["pipeline_jobs/job1"]


def test_run_biu_job_raises_and_still_cleans_up_on_failed_state(monkeypatch, tmp_path):
    fake_ssh = MagicMock()
    monkeypatch.setattr(biu_sync, "connect", lambda creds: _FakeConnectCtx(fake_ssh))
    monkeypatch.setattr(biu_sync, "upload_job", lambda ssh, chunks, sample_rate, email: "pipeline_jobs/job1")
    monkeypatch.setattr(biu_sync, "submit_slurm_job", lambda ssh, job_dir: "42")
    monkeypatch.setattr(biu_sync, "poll_job_status", lambda ssh, job_id: JobStatus(job_id, "FAILED"))

    cleanup_calls = []
    monkeypatch.setattr(biu_sync, "cleanup_job", lambda ssh, job_dir: cleanup_calls.append(job_dir))

    creds = BIUCredentials(host="biu.example.edu", username="shira", password="x")
    with pytest.raises(BIUJobError):
        biu_sync.run_biu_job(creds, [_make_chunk()], tmp_path / "result.csv", poll_interval_s=0)

    assert cleanup_calls == ["pipeline_jobs/job1"]  # cleanup still ran despite the failure


def test_run_biu_job_downloads_logs_before_cleanup_on_failure(monkeypatch, tmp_path):
    fake_ssh = MagicMock()
    monkeypatch.setattr(biu_sync, "connect", lambda creds: _FakeConnectCtx(fake_ssh))
    monkeypatch.setattr(biu_sync, "upload_job", lambda ssh, chunks, sample_rate, email: "pipeline_jobs/job1")
    monkeypatch.setattr(biu_sync, "submit_slurm_job", lambda ssh, job_dir: "42")
    monkeypatch.setattr(biu_sync, "poll_job_status_with_retry", lambda ssh, job_id: JobStatus(job_id, "FAILED"))

    call_order = []
    fake_log_paths = [tmp_path / "logs" / "job_42.out", tmp_path / "logs" / "job_42.err"]
    monkeypatch.setattr(
        biu_sync, "download_logs",
        lambda ssh, job_dir, local_dir: call_order.append("download_logs") or fake_log_paths,
    )
    monkeypatch.setattr(biu_sync, "cleanup_job", lambda ssh, job_dir: call_order.append("cleanup_job"))

    creds = BIUCredentials(host="biu.example.edu", username="shira", password="x")
    with pytest.raises(BIUJobError) as exc_info:
        biu_sync.run_biu_job(
            creds, [_make_chunk()], tmp_path / "result.csv",
            poll_interval_s=0, local_log_dir=tmp_path / "logs",
        )

    assert exc_info.value.log_paths == fake_log_paths
    assert call_order == ["download_logs", "cleanup_job"]  # logs fetched before the dir is wiped


def test_run_biu_job_skips_log_download_when_no_local_log_dir_given(monkeypatch, tmp_path):
    fake_ssh = MagicMock()
    monkeypatch.setattr(biu_sync, "connect", lambda creds: _FakeConnectCtx(fake_ssh))
    monkeypatch.setattr(biu_sync, "upload_job", lambda ssh, chunks, sample_rate, email: "pipeline_jobs/job1")
    monkeypatch.setattr(biu_sync, "submit_slurm_job", lambda ssh, job_dir: "42")
    monkeypatch.setattr(biu_sync, "poll_job_status_with_retry", lambda ssh, job_id: JobStatus(job_id, "FAILED"))
    monkeypatch.setattr(biu_sync, "cleanup_job", lambda ssh, job_dir: None)

    download_logs_calls = []
    monkeypatch.setattr(biu_sync, "download_logs", lambda *a, **kw: download_logs_calls.append(1))

    creds = BIUCredentials(host="biu.example.edu", username="shira", password="x")
    with pytest.raises(BIUJobError) as exc_info:
        biu_sync.run_biu_job(creds, [_make_chunk()], tmp_path / "result.csv", poll_interval_s=0)  # no local_log_dir

    assert download_logs_calls == []
    assert exc_info.value.log_paths == []


def test_run_biu_job_cleanup_drop_does_not_mask_the_real_failure(monkeypatch, tmp_path):
    """If the connection dies right after a SLURM failure (so cleanup_job
    also fails), the user should still see the SLURM failure message -
    not a less useful "lost connection during cleanup" one instead."""
    fake_ssh = MagicMock()
    monkeypatch.setattr(biu_sync, "connect", lambda creds: _FakeConnectCtx(fake_ssh))
    monkeypatch.setattr(biu_sync, "upload_job", lambda ssh, chunks, sample_rate, email: "pipeline_jobs/job1")
    monkeypatch.setattr(biu_sync, "submit_slurm_job", lambda ssh, job_dir: "42")
    monkeypatch.setattr(biu_sync, "poll_job_status_with_retry", lambda ssh, job_id: JobStatus(job_id, "FAILED"))
    monkeypatch.setattr(biu_sync, "download_logs", lambda ssh, job_dir, local_dir: [])

    def _cleanup_also_drops(ssh, job_dir):
        raise BIUJobError("Lost connection to BIU while running a remote command: Socket is closed")

    monkeypatch.setattr(biu_sync, "cleanup_job", _cleanup_also_drops)

    creds = BIUCredentials(host="biu.example.edu", username="shira", password="x")
    with pytest.raises(BIUJobError, match="ended with state FAILED"):
        biu_sync.run_biu_job(creds, [_make_chunk()], tmp_path / "result.csv", poll_interval_s=0)


def test_run_biu_job_cleanup_drop_does_not_break_the_happy_path(monkeypatch, tmp_path):
    """Same idea on success: if cleanup fails after a completed job (e.g.
    the connection drops right after the result was already downloaded),
    the job should still be reported as a success - the result is already
    safely on disk."""
    fake_ssh = MagicMock()
    monkeypatch.setattr(biu_sync, "connect", lambda creds: _FakeConnectCtx(fake_ssh))
    monkeypatch.setattr(biu_sync, "upload_job", lambda ssh, chunks, sample_rate, email: "pipeline_jobs/job1")
    monkeypatch.setattr(biu_sync, "submit_slurm_job", lambda ssh, job_dir: "42")
    monkeypatch.setattr(biu_sync, "poll_job_status_with_retry", lambda ssh, job_id: JobStatus(job_id, "COMPLETED"))

    result_path = tmp_path / "result.csv"
    pd.DataFrame({"word": ["hi"]}).to_csv(result_path, index=False)
    monkeypatch.setattr(biu_sync, "download_result", lambda ssh, job_dir, local_path: None)

    def _cleanup_also_drops(ssh, job_dir):
        raise BIUJobError("Lost connection to BIU while running a remote command: Socket is closed")

    monkeypatch.setattr(biu_sync, "cleanup_job", _cleanup_also_drops)

    creds = BIUCredentials(host="biu.example.edu", username="shira", password="x")
    df = biu_sync.run_biu_job(creds, [_make_chunk()], result_path, poll_interval_s=0)

    assert list(df["word"]) == ["hi"]  # no exception - the job still succeeded


class _FakeConnectCtx:
    """Minimal stand-in for the `connect()` context manager in tests."""

    def __init__(self, ssh):
        self._ssh = ssh

    def __enter__(self):
        return self._ssh

    def __exit__(self, *exc_info):
        return False
