"""Upload UI for the boundary-detection pipeline."""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the `pipeline` package

from pipeline_runner import (
    BIUJobError,
    PipelineError,
    check_pending_job_status,
    discard_pending_job_by_dir,
    find_pending_job,
    recover_pending_job_result,
    run_pipeline,
    test_connection,
)

st.set_page_config(page_title="Prosodic Boundary Detection", layout="centered")

css_path = Path(__file__).resolve().parent / "style.css"
st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Quiet corner accents used on every screen after the intro (connect,
# upload, and - since results render further down the same upload screen -
# the results view too).
_CORNER_GLOW = '<div class="corner-glow-tl"></div><div class="corner-glow-br"></div>'

# Decorative brain icon: a node-and-edge network traced in the shape of a
# brain in profile (forehead front-left, occipital lobe back-right,
# cerebellum + brainstem stub at the bottom-back). Purely visual data -
# doesn't touch anything below it.
_BRAIN_OUTLINE = [
    (55, 20), (80, 13), (105, 11), (130, 13), (155, 18), (175, 28),
    (190, 45), (198, 65), (196, 85), (186, 102), (172, 113), (178, 124),
    (168, 136), (155, 148), (140, 140), (122, 136), (100, 132), (80, 124),
    (65, 112), (50, 98), (38, 80), (33, 60), (38, 40), (46, 27),
]
_BRAIN_INTERIOR = [
    (80, 45), (105, 40), (130, 42), (150, 50), (165, 65), (175, 80),
    (160, 90), (140, 75), (115, 70), (90, 72), (70, 68), (60, 85),
    (80, 95), (100, 100), (120, 105), (140, 110), (155, 105), (90, 55), (130, 60),
]
_BRAIN_NODES = _BRAIN_OUTLINE + _BRAIN_INTERIOR
_N_OUTLINE = len(_BRAIN_OUTLINE)
_BRAIN_EDGES = (
    [(i, (i + 1) % _N_OUTLINE) for i in range(_N_OUTLINE)]  # closed outline loop
    + [(24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
       (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
       (38, 39), (39, 40), (40, 30), (32, 41), (25, 41), (41, 24), (26, 42),
       (42, 31), (33, 41)]  # interior mesh
    + [(1, 25), (3, 26), (5, 28), (7, 29), (9, 30), (11, 39), (13, 40),
       (15, 38), (17, 36), (19, 35), (21, 34), (23, 41)]  # outline reaching in to nearby interior nodes
)

# A handful of nodes drawn larger, as visual "hubs" - matches the
# big-dot/small-dot hierarchy in the reference image rather than
# every node being the same size. Each hub also drifts a couple of pixels
# and settles back on its own loop (dx, dy, delay_s) - varied per node so
# they don't all move in lockstep.
_HUB_NODES = {
    0: (3, -3, 0.0), 7: (-3, 3, 0.6), 12: (4, 2, 1.2), 18: (-4, -2, 0.3),
    21: (2, 4, 1.8), 30: (-3, -4, 0.9), 32: (4, -2, 1.5), 38: (-2, 3, 2.1),
}


def _brain_svg() -> str:
    """Build the brain network icon as inline SVG markup from the node/edge data above."""
    lines = "".join(
        f'<line class="edge" x1="{_BRAIN_NODES[a][0]}" y1="{_BRAIN_NODES[a][1]}" '
        f'x2="{_BRAIN_NODES[b][0]}" y2="{_BRAIN_NODES[b][1]}"/>'
        for a, b in _BRAIN_EDGES
    )

    def _dot(i: int, x: float, y: float) -> str:
        if i in _HUB_NODES:
            dx, dy, delay = _HUB_NODES[i]
            style = f"--dx:{dx}px;--dy:{dy}px;--delay:{delay}s"
            return f'<circle class="node hub-node" cx="{x}" cy="{y}" r="5.5" style="{style}"/>'
        return f'<circle class="node" cx="{x}" cy="{y}" r="2"/>'

    dots = "".join(_dot(i, x, y) for i, (x, y) in enumerate(_BRAIN_NODES))
    # Deliberately a single line with no leading whitespace: Streamlit's
    # markdown renderer treats indented lines as a code block, which would
    # otherwise break this SVG (and everything rendered after it).
    return (
        '<svg class="brain-icon" viewBox="0 0 220 170" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="brainGradient" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="220" y2="170">'
        '<stop offset="0%" stop-color="#6c5ce7"/><stop offset="50%" stop-color="#3b82f6"/>'
        '<stop offset="100%" stop-color="#10b981"/></linearGradient></defs>'
        f"{lines}{dots}</svg>"
    )

# Which of the three screens (intro / connect / upload) is currently shown.
# Pure display state - has no effect on how the pipeline itself runs.
if "screen" not in st.session_state:
    st.session_state.screen = "intro"

# Streamlit drops a widget's session_state entry once that widget stops
# being drawn on screen - so the connect screen's text_input values (keyed
# "biu_address" etc.) disappear the moment we navigate to the upload screen,
# every time, not just on a dropped session. We copy them into these plain
# (non-widget) keys instead, which persist for the rest of the session.
st.session_state.setdefault("biu_conn_address", "")
st.session_state.setdefault("biu_conn_password", "")
st.session_state.setdefault("biu_conn_notify_email", "")


def _parse_biu_address(address: str):
    """Split a single "username@host" field into its two parts.

    Input: address - raw text from the combined server field, e.g.
        "agmonlab@slurm-login1.lnx.biu.ac.il".
    Output: (username, host) tuple, or (None, None) if the text isn't in
        that shape yet (still typing, missing "@", or a part is empty).
    """
    username, sep, host = address.partition("@")
    if not sep or not username or not host:
        return None, None
    return username, host


def _go(screen_name: str) -> None:
    st.session_state.screen = screen_name


def _trim_for_display(results_df):
    """Drop columns that are internal details, not useful to a person looking
    at results: threshold is the same constant value on every row, and
    chunk_id is an implementation detail of how the audio was split. Both
    stay in the schema QC validates on the BIU side - only the copy shown/
    downloaded here is trimmed. Shared by the normal Run flow and by
    recovering a pending job, so both show the same columns.

    Input: results_df - the full result table as returned by the pipeline.
    Output: a copy with 'threshold' and 'chunk_id' removed, if present.
    """
    return results_df.drop(columns=["threshold", "chunk_id"], errors="ignore")


# ---------------------------------------------------------------- intro ---
if st.session_state.screen == "intro":
    st.markdown(
        f"""
        <div class="hero screen">
          <div class="hero-backdrop"></div>
          <div class="orbit-wrap">
            {_brain_svg()}
          </div>
          <h1 class="hero-title">Welcome.</h1>
          <p class="hero-sub">Prosodic Boundary Detection &mdash; mapping where sentences end.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1, 1])
    with center:
        st.button("Start", type="primary", use_container_width=True, on_click=_go, args=("connect",))

# -------------------------------------------------------------- connect ---
elif st.session_state.screen == "connect":
    st.markdown(_CORNER_GLOW, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="screen">
          <p class="section-label">Step 01</p>
          <h2 class="screen-title">Connect to the server.</h2>
          <p class="screen-sub">Enter your BIU lab account details - used to open the connection once you run a job.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    biu_address = st.text_input(
        "Server (username@host)",
        key="biu_address",
        help="Your BIU lab username and the cluster address together, separated by '@'.",
    )
    st.session_state.biu_conn_address = biu_address
    biu_username, biu_host = _parse_biu_address(biu_address)
    biu_password = st.text_input("Lab password", type="password", key="biu_password")
    st.session_state.biu_conn_password = biu_password
    notify_email = st.text_input(
        "Notification email (optional)",
        key="notify_email",
        help="If you enter an email here, the BIU server will send you a message when the job starts, finishes, or fails. Leave blank to skip.",
    )
    st.session_state.biu_conn_notify_email = notify_email

    col_back, col_test = st.columns([1, 1])
    with col_back:
        st.button("Back", on_click=_go, args=("intro",))
    with col_test:
        ready = bool(biu_host and biu_username and biu_password)  # notify_email is optional
        test_clicked = st.button("Test connection", type="primary", disabled=not ready, use_container_width=True)

    # Open a real SSH connection now (and close it right away) so a wrong
    # host/username/password is caught here - before the user spends time
    # uploading files - instead of only failing much later during Run.
    if test_clicked:
        try:
            with st.spinner("Connecting to the BIU server..."):
                test_connection(biu_host, biu_username, biu_password)
        except BIUJobError as exc:
            st.session_state.connection_ok = False
            st.error(f"Could not connect: {exc}")
        else:
            st.session_state.connection_ok = True

    # Also require the fields to still be filled in - connection_ok alone
    # would otherwise keep showing "Connected" even after the fields above
    # come back empty (e.g. after a page reload reset the widgets).
    if st.session_state.get("connection_ok") and ready:
        st.success("Connected to the server successfully.")

        # A job from an earlier session that we lost track of (e.g. the
        # connection dropped mid-poll) before its own cleanup could run -
        # see record_pending_job/find_pending_jobs in pipeline/biu_sync.py.
        # Offer to check on it and either fetch its results or delete it
        # from the shared server, instead of leaving it there forever.
        pending_job = find_pending_job(biu_host, biu_username)
        if pending_job:
            st.warning(
                f"Found a job from an earlier session (started {pending_job['created_at']}) "
                f"that we lost track of - possibly a dropped connection."
            )
            col_check, col_discard = st.columns(2)
            with col_check:
                check_clicked = st.button("Check status", use_container_width=True, key="check_pending")
            with col_discard:
                discard_clicked = st.button(
                    "Discard (delete from server)", use_container_width=True, key="discard_pending"
                )

            if check_clicked:
                try:
                    with st.spinner("Checking job status..."):
                        status = check_pending_job_status(
                            biu_host, biu_username, biu_password, pending_job["slurm_job_id"]
                        )
                except BIUJobError as exc:
                    st.error(f"Could not check job status: {exc}")
                else:
                    if status.state == "COMPLETED":
                        st.session_state.pending_job_completed = True
                    else:
                        st.session_state.pending_job_completed = False
                        if status.state in {
                            "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
                        }:
                            st.error(f"That job ended with state {status.state}.")
                        else:
                            st.info(f"Still {status.state.lower()} - check back again in a bit.")

            if discard_clicked:
                try:
                    with st.spinner("Deleting from the server..."):
                        discard_pending_job_by_dir(biu_host, biu_username, biu_password, pending_job["job_dir"])
                except BIUJobError as exc:
                    st.error(f"Could not delete: {exc}")
                else:
                    st.session_state.pending_job_completed = False
                    st.success("Deleted from the server.")

            if st.session_state.get("pending_job_completed"):
                st.success("That job finished - fetch its results?")
                if st.button("Fetch results", key="fetch_pending"):
                    try:
                        with st.spinner("Downloading results..."):
                            result_path = Path(tempfile.mkdtemp()) / "result.csv"
                            recovered_df = recover_pending_job_result(
                                biu_host, biu_username, biu_password, pending_job["job_dir"], result_path
                            )
                    except BIUJobError as exc:
                        st.error(f"Could not fetch results: {exc}")
                    else:
                        st.session_state.pending_job_completed = False
                        display_df = _trim_for_display(recovered_df)
                        st.success(f"Done - {len(display_df)} words processed.")
                        st.dataframe(display_df, use_container_width=True)
                        st.download_button(
                            "Download results (CSV)",
                            data=display_df.to_csv(index=False),
                            file_name="boundary_predictions.csv",
                            mime="text/csv",
                            key="download_pending",
                        )

        st.button(
            "Continue to upload →",
            type="primary",
            use_container_width=True,
            on_click=_go,
            args=("upload",),
        )

# --------------------------------------------------------------- upload ---
elif st.session_state.screen == "upload":
    st.markdown(_CORNER_GLOW, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="screen">
          <p class="section-label">Step 02</p>
          <h2 class="screen-title">Upload your files.</h2>
          <p class="screen-sub">An audio file and its word-level transcript are both required.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    audio_file = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
    transcript_file = st.file_uploader("Transcript CSV (word, start_s, end_s)", type=["csv"])

    st.button("Back", on_click=_go, args=("connect",))

    # Run stays disabled until every required field is filled in, and again
    # once a run is already in flight - a fast double-click can otherwise
    # land as two separate script runs (Streamlit interrupts the first run's
    # UI updates on a new interaction, but the already-started Python call
    # keeps executing), each submitting its own SLURM job for the same
    # upload. run_in_progress is set below before the slow part starts, so
    # the second click's run sees it immediately and refuses instead of
    # submitting a duplicate job.
    #
    # run_in_progress only lives in this browser session's memory, though -
    # it says nothing about whether a job is really running, and a page
    # reload wipes it clean regardless. find_pending_job checks the actual
    # persistent record instead (pipeline/biu_sync.py's .pending_jobs.json,
    # written the moment a job is submitted and only cleared once its
    # cleanup succeeds) - that one survives a reload, and is what "unlock"
    # below must not bypass.
    run_username, run_host = _parse_biu_address(st.session_state.get("biu_conn_address", ""))
    pending_job = find_pending_job(run_host, run_username) if run_host and run_username else None

    already_running = st.session_state.get("run_in_progress", False)
    blocked_by_pending = bool(pending_job) and not already_running
    ready_to_run = bool(audio_file and transcript_file) and not already_running and not blocked_by_pending
    run_clicked = st.button("Run", type="primary", disabled=not ready_to_run, use_container_width=True)

    if already_running:
        st.info("A run is already in progress - please wait for it to finish.")
        # The flag above only gets cleared by the run that set it, once it
        # finishes. If that run was interrupted before reaching its own
        # cleanup (a dropped connection, a page reload mid-run, etc.), it
        # never clears - leaving Run permanently disabled with nothing
        # actually running. This unlocks the *session's* flag only - if a
        # job is still genuinely outstanding, blocked_by_pending (checked
        # again on the very next rerun, since it re-reads the persistent
        # record) keeps Run disabled regardless.
        if st.button("Nothing is actually running - unlock Run", key="force_unlock_run"):
            st.session_state.run_in_progress = False
            st.rerun()
    elif blocked_by_pending:
        st.warning(
            f"There's a job from an earlier session (started {pending_job['created_at']}) "
            f"that hasn't been resolved yet. Go back to Connect to check its status or "
            f"discard it before starting a new run."
        )
        st.button("Back to connect", on_click=_go, args=("connect",), key="back_to_connect_from_upload")

    if run_clicked and not already_running and not blocked_by_pending:
        st.session_state.run_in_progress = True
        try:
            run_password = st.session_state.get("biu_conn_password", "")

            if not (run_host and run_username and run_password):
                st.error(
                    "Your connection details are missing. Please go back and connect to "
                    "the server again."
                )
                st.button("Back to connect", on_click=_go, args=("connect",))
            else:
                # Streamlit's uploaded files live in memory; write them to a temp
                # folder so the pipeline functions (which expect file paths) can read them.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_dir = Path(tmp_dir)
                    audio_path = tmp_dir / audio_file.name
                    transcript_path = tmp_dir / transcript_file.name
                    audio_path.write_bytes(audio_file.getvalue())
                    transcript_path.write_bytes(transcript_file.getvalue())

                    result_path = tmp_dir / "result.csv"
                    log_dir = tmp_dir / "logs"

                    try:
                        with st.spinner("Running pipeline - this can take a while on a busy SLURM queue..."):
                            results_df = run_pipeline(
                                audio_path,
                                transcript_path,
                                host=run_host,
                                username=run_username,
                                password=run_password,
                                local_result_path=result_path,
                                notify_email=st.session_state.get("biu_conn_notify_email") or None,
                                local_log_dir=log_dir,
                            )
                    except PipelineError as exc:
                        st.error(f"Failed at stage '{exc.stage}': {exc}")
                        for log_path in exc.log_paths:
                            with st.expander(f"Log: {log_path.name}"):
                                st.code(log_path.read_text(errors="replace"))
                    else:
                        display_df = _trim_for_display(results_df)
                        st.success(f"Done - {len(display_df)} words processed.")
                        st.dataframe(display_df, use_container_width=True)
                        st.download_button(
                            "Download results (CSV)",
                            data=display_df.to_csv(index=False),
                            file_name="boundary_predictions.csv",
                            mime="text/csv",
                        )
        finally:
            st.session_state.run_in_progress = False
