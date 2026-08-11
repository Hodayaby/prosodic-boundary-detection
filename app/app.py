"""Upload UI for the boundary-detection pipeline."""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the `pipeline` package

from pipeline_runner import PipelineError, run_pipeline

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


def _go(screen_name: str) -> None:
    st.session_state.screen = screen_name


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
    biu_host = st.text_input("Server host (SSH address)", key="biu_host")
    biu_email = st.text_input("Lab email", key="biu_email")
    biu_password = st.text_input("Lab password", type="password", key="biu_password")

    col_back, col_next = st.columns([1, 1])
    with col_back:
        st.button("Back", on_click=_go, args=("intro",))
    with col_next:
        ready = bool(biu_host and biu_email and biu_password)
        st.button("Continue", type="primary", disabled=not ready, use_container_width=True, on_click=_go, args=("upload",))

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

    # Run stays disabled until every required field is filled in.
    ready_to_run = bool(audio_file and transcript_file)
    run_clicked = st.button("Run", type="primary", disabled=not ready_to_run, use_container_width=True)

    if run_clicked:
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
                        host=st.session_state.biu_host,
                        username=st.session_state.biu_email,
                        password=st.session_state.biu_password,
                        local_result_path=result_path,
                        local_log_dir=log_dir,
                    )
            except PipelineError as exc:
                st.error(f"Failed at stage '{exc.stage}': {exc}")
                for log_path in exc.log_paths:
                    with st.expander(f"Log: {log_path.name}"):
                        st.code(log_path.read_text(errors="replace"))
            else:
                st.success(f"Done - {len(results_df)} words processed.")
                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "Download results (CSV)",
                    data=results_df.to_csv(index=False),
                    file_name="boundary_predictions.csv",
                    mime="text/csv",
                )
