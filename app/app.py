"""Upload UI for the boundary-detection pipeline."""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the `pipeline` package

from pipeline_runner import PipelineError, run_pipeline

st.set_page_config(page_title="Prosodic Boundary Detection", layout="centered")

st.title("Prosodic Boundary Detection")
st.caption("Upload an audio file and its word-level transcript to detect sentence boundaries.")

# Step 1: BIU login. Used to open the real SSH connection when Run is clicked.
st.subheader("1. Connect to the BIU lab server")
biu_host = st.text_input("Server host (SSH address)")
biu_email = st.text_input("Lab email")
biu_password = st.text_input("Lab password", type="password")

# Step 2: the two files every pipeline job needs.
st.subheader("2. Upload input")
audio_file = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
transcript_file = st.file_uploader("Transcript CSV (word, start_s, end_s)", type=["csv"])

# Run stays disabled until every required field is filled in.
ready_to_run = bool(biu_host and biu_email and biu_password and audio_file and transcript_file)
run_clicked = st.button("Run", type="primary", disabled=not ready_to_run)

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
                    host=biu_host,
                    username=biu_email,
                    password=biu_password,
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
