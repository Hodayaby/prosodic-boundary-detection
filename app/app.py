"""Upload UI for the boundary-detection pipeline."""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the `pipeline` package

from pipeline.input_validation import InputValidationError
from pipeline_runner import run_pipeline

st.set_page_config(page_title="Prosodic Boundary Detection", layout="centered")

st.title("Prosodic Boundary Detection")
st.caption("Upload an audio file and its word-level transcript to detect sentence boundaries.")

st.subheader("1. Connect to the BIU lab server")
st.caption(
    "Not wired up yet - the pipeline currently runs with placeholder data regardless "
    "of what's entered here. This is just the login UI, ready for when the real "
    "server connection (SSH) is built."
)
biu_email = st.text_input("Lab email")
biu_password = st.text_input("Lab password", type="password")
credentials = {"email": biu_email, "password": biu_password}

st.subheader("2. Upload input")
audio_file = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
transcript_file = st.file_uploader("Transcript CSV (word, start_s, end_s)", type=["csv"])

ready_to_run = bool(biu_email and biu_password and audio_file and transcript_file)
run_clicked = st.button("Run", type="primary", disabled=not ready_to_run)

if run_clicked:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        audio_path = tmp_dir / audio_file.name
        transcript_path = tmp_dir / transcript_file.name
        audio_path.write_bytes(audio_file.getvalue())
        transcript_path.write_bytes(transcript_file.getvalue())

        try:
            with st.spinner("Running pipeline..."):
                results_df = run_pipeline(audio_path, transcript_path, credentials)
        except InputValidationError as exc:
            st.error(str(exc))
        else:
            st.success(f"Done - {len(results_df)} words processed.")
            st.dataframe(results_df, use_container_width=True)
            st.download_button(
                "Download results (CSV)",
                data=results_df.to_csv(index=False),
                file_name="boundary_predictions.csv",
                mime="text/csv",
            )
