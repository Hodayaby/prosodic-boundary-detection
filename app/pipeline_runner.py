"""Runs the full pipeline end to end for one uploaded job."""

import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the `pipeline` package

from pipeline.input_validation import validate_audio, validate_transcript_csv
from pipeline.audio_preprocessing import preprocess_audio
from pipeline.schema import OUTPUT_COLUMNS, validate_output_schema
from server_connection import run_on_server


def run_pipeline(
    audio_path: Union[str, Path],
    transcript_csv_path: Union[str, Path],
    credentials: Optional[dict] = None,
) -> pd.DataFrame:
    """Validate, preprocess, run the model, and return the final output table.

    credentials (email/password for the BIU lab account) is threaded through
    to run_on_server() for when the real SSH connection is built - the stub
    ignores it today.

    Raises InputValidationError (from pipeline.input_validation) if the
    inputs are invalid - callers should catch that and show it to the user.
    """
    # Step 1: fail fast on a bad file before doing any real work.
    validate_audio(audio_path)
    transcript_df = validate_transcript_csv(transcript_csv_path)

    # Step 2: match the audio format the model was trained on.
    preprocess_audio(audio_path)  # resamples to 16kHz mono; not yet consumed further

    # Step 3: the actual prediction, currently a stand-in for the real server call.
    predictions_df = run_on_server(transcript_df, credentials)

    # Step 4: combine the transcript with its predictions into the final table.
    output_df = pd.concat([transcript_df.reset_index(drop=True), predictions_df], axis=1)
    output_df = output_df[list(OUTPUT_COLUMNS)]

    validate_output_schema(output_df)
    return output_df
