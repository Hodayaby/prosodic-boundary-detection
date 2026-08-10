"""Canonical output schema for the boundary-detection pipeline.

Every pipeline stage that produces or consumes the final result table must
use exactly these columns. boundary_probability is the raw model output
and must never be dropped in favor of only the thresholded decision.
"""

import pandas as pd

# The exact column order every output table must follow.
OUTPUT_COLUMNS = (
    "word",
    "start_s",
    "end_s",
    "boundary_probability",
    "boundary_prediction",
    "threshold",
    "chunk_id",
)

# The expected data type for each column above.
OUTPUT_DTYPES = {
    "word": "object",
    "start_s": "float64",
    "end_s": "float64",
    "boundary_probability": "float64",
    "boundary_prediction": "int64",
    "threshold": "float64",
    "chunk_id": "object",
}


class SchemaError(ValueError):
    """Raised when a DataFrame doesn't match the pipeline output schema."""


def validate_output_schema(df: pd.DataFrame) -> None:
    """Check that df has the required columns and that value ranges make sense.

    Input: df - a result table to check.
    Output: None. Raises SchemaError if df doesn't match the expected schema.
    """
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Output table is missing required columns: {missing}")

    if df["end_s"].lt(df["start_s"]).any():
        raise SchemaError("Output table has rows where end_s < start_s")

    if not df["boundary_probability"].between(0.0, 1.0).all():
        raise SchemaError("boundary_probability must be within [0, 1] for all rows")

    if not df["boundary_prediction"].isin([0, 1]).all():
        raise SchemaError("boundary_prediction must be 0 or 1 for all rows")


def new_output_table() -> pd.DataFrame:
    """An empty DataFrame with the correct columns and dtypes, ready to append rows to.

    Input: none.
    Output: empty DataFrame with OUTPUT_COLUMNS in order and OUTPUT_DTYPES applied.
    """
    empty = {col: pd.Series(dtype=dtype) for col, dtype in OUTPUT_DTYPES.items()}
    return pd.DataFrame(empty)[list(OUTPUT_COLUMNS)]
