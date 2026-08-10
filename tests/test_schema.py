import pandas as pd
import pytest

from pipeline import schema


def _valid_output_table():
    return pd.DataFrame({
        "word": ["hi", "there"],
        "start_s": [0.0, 0.5],
        "end_s": [0.4, 0.9],
        "boundary_probability": [0.1, 0.8],
        "boundary_prediction": [0, 1],
        "threshold": [0.3, 0.3],
        "chunk_id": ["chunk_0", "chunk_0"],
    })


def test_new_output_table_has_correct_columns():
    assert list(schema.new_output_table().columns) == list(schema.OUTPUT_COLUMNS)


def test_validate_output_schema_accepts_valid_table():
    schema.validate_output_schema(_valid_output_table())  # should not raise


def test_validate_output_schema_rejects_out_of_range_probability():
    df = _valid_output_table()
    df["boundary_probability"] = [1.5, 0.8]
    with pytest.raises(schema.SchemaError):
        schema.validate_output_schema(df)


def test_validate_output_schema_rejects_invalid_prediction_value():
    df = _valid_output_table()
    df["boundary_prediction"] = [0, 2]
    with pytest.raises(schema.SchemaError):
        schema.validate_output_schema(df)


def test_validate_output_schema_rejects_missing_column():
    df = _valid_output_table().drop(columns=["chunk_id"])
    with pytest.raises(schema.SchemaError):
        schema.validate_output_schema(df)


def test_validate_output_schema_rejects_end_before_start():
    df = _valid_output_table()
    df.loc[0, ["start_s", "end_s"]] = [1.0, 0.5]
    with pytest.raises(schema.SchemaError):
        schema.validate_output_schema(df)
