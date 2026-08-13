from unittest.mock import MagicMock

import pandas as pd
import pytest

import run_pipeline_job
from pipeline.schema import SchemaError
from run_pipeline_job import (
    _chunk_sort_key,
    apply_threshold,
    classify_chunk,
    find_chunk_ids,
    quality_check,
)


# --- pure logic ---

def test_chunk_sort_key_orders_numerically_not_lexicographically():
    chunk_ids = ["chunk_10", "chunk_2", "chunk_1", "chunk_0"]
    assert sorted(chunk_ids, key=_chunk_sort_key) == ["chunk_0", "chunk_1", "chunk_2", "chunk_10"]


def test_find_chunk_ids_reads_words_csv_files_in_numeric_order(tmp_path):
    for name in ["chunk_10_words.csv", "chunk_2_words.csv", "chunk_0_words.csv"]:
        (tmp_path / name).write_text("word,start_s,end_s\n")

    assert find_chunk_ids(tmp_path) == ["chunk_0", "chunk_2", "chunk_10"]


def _merged_df():
    return pd.DataFrame({
        "word": ["hi", "there", "friend"],
        "start_s": [0.0, 0.3, 0.6],
        "end_s": [0.3, 0.6, 0.9],
        "boundary_probability": [0.1, 0.6, 0.9],
        "chunk_id": ["chunk_0", "chunk_0", "chunk_1"],
    })


def test_apply_threshold_produces_correct_columns_and_predictions():
    result = apply_threshold(_merged_df(), threshold=0.5)

    assert list(result.columns) == ["word", "start_s", "end_s", "boundary_probability", "boundary_prediction", "threshold", "chunk_id"]
    assert result["boundary_prediction"].tolist() == [0, 1, 1]
    assert (result["threshold"] == 0.5).all()


def test_quality_check_accepts_well_formed_output():
    result = apply_threshold(_merged_df(), threshold=0.5)
    quality_check(result)  # should not raise


def test_quality_check_rejects_missing_word():
    result = apply_threshold(_merged_df(), threshold=0.5)
    result.loc[0, "word"] = None
    with pytest.raises(ValueError):
        quality_check(result)


def test_quality_check_rejects_duplicate_rows():
    result = apply_threshold(_merged_df(), threshold=0.5)
    duplicated = pd.concat([result, result.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        quality_check(duplicated)


def test_quality_check_rejects_suspiciously_high_boundary_rate():
    df = _merged_df()
    df["boundary_probability"] = [0.95, 0.96, 0.97]  # everything predicted as a boundary
    result = apply_threshold(df, threshold=0.5)
    with pytest.raises(ValueError):
        quality_check(result)


def test_quality_check_rejects_out_of_range_probability():
    result = apply_threshold(_merged_df(), threshold=0.5)
    result.loc[0, "boundary_probability"] = 1.5
    with pytest.raises(SchemaError):
        quality_check(result)


# --- classify_chunk / main, with the model mocked out ---

def test_classify_chunk_uses_prob_1_as_boundary_probability(monkeypatch, tmp_path):
    words_df = pd.DataFrame({"word": ["a", "b"], "start_s": [0.0, 0.5], "end_s": [0.5, 1.0]})

    fake_predictions = [
        {"word_index": 0, "word": "a", "pred_label": "0", "prob_0": 0.9, "prob_1": 0.1},
        {"word_index": 1, "word": "b", "pred_label": "1", "prob_0": 0.2, "prob_1": 0.8},
    ]
    monkeypatch.setattr(run_pipeline_job, "predict_labels_for_words", lambda *a, **kw: fake_predictions)

    result = classify_chunk(MagicMock(), MagicMock(), "cpu", tmp_path / "chunk_0.wav", words_df, "chunk_0")

    assert result["boundary_probability"].tolist() == [0.1, 0.8]
    assert (result["chunk_id"] == "chunk_0").all()
    assert result["word"].tolist() == ["a", "b"]


def test_main_end_to_end_with_mocked_model(monkeypatch, tmp_path, capsys):
    job_dir = tmp_path
    pd.DataFrame({"word": ["a", "b"], "start_s": [0.0, 0.5], "end_s": [0.5, 1.0]}).to_csv(job_dir / "chunk_0_words.csv", index=False)
    pd.DataFrame({"word": ["c"], "start_s": [1.0], "end_s": [1.5]}).to_csv(job_dir / "chunk_1_words.csv", index=False)
    (job_dir / "chunk_0.wav").write_bytes(b"")
    (job_dir / "chunk_1.wav").write_bytes(b"")

    monkeypatch.setattr(run_pipeline_job, "WhisperProcessor", MagicMock(from_pretrained=MagicMock(return_value=MagicMock())))
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    monkeypatch.setattr(run_pipeline_job, "WhisperForConditionalGeneration", MagicMock(from_pretrained=MagicMock(return_value=fake_model)))

    def fake_classify_chunk(model, processor, device, wav_path, words_df, chunk_id):
        result = words_df[["word", "start_s", "end_s"]].copy()
        result["boundary_probability"] = 0.9 if chunk_id == "chunk_1" else 0.1
        result["chunk_id"] = chunk_id
        return result

    monkeypatch.setattr(run_pipeline_job, "classify_chunk", fake_classify_chunk)
    monkeypatch.setattr(run_pipeline_job, "parse_args", lambda: MagicMock(job_dir=str(job_dir), model_dir="fake-model", threshold=0.5))

    run_pipeline_job.main()

    result = pd.read_csv(job_dir / "result.csv")
    assert list(result["word"]) == ["a", "b", "c"]  # chunk_0 before chunk_1, numeric order preserved
    assert list(result["boundary_prediction"]) == [0, 0, 1]


def test_main_exits_with_error_when_no_chunks_found(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline_job, "parse_args", lambda: MagicMock(job_dir=str(tmp_path), model_dir="fake-model", threshold=0.5))

    with pytest.raises(SystemExit):
        run_pipeline_job.main()
