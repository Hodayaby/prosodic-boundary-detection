"""BIU-side entry point for one pipeline job: classify, merge, threshold, QC.

Invoked by the SLURM job pipeline/biu_sync.py submits (its REMOTE_ENTRY_POINT).
Reads the chunks from a job's directory - <chunk_id>.wav plus
<chunk_id>_words.csv per chunk, uploaded by biu_sync.upload_job() - runs the
trained model on each chunk using the same forced-transcript decoding
predict_labels_for_words() already uses in model_development/evaluate_boundary_final2.py,
merges the per-chunk predictions in order (chunk_audio() guarantees no
overlap between chunks, so this is a plain concatenation), applies the
probability threshold, runs a few quality-control sanity checks, and writes
the final table to <job_dir>/result.csv - the file biu_sync.download_result()
expects, using pipeline/schema.py's OUTPUT_COLUMNS.

Run: python run_pipeline_job.py --job-dir pipeline_jobs/<job_id> [--model-dir DIR] [--threshold 0.3]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent / "model_development"))
from evaluate_boundary_final2 import predict_labels_for_words
from pipeline.schema import OUTPUT_COLUMNS, validate_output_schema

# Avoid cuDNN initialization crash on this cluster, as in model_development/evaluate_boundary_final2.py.
torch.backends.cudnn.enabled = False

DEFAULT_MODEL_DIR = "whisper-binary-boundary-model2"
DEFAULT_THRESHOLD = 0.5  # matches the threshold evaluate_test_model2.slurm's latest run used
MAX_REASONABLE_BOUNDARY_RATE = 0.9  # QC sanity ceiling, not a scientific bound


# ============================================================
# Setup
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classify+merge+threshold+QC for one pipeline job.")
    parser.add_argument("--job-dir", required=True, help="Directory with <chunk_id>.wav / <chunk_id>_words.csv, as written by biu_sync.upload_job().")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


# ============================================================
# Classify + merge + threshold + QC
# ============================================================

def _chunk_sort_key(chunk_id: str):
    """Sorts chunk_0, chunk_1, ..., chunk_10 in numeric order, not lexicographic
    (which would put chunk_10 before chunk_2)."""
    match = re.search(r"(\d+)$", chunk_id)
    return int(match.group(1)) if match else chunk_id


def find_chunk_ids(job_dir: Path) -> list:
    """Input: job_dir - directory uploaded by biu_sync.upload_job().
    Output: chunk ids (e.g. "chunk_0") in the order chunk_audio() produced them.
    """
    # globs on *_words.csv specifically, not *.wav - each chunk has one of
    # each, so globbing both would double-count every chunk id
    chunk_ids = [p.name[: -len("_words.csv")] for p in job_dir.glob("*_words.csv")]
    return sorted(chunk_ids, key=_chunk_sort_key)


def classify_chunk(model, processor, device: str, wav_path: Path, words_df: pd.DataFrame, chunk_id: str) -> pd.DataFrame:
    """Run the model on one chunk. Input: loaded model/processor, device, this
    chunk's audio path and word table, and its chunk_id. Output: a DataFrame
    with word/start_s/end_s/boundary_probability/chunk_id (threshold and
    boundary_prediction are added later, in apply_threshold())."""
    words = words_df["word"].astype(str).tolist()
    # threshold=0.5 here is a placeholder - pred_label from this call is unused,
    # only prob_1 (the raw probability) matters; the real threshold is applied later.
    predictions = predict_labels_for_words(model, processor, str(wav_path), words, device, threshold=0.5)

    boundary_probability = [p["prob_1"] for p in predictions]

    result = words_df[["word", "start_s", "end_s"]].reset_index(drop=True).copy()
    result["boundary_probability"] = boundary_probability
    result["chunk_id"] = chunk_id
    return result


def apply_threshold(merged: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Turn boundary_probability into the final binary decision.
    Input: merged - concatenated chunk predictions; threshold - decision cutoff.
    Output: DataFrame with pipeline/schema.py's OUTPUT_COLUMNS, in order."""
    result = merged.copy()
    result["threshold"] = threshold
    result["boundary_prediction"] = (result["boundary_probability"] >= threshold).astype(int)
    return result[list(OUTPUT_COLUMNS)]


def quality_check(result: pd.DataFrame) -> None:
    """Sanity-check the final output before it's written out.
    Input: result - the final, thresholded output table.
    Output: none. Raises ValueError (or SchemaError) on the first check that fails.
    """
    validate_output_schema(result)  # required columns, prob in [0,1], prediction in {0,1}, end_s >= start_s

    if result["word"].isna().any() or (result["word"].astype(str).str.strip() == "").any():
        raise ValueError("QC failed: output has missing or empty words")

    # keyed on (chunk_id, start_s, end_s), not word - two different words can
    # legitimately repeat ("the the"), but the same chunk producing the same
    # timestamp twice means something upstream actually broke
    duplicates = result.duplicated(subset=["chunk_id", "start_s", "end_s"])
    if duplicates.any():
        raise ValueError(f"QC failed: {int(duplicates.sum())} duplicate (chunk_id, start_s, end_s) rows")

    boundary_rate = result["boundary_prediction"].mean() if len(result) else 0.0
    if boundary_rate > MAX_REASONABLE_BOUNDARY_RATE:
        raise ValueError(
            f"QC failed: {boundary_rate:.0%} of words predicted as boundaries - "
            f"suspiciously high, likely points to a broken run rather than real predictions"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir)

    chunk_ids = find_chunk_ids(job_dir)
    if not chunk_ids:
        print(f"No chunks found in {job_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.model_dir}")
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = model.to(device)
    model.eval()

    chunk_tables = []
    for chunk_id in chunk_ids:
        wav_path = job_dir / f"{chunk_id}.wav"
        words_df = pd.read_csv(job_dir / f"{chunk_id}_words.csv")
        print(f"Classifying {chunk_id} ({len(words_df)} words)")
        chunk_tables.append(classify_chunk(model, processor, device, wav_path, words_df, chunk_id))

    # Plain concatenation, not a real merge - chunk_audio() guarantees no overlap between chunks.
    merged = pd.concat(chunk_tables, ignore_index=True)

    result = apply_threshold(merged, args.threshold)
    quality_check(result)

    result_path = job_dir / "result.csv"
    result.to_csv(result_path, index=False)
    print(f"Wrote {len(result)} rows to {result_path}")


if __name__ == "__main__":
    main()
