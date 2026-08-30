"""Combine our pipeline's per-recording MARSEC results with the ground
truth from prepare_marsec_data.py into one table, for both downstream
evaluations (see KAN-65/KAN-66):

- external_evaluation/evaluate_pipeline_results.py needs an .xlsx with
  word/boundary(ground truth)/boundary_probability/boundary_prediction.
- oz_comparison/compare_with_oz.py's --ours and --ground-truth need CSVs.

Expects a directory with one <recording>.csv (ground truth, from
prepare_marsec_data.py) and one <recording>Result.csv (our pipeline's own
output, e.g. downloaded from the app after running it on <recording>.wav)
per recording. Recordings are combined in sorted order (A01, A02, ..., A12),
matching the order prepare_marsec_data.py's --oz-input uses - keeping this
order is what lets the combined output line up with Oz's predictions later.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def _sort_key(recording: str):
    """Sorts A2 before A10 - same numeric-aware ordering chunker/run_pipeline_job.py use."""
    match = re.search(r"(\d+)$", recording)
    return (recording[: match.start()] if match else recording, int(match.group(1)) if match else 0)


def find_recording_pairs(input_dir: Path) -> list:
    """Input: input_dir - folder with <recording>.csv + <recording>Result.csv pairs.
    Output: list of (recording, ground_truth_path, result_path), sorted A01..A12-style.
    Raises FileNotFoundError if a Result.csv has no matching ground-truth file.
    """
    pairs = []
    for result_path in input_dir.glob("*Result.csv"):
        recording = result_path.name[: -len("Result.csv")]
        truth_path = input_dir / f"{recording}.csv"
        if not truth_path.exists():
            raise FileNotFoundError(f"No ground-truth file {truth_path} for {result_path}")
        pairs.append((recording, truth_path, result_path))

    return sorted(pairs, key=lambda p: _sort_key(p[0]))


def combine_one_recording(recording: str, truth_path: Path, result_path: Path) -> pd.DataFrame:
    """Merge one recording's ground truth and predictions by row order,
    checking word text agrees at every row first.

    Input: recording - its name, only used in error messages; truth_path,
        result_path - the two files to merge.
    Output: DataFrame with word/start_s/end_s/boundary/boundary_probability/boundary_prediction.
    """
    truth = pd.read_csv(truth_path)
    result = pd.read_csv(result_path)

    if len(truth) != len(result):
        raise ValueError(f"{recording}: {len(truth)} ground-truth words vs {len(result)} predicted words")

    mismatches = truth["word"].reset_index(drop=True) != result["word"].reset_index(drop=True)
    if mismatches.any():
        i = mismatches.idxmax()
        raise ValueError(f"{recording}: word mismatch at row {i}: '{truth['word'][i]}' vs '{result['word'][i]}'")

    return pd.DataFrame({
        "word": truth["word"],
        "start_s": truth["start_s"],
        "end_s": truth["end_s"],
        "boundary": truth["boundary"],
        "boundary_probability": result["boundary_probability"].reset_index(drop=True),
        "boundary_prediction": result["boundary_prediction"].reset_index(drop=True),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine per-recording MARSEC results + ground truth into one table.")
    parser.add_argument("input_dir", type=Path, help="Folder with <recording>.csv + <recording>Result.csv pairs.")
    parser.add_argument("--output-stem", type=Path, default=Path("our_model_marsec_combined"), help="Output path without extension - writes <stem>.csv and <stem>.xlsx.")
    args = parser.parse_args()

    pairs = find_recording_pairs(args.input_dir)
    if not pairs:
        raise SystemExit(f"No <recording>Result.csv files found in {args.input_dir}")

    combined = pd.concat(
        [combine_one_recording(rec, truth, result) for rec, truth, result in pairs],
        ignore_index=True,
    )

    csv_path = args.output_stem.with_suffix(".csv")
    xlsx_path = args.output_stem.with_suffix(".xlsx")
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")  # BOM, so Excel renders non-English words correctly
    combined.to_excel(xlsx_path, index=False)

    print(f"Combined {len(pairs)} recordings ({[rec for rec, _, _ in pairs]}), {len(combined)} words total.")
    print(f"-> {csv_path}")
    print(f"-> {xlsx_path}")
