"""Compare our model's boundary predictions against Oz's LLAMA3 model.

Both models are meant to run on the same MARSEC-derived words (see KAN-66) -
data neither model trained on, unlike the LLMs/sherlock/merlin data where we
each trained on part of what the other uses for evaluation (see KAN-65).

Expected input for --ours and --ground-truth: a CSV with word/start_s/end_s
plus a boundary column (see OURS_BOUNDARY_COL / GROUND_TRUTH_BOUNDARY_COL
below) - the same shape pipeline/schema.py's OUTPUT_COLUMNS already uses, and
what prepare_marsec_data.py in this folder produces.

Expected input for --oz: whatever his LLAMA_training.py's run_model(...,
save_results=True) writes out - confirmed columns are word/label/p_boundary_0/
p_boundary_1 (label there is just an echo of the ground truth we fed in, not
his prediction - see load_oz_predictions()).

All three inputs are matched by row order, not by re-joining on word text -
this only makes sense when all three were produced by running on the exact
same word list in the exact same order. Word text is still cross-checked at
every row as a sanity check, so a silent misalignment fails loudly instead
of quietly comparing the wrong words to each other.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_development"))
from evaluate_boundary_final2 import compute_metrics

OURS_BOUNDARY_COL = "boundary_prediction"
GROUND_TRUTH_BOUNDARY_COL = "boundary"


def load_predictions(csv_path: Path, boundary_col: str) -> pd.DataFrame:
    """Load a word/start_s/end_s/<boundary_col> CSV - our own output shape.

    Input: csv_path - the file to load; boundary_col - which column holds
        the 0/1 boundary label.
    Output: DataFrame with a standardized "word" and "boundary" column.
    """
    df = pd.read_csv(csv_path)
    missing = {"word", boundary_col} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}. Found: {list(df.columns)}")
    return pd.DataFrame({"word": df["word"], "boundary": df[boundary_col].astype(int)})


def load_oz_predictions(path: Path, threshold: float) -> pd.DataFrame:
    """Load Oz's model's predictions and adapt them to our word/boundary shape.

    His LLAMA_training.py (predict_file/run_model) writes exactly these
    columns: index, word, label, p_boundary_0, p_boundary_1 - label is just
    an echo of whatever ground truth was in the input file, not his
    prediction, so the actual 0/1 call has to be made here by thresholding
    p_boundary_1 ourselves (his own script only ever uses a threshold
    internally, for the xlsx highlighting - it never saves a binary
    prediction column).

    Input: path - Oz's prediction file (.xlsx or .csv, from run_model(...,
        save_results=True)); threshold - cutoff applied to p_boundary_1.
    Output: DataFrame with "word" and "boundary" columns.
    """
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)

    missing = {"word", "p_boundary_1"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}. Found: {list(df.columns)}")
    return pd.DataFrame({"word": df["word"], "boundary": (df["p_boundary_1"] >= threshold).astype(int)})


def align_and_check(dfs: dict) -> pd.DataFrame:
    """Join predictions + ground truth by row order, verifying every row's
    word text agrees across all of them first.

    Input: dfs - {name: DataFrame}, each with "word"/"boundary" columns,
        all produced by running on the identical word list in the same order.
    Output: one DataFrame with a "word" column and one "<name>_boundary"
        column per input. Raises ValueError on the first row where two
        sources disagree on which word they're even talking about.
    """
    lengths = {name: len(df) for name, df in dfs.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Row count mismatch - these weren't run on the same word list: {lengths}")

    names = list(dfs.keys())
    reference = dfs[names[0]]["word"].reset_index(drop=True)
    for name in names[1:]:
        other = dfs[name]["word"].reset_index(drop=True)
        mismatches = reference != other
        if mismatches.any():
            i = mismatches.idxmax()
            raise ValueError(
                f"Word mismatch at row {i}: {names[0]}='{reference[i]}' vs {name}='{other[i]}' - "
                "these inputs aren't aligned to the same word list."
            )

    combined = pd.DataFrame({"word": reference})
    for name, df in dfs.items():
        combined[f"{name}_boundary"] = df["boundary"].reset_index(drop=True)
    return combined


def compare(ours_path: Path, oz_path: Path, ground_truth_path: Path, oz_threshold: float) -> None:
    ground_truth = load_predictions(ground_truth_path, GROUND_TRUTH_BOUNDARY_COL)
    ours = load_predictions(ours_path, OURS_BOUNDARY_COL)
    oz = load_oz_predictions(oz_path, oz_threshold)

    combined = align_and_check({"truth": ground_truth, "ours": ours, "oz": oz})

    print("=" * 80)
    print("OUR MODEL vs ground truth")
    print("=" * 80)
    for k, v in compute_metrics(combined["truth_boundary"], combined["ours_boundary"]).items():
        print(f"{k}: {v}")

    print()
    print("=" * 80)
    print("OZ'S MODEL vs ground truth")
    print("=" * 80)
    for k, v in compute_metrics(combined["truth_boundary"], combined["oz_boundary"]).items():
        print(f"{k}: {v}")

    print()
    print("=" * 80)
    print("AGREEMENT BETWEEN THE TWO MODELS (not vs. ground truth)")
    print("=" * 80)
    both_boundary = ((combined["ours_boundary"] == 1) & (combined["oz_boundary"] == 1)).sum()
    only_ours = ((combined["ours_boundary"] == 1) & (combined["oz_boundary"] == 0)).sum()
    only_oz = ((combined["ours_boundary"] == 0) & (combined["oz_boundary"] == 1)).sum()
    neither = ((combined["ours_boundary"] == 0) & (combined["oz_boundary"] == 0)).sum()
    agreement_rate = (combined["ours_boundary"] == combined["oz_boundary"]).mean()
    print(f"both predicted boundary: {both_boundary}")
    print(f"only we predicted boundary: {only_ours}")
    print(f"only Oz predicted boundary: {only_oz}")
    print(f"neither predicted boundary: {neither}")
    print(f"overall agreement rate: {agreement_rate:.4f}")

    disagreements = combined[combined["ours_boundary"] != combined["oz_boundary"]]
    out_path = ground_truth_path.parent / "oz_comparison_disagreements.csv"
    disagreements.to_csv(out_path, index=False)
    print(f"\nSaved {len(disagreements)} disagreement rows to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare our model vs Oz's model on the same word-aligned data.")
    parser.add_argument("--ours", required=True, type=Path, help="Our model's predictions CSV (word/start_s/end_s/boundary_prediction).")
    parser.add_argument("--oz", required=True, type=Path, help="Oz's model's predictions file (.xlsx or .csv).")
    parser.add_argument("--ground-truth", required=True, type=Path, help="Ground-truth boundary labels CSV (word/start_s/end_s/boundary).")
    parser.add_argument(
        "--oz-threshold", type=float, default=0.5,
        help="Cutoff applied to Oz's p_boundary_1 to get a 0/1 prediction (his script picks an 'Optimal Threshold' per run - check hyperparam_results.csv for it if you want to match that instead of 0.5).",
    )
    args = parser.parse_args()
    compare(args.ours, args.oz, args.ground_truth, args.oz_threshold)
