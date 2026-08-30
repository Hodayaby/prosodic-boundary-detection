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
from pathlib import Path
import sys

# Same non-interactive backend as evaluate_boundary_final2.py/analyze_false_positives.py -
# this runs headlessly as often as not (SLURM, or just no need for a popup window).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_development"))
from evaluate_boundary_final2 import compute_metrics

OURS_BOUNDARY_COL = "boundary_prediction"
OURS_PROBABILITY_COL = "boundary_probability"
GROUND_TRUTH_BOUNDARY_COL = "boundary"


def load_predictions(csv_path: Path, boundary_col: str, probability_col: str = None) -> pd.DataFrame:
    """Load a word/start_s/end_s/<boundary_col> CSV - our own output shape.

    Input: csv_path - the file to load; boundary_col - which column holds
        the 0/1 boundary label; probability_col - optional, which column
        holds the predicted probability (ground truth has none).
    Output: DataFrame with standardized "word"/"boundary" columns, plus
        "probability" if probability_col was given.
    """
    df = pd.read_csv(csv_path)
    required = {"word", boundary_col} | ({probability_col} if probability_col else set())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}. Found: {list(df.columns)}")

    result = pd.DataFrame({"word": df["word"], "boundary": df[boundary_col].astype(int)})
    if probability_col:
        result["probability"] = df[probability_col].astype(float)
    return result


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
    Output: DataFrame with "word", "boundary", and "probability" columns.
    """
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)

    missing = {"word", "p_boundary_1"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}. Found: {list(df.columns)}")
    return pd.DataFrame({
        "word": df["word"],
        "boundary": (df["p_boundary_1"] >= threshold).astype(int),
        "probability": df["p_boundary_1"].astype(float),
    })


def align_and_check(dfs: dict) -> pd.DataFrame:
    """Join predictions + ground truth by row order, verifying every row's
    word text agrees across all of them first.

    Input: dfs - {name: DataFrame}, each with a "word"/"boundary" column and
        optionally a "probability" column, all produced by running on the
        identical word list in the same order.
    Output: one DataFrame with a "word" column and one "<name>_boundary"
        column per input (plus "<name>_probability" for inputs that had
        one). Raises ValueError on the first row where two sources disagree
        on which word they're even talking about.
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
        if "probability" in df.columns:
            combined[f"{name}_probability"] = df["probability"].reset_index(drop=True)
    return combined


def make_confusion_matrix_graph(combined: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: TP/FP/FN/TN counts, ours vs Oz, side by side per category.

    Input: combined - align_and_check()'s output; output_dir - where to save the PNG.
    Output: the saved file's path.
    """
    truth = combined["truth_boundary"]
    categories = ["TP", "FP", "FN", "TN"]

    def counts_for(pred_col: str):
        pred = combined[pred_col]
        return [
            int(((truth == 1) & (pred == 1)).sum()),  # TP
            int(((truth == 0) & (pred == 1)).sum()),  # FP
            int(((truth == 1) & (pred == 0)).sum()),  # FN
            int(((truth == 0) & (pred == 0)).sum()),  # TN
        ]

    ours_counts = counts_for("ours_boundary")
    oz_counts = counts_for("oz_boundary")

    x = range(len(categories))
    width = 0.35

    plt.figure(figsize=(8, 6))
    bars_ours = plt.bar([i - width / 2 for i in x], ours_counts, width, label="Ours", color="#4C72B0")
    bars_oz = plt.bar([i + width / 2 for i in x], oz_counts, width, label="Oz", color="#DD8452")
    for bars in (bars_ours, bars_oz):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha="center", va="bottom")

    plt.xticks(list(x), categories)
    plt.ylabel("Count")
    plt.title("Confusion matrix breakdown: ours vs Oz")
    plt.legend()
    plt.tight_layout()

    out_path = output_dir / "confusion_matrix_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def make_probability_distribution_graph(combined: pd.DataFrame, output_dir: Path) -> Path:
    """Predicted probability vs. true label, one panel per model, side by side.

    Same style as analyze_false_positives.py's FP/TP histogram - here split
    by true_label (0/1) instead of error type, to see how well each model's
    probabilities separate the two classes.

    Input: combined - align_and_check()'s output (needs *_probability columns);
        output_dir - where to save the PNG.
    Output: the saved file's path.
    """
    truth = combined["truth_boundary"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    for ax, name in zip(axes, ["ours", "oz"]):
        prob = combined[f"{name}_probability"]
        ax.hist(prob[truth == 0], bins=30, alpha=0.6, label="true boundary=0", color="#4C72B0")
        ax.hist(prob[truth == 1], bins=30, alpha=0.6, label="true boundary=1", color="#DD8452")
        # log scale - the huge majority-class spike near 0 would otherwise
        # flatten every other bar into invisibility on a linear axis
        ax.set_yscale("log")
        ax.set_title("Ours" if name == "ours" else "Oz")
        ax.set_xlabel("predicted P(boundary=1)")
        ax.legend()

    axes[0].set_ylabel("Count (log scale)")
    fig.suptitle("Predicted probability vs. true label")
    fig.tight_layout()

    out_path = output_dir / "probability_vs_truth.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def split_into_recordings(start_s: pd.Series) -> list:
    """Find where one MARSEC recording ends and the next begins in a combined
    table like our_model_marsec_combined.csv - each recording's start_s
    resets to (near) 0, so a recording boundary is wherever start_s drops
    instead of increasing. There's no explicit recording-id column to key
    off instead, since combine_marsec_results.py doesn't keep one.

    Input: start_s - the combined table's start_s column, in original row order.
    Output: list of (start_row, end_row) index pairs, one per recording, covering every row.
    """
    start_s = start_s.reset_index(drop=True)
    reset_points = [0] + list(start_s.index[start_s.diff() < 0]) + [len(start_s)]
    return [(reset_points[i], reset_points[i + 1]) for i in range(len(reset_points) - 1)]


def make_recording_timeline_graph(
    combined: pd.DataFrame, start_s: pd.Series, output_dir: Path, recording_index: int = 0
) -> Path:
    """Line graph of both models' predicted probability over time, for one
    recording - a single continuous stretch of audio is much easier to read
    as a timeline than the full multi-recording dataset would be (8,866
    words across 12 unrelated recordings has no meaningful single timeline).

    Input: combined - align_and_check()'s output; start_s - the matching
        start_s column from the --ours file, used only to find where one
        recording ends and the next begins (see split_into_recordings);
        output_dir - where to save the PNG; recording_index - which
        recording to plot (0 = first).
    Output: the saved file's path. Raises IndexError if recording_index is
        out of range for how many recordings split_into_recordings() found.
    """
    recordings = split_into_recordings(start_s)
    if not (0 <= recording_index < len(recordings)):
        raise IndexError(f"recording_index={recording_index}, but only {len(recordings)} recordings were found")

    start_row, end_row = recordings[recording_index]
    segment = combined.iloc[start_row:end_row]
    t = start_s.iloc[start_row:end_row]

    plt.figure(figsize=(16, 5))
    plt.plot(t, segment["ours_probability"], label="Ours", color="#4C72B0", linewidth=1)
    plt.plot(t, segment["oz_probability"], label="Oz", color="#DD8452", linewidth=1)

    true_boundary_times = t[segment["truth_boundary"] == 1]
    for i, x in enumerate(true_boundary_times):
        plt.axvline(x, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, label="true boundary" if i == 0 else None)

    plt.xlabel("time in recording (s)")
    plt.ylabel("predicted P(boundary=1)")
    plt.title(f"Predicted probability over time - recording #{recording_index} ({len(segment)} words)")
    plt.legend()
    plt.tight_layout()

    out_path = output_dir / f"timeline_recording_{recording_index}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def compare(
    ours_path: Path, oz_path: Path, ground_truth_path: Path, oz_threshold: float,
    output_dir: Path = None, timeline_recording: int = 0,
) -> None:
    ground_truth = load_predictions(ground_truth_path, GROUND_TRUTH_BOUNDARY_COL)
    ours = load_predictions(ours_path, OURS_BOUNDARY_COL, OURS_PROBABILITY_COL)
    oz = load_oz_predictions(oz_path, oz_threshold)
    output_dir = output_dir or ground_truth_path.parent

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
    out_path = output_dir / "oz_comparison_disagreements.csv"
    disagreements.to_csv(out_path, index=False, encoding="utf-8-sig")  # BOM, so Excel renders non-English words (e.g. Hebrew) correctly
    print(f"\nSaved {len(disagreements)} disagreement rows to: {out_path}")

    confusion_path = make_confusion_matrix_graph(combined, output_dir)
    print(f"Saved confusion matrix comparison to: {confusion_path}")

    prob_path = make_probability_distribution_graph(combined, output_dir)
    print(f"Saved probability-vs-truth graph to: {prob_path}")

    start_s = pd.read_csv(ours_path)["start_s"]
    num_recordings = len(split_into_recordings(start_s))
    timeline_path = make_recording_timeline_graph(combined, start_s, output_dir, timeline_recording)
    print(f"Saved timeline graph for recording #{timeline_recording} (of {num_recordings} found) to: {timeline_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare our model vs Oz's model on the same word-aligned data.")
    parser.add_argument("--ours", required=True, type=Path, help="Our model's predictions CSV (word/start_s/end_s/boundary_prediction).")
    parser.add_argument("--oz", required=True, type=Path, help="Oz's model's predictions file (.xlsx or .csv).")
    parser.add_argument("--ground-truth", required=True, type=Path, help="Ground-truth boundary labels CSV (word/start_s/end_s/boundary).")
    parser.add_argument(
        "--oz-threshold", type=float, default=0.5,
        help="Cutoff applied to Oz's p_boundary_1 to get a 0/1 prediction (his script picks an 'Optimal Threshold' per run - check hyperparam_results.csv for it if you want to match that instead of 0.5).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save the disagreements CSV and graphs (default: next to --ground-truth).",
    )
    parser.add_argument(
        "--timeline-recording", type=int, default=0,
        help="Which recording (0-indexed, in file order) to plot the probability-over-time timeline for.",
    )
    args = parser.parse_args()
    compare(args.ours, args.oz, args.ground_truth, args.oz_threshold, args.output_dir, args.timeline_recording)
