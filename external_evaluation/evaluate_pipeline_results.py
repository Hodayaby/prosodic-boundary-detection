"""Score the real pipeline's Excel output against ground truth - no model
inference here, just evaluation. For corpora that already have hand-labeled
boundaries (MARSEC, the Hebrew transcripts from Galit's lab), where the
question is "how well does the trained model do on this data", not "what
does the model predict for this new data".

Reuses every metric, report, and graph from evaluate_boundary_final2.py's
--skip-inference path - this script's only job is turning one corpus's
Excel file (the real pipeline's own output columns: word, boundary_probability,
boundary_prediction, ... - see pipeline/schema.py - plus a ground-truth
column this pipeline never produces itself) into the predictions table that
path already expects, so the exact same evaluation code runs either way.

Expects one Excel file per corpus (MARSEC.xlsx, hebrew_galit.xlsx, ...), not
per recording - if a corpus is split across many files, concatenate them
into one before running this.

Typical run (from anywhere - see the sys.path line below):
    python evaluate_pipeline_results.py MARSEC_results.xlsx --output-dir eval_marsec
    python evaluate_pipeline_results.py hebrew_galit_results.xlsx --output-dir eval_hebrew --corpus-name "Hebrew (Galit's lab)"
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_development"))
from evaluate_boundary_final2 import (
    DEFAULT_THRESHOLDS,
    add_error_type,
    compute_metrics,
    create_graphs,
    make_manual_review_examples,
    make_threshold_summary,
    write_evaluation_report,
    write_metrics_summary,
)

# Each canonical field -> the column names it might actually show up as.
# Tries an exact match first (case-insensitive), same approach as
# pipeline/input_validation.py's transcript column resolution - only
# guesses from a short, deliberate candidate list, never silently picks
# an unrelated column.
WORD_COLUMN_CANDIDATES = ["word", "token"]
TRUE_LABEL_COLUMN_CANDIDATES = ["true_label", "original_label", "boundary", "label", "gt_label", "ground_truth"]
PROBABILITY_COLUMN_CANDIDATES = ["boundary_probability", "prob_1", "probability"]
PREDICTION_COLUMN_CANDIDATES = ["boundary_prediction", "pred_label", "prediction"]

# Same binarization used throughout this project (data_cleaning.py,
# fix_llms_csv.py): 1 (clause boundary) and 4 (transient boundary) are real
# boundaries; 2 (restart) and 3 (back to previous clause) are disfluencies
# mid-utterance, not the speaker actually finishing a clause.
ORIGINAL_LABEL_TO_BINARY = {0: 0, 1: 1, 2: 0, 3: 0, 4: 1}


def _find_column(df: pd.DataFrame, candidates: List[str], description: str) -> str:
    """Input: df - the loaded Excel sheet; candidates - column names to look
    for, in priority order; description - used only in the error message.
    Output: the first matching column name. Raises ValueError if none match.
    """
    lower_to_actual = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    raise ValueError(
        f"Could not find a {description} column. Looked for: {candidates}. "
        f"Found columns: {list(df.columns)}"
    )


def load_pipeline_results(excel_path: Path, corpus_name: str, fallback_threshold: float) -> pd.DataFrame:
    """Load one corpus's Excel file and reshape it into the columns
    evaluate_boundary_final2.py's --skip-inference path expects:
    audio_file, word_index, word, true_label, pred_label, prob_1.

    word_index is just this file's row order - there's no chunk structure
    to preserve here, only one corpus per file. true_label is binarized if
    it still uses the original 0-4 scheme; already-binary 0/1 values pass
    through unchanged. pred_label is taken from the pipeline's own
    boundary_prediction column if present, otherwise derived from the
    probability at fallback_threshold - lets a different threshold be
    tried against the same probabilities without rerunning the pipeline.

    Input: excel_path - the corpus's Excel file; corpus_name - label to
        tag every row with; fallback_threshold - used only if no prediction
        column is found.
    Output: DataFrame with exactly the columns the eval functions need.
    """
    df = pd.read_excel(excel_path)

    word_col = _find_column(df, WORD_COLUMN_CANDIDATES, "word")
    true_col = _find_column(df, TRUE_LABEL_COLUMN_CANDIDATES, "ground-truth label")
    prob_col = _find_column(df, PROBABILITY_COLUMN_CANDIDATES, "predicted probability")

    true_label_raw = df[true_col].fillna(0).astype(int)
    true_label = true_label_raw.map(ORIGINAL_LABEL_TO_BINARY)
    if true_label.isna().any():
        bad_values = sorted(true_label_raw[true_label.isna()].unique().tolist())
        raise ValueError(
            f"Ground-truth column '{true_col}' has values outside the expected 0-4 scheme: {bad_values}"
        )

    try:
        pred_col = _find_column(df, PREDICTION_COLUMN_CANDIDATES, "predicted label")
        pred_label = df[pred_col].astype(int)
    except ValueError:
        pred_label = (df[prob_col].astype(float) >= fallback_threshold).astype(int)

    return pd.DataFrame({
        "audio_file": corpus_name,
        "audio_path": str(excel_path),  # only used for display in manual_review_examples.csv - not a real audio file here
        "word_index": range(len(df)),
        "word": df[word_col],
        "true_label": true_label.astype(int),
        "pred_label": pred_label,
        "prob_1": df[prob_col].astype(float),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the real pipeline's Excel output against ground truth (no model inference)."
    )
    parser.add_argument("excel_path", help="Excel file with the pipeline's predictions plus a ground-truth label column")
    parser.add_argument("--corpus-name", default=None, help="Label for this corpus in the report (default: the file's name)")
    parser.add_argument("--output-dir", default="evaluation_outputs_external")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Only used if the Excel file has no prediction column of its own - "
             "otherwise predictions are taken exactly as the pipeline produced them.",
    )
    parser.add_argument("--max-review-per-type", type=int, default=25)
    parser.add_argument("--context-window", type=int, default=6)
    parser.add_argument("--high-conf-fp-threshold", type=float, default=0.985)
    parser.add_argument("--no-graphs", action="store_true", help="Skip PNG graphs and evaluation_report.md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_name = args.corpus_name or Path(args.excel_path).stem

    df = load_pipeline_results(Path(args.excel_path), corpus_name, args.threshold)
    df = add_error_type(df)

    predictions_path = output_dir / "predictions_with_errors.csv"
    # utf-8-sig, not plain utf-8: Excel doesn't auto-detect BOM-less UTF-8 and
    # renders non-ASCII text (Hebrew here) as mojibake without the BOM
    df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    metrics = compute_metrics(df["true_label"], df["pred_label"], df["prob_1"])
    error_counts = df["error_type"].value_counts()

    threshold_df = make_threshold_summary(df, DEFAULT_THRESHOLDS)
    threshold_path = output_dir / "threshold_summary.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    # A minimal stand-in for the real evaluate_boundary_final2.py argparse
    # Namespace - only the fields the report-writing functions actually
    # read (model_dir/eval_csv/random_seed are cosmetic report metadata
    # here, since no model ran).
    report_args = argparse.Namespace(
        model_dir="(none - scored an existing pipeline output, no model run here)",
        eval_csv=str(args.excel_path),
        random_seed="n/a",
        threshold=args.threshold,
        max_review_per_type=args.max_review_per_type,
        context_window=args.context_window,
        high_conf_fp_threshold=args.high_conf_fp_threshold,
    )

    manual_review = make_manual_review_examples(df, report_args)
    manual_review_path = output_dir / "manual_review_examples.csv"
    manual_review.to_csv(manual_review_path, index=False, encoding="utf-8-sig")

    saved_files = [predictions_path, threshold_path, manual_review_path]

    print(f"Corpus: {corpus_name}")
    print(f"Total words: {metrics['total_evaluated_words']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    write_metrics_summary(
        output_dir / "metrics_summary.txt",
        report_args,
        metrics,
        error_counts,
        threshold_df,
        fp_deep_needed=False,
        high_conf_fp_count=0,
        merge_notes=[],
        saved_files=saved_files,
    )
    saved_files.append(output_dir / "metrics_summary.txt")

    if not args.no_graphs:
        graph_paths = create_graphs(output_dir, df, metrics, threshold_df)
        saved_files += graph_paths
        write_evaluation_report(
            output_dir / "evaluation_report.md", report_args, metrics, threshold_df,
            fp_deep_needed=False, saved_files=saved_files, graph_paths=graph_paths,
        )

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
