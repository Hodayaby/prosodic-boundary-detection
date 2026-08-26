"""
evaluate_boundary_improved.py
=============================
Improved evaluation pipeline for Whisper binary prosodic-boundary detection.

What this script does:
1. Runs forced-transcript evaluation: audio + known transcript words -> predicts only 0/1 labels.
2. Saves one detailed prediction CSV with error_type per word.
3. Saves one readable metrics_summary.txt with the important numbers.
4. Saves one compact threshold_summary.csv for choosing a better decision threshold.
5. Saves one manual_review_examples.csv with a reasonable number of examples to inspect manually.
6. Runs deeper FP analysis only when it is useful, based on the number/rate of high-confidence FP.
7. Optionally merges original labels from alignment files, but only if the files exist and FP analysis is needed.

Default output files in evaluation_outputs/:
- predictions_with_errors.csv
- metrics_summary.txt
- threshold_summary.csv
- manual_review_examples.csv
- high_conf_false_positive_context.csv          only if FP analysis is triggered
- high_conf_fp_original_label_summary.csv       only if FP analysis is triggered and alignment files exist
- predictions_with_original_labels_clean.csv    only if FP analysis is triggered and alignment files exist

Typical run:
    python evaluate_boundary_improved.py

Useful variants:
    python evaluate_boundary_improved.py --threshold 0.5 --output-dir eval_run_01
    python evaluate_boundary_improved.py --threshold 0.4 --max-review-per-type 20
    python evaluate_boundary_improved.py --skip-inference --predictions-csv evaluation_outputs/predictions_with_errors.csv
"""

import argparse
import os

# Use a non-interactive backend so graph creation works on SLURM / headless servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Avoid cuDNN initialization crash during evaluation, as in the earlier scripts.
torch.backends.cudnn.enabled = False

DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.985, 0.99, 0.995]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improved forced-transcript boundary evaluation with compact outputs."
    )

    parser.add_argument("--model-dir", default="whisper-binary-boundary-model2")
    parser.add_argument("--eval-csv", default="data/splits/valid_chunks.csv")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="evaluation_outputs2")
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Do not create PNG graphs or evaluation_report.md.",
    )

    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Use an existing predictions CSV instead of running the model again.",
    )
    parser.add_argument(
        "--predictions-csv",
        default=None,
        help="Existing predictions CSV to analyze when --skip-inference is used.",
    )

    parser.add_argument(
        "--max-review-per-type",
        type=int,
        default=25,
        help="Max examples per error/review type in manual_review_examples.csv.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=6,
        help="Number of words before/after target word in manual context strings.",
    )

    parser.add_argument(
        "--high-conf-fp-threshold",
        type=float,
        default=0.985,
        help="FP with prob_1 >= this value is considered high-confidence FP.",
    )
    parser.add_argument(
        "--min-high-conf-fp-for-deep-analysis",
        type=int,
        default=10,
        help="Run deep FP analysis if at least this many high-confidence FP exist.",
    )
    parser.add_argument(
        "--fp-rate-trigger",
        type=float,
        default=0.05,
        help="Run deep FP analysis if FP / predicted-boundaries is at least this value.",
    )

    parser.add_argument(
        "--merlin-align-csv",
        default="data/data_with_numbers2341/merlin_audio_word_alignment.csv",
    )
    parser.add_argument(
        "--sherlock-align-csv",
        default="data/data_with_numbers2341/sherlock_audio_word_alignment.csv",
    )

    return parser.parse_args()


def parse_labeled_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Converts:
        '1 okay 0 so 0 they 1 stopped'
    Into:
        labels = ['1', '0', '0', '1']
        words  = ['okay', 'so', 'they', 'stopped']
    """
    tokens = str(text).strip().split()
    labels: List[str] = []
    words: List[str] = []

    i = 0
    while i < len(tokens) - 1:
        label = tokens[i]
        word = tokens[i + 1]

        if label not in {"0", "1"}:
            i += 1
            continue

        labels.append(label)
        words.append(word)
        i += 2

    return labels, words


def encode_prefix(tokenizer, prefix_text: str) -> List[int]:
    """
    Tokenizes the current decoder prefix and removes EOS because generation is not done yet.
    """
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
    if prefix_ids and prefix_ids[-1] == tokenizer.eos_token_id:
        prefix_ids = prefix_ids[:-1]
    return prefix_ids


def get_label_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    # Whisper's BPE tokenizer encodes a digit differently as the very first
    # token of a sequence vs. after a space later on, so the label ids for
    # "the first word" and "every word after that" have to be looked up
    # separately - reusing one set for both would read the wrong logits.
    first_label_token_ids = [
        tokenizer.encode("0", add_special_tokens=False)[0],
        tokenizer.encode("1", add_special_tokens=False)[0],
    ]
    next_label_token_ids = [
        tokenizer.encode(" 0", add_special_tokens=False)[0],
        tokenizer.encode(" 1", add_special_tokens=False)[0],
    ]
    return first_label_token_ids, next_label_token_ids


def predict_labels_for_words(
    model,
    processor,
    audio_path: str,
    words: List[str],
    device: str,
    threshold: float,
) -> List[Dict]:
    """Force-decode one boundary label per known word, instead of letting
    Whisper freely generate the whole "0 word 0 word 1 word..." sequence.

    Free generation was the baseline approach (prepare_data.py) and it can
    drift off the real transcript - skip, repeat, or invent words - which
    breaks any word-by-word comparison against ground truth. Here the actual
    words are fed into the decoder one at a time as a fixed prefix, and at
    each step only the "0"/"1" label-token logits are read off - the model
    never gets a chance to predict the word itself, only the boundary
    before it.
    """
    tokenizer = processor.tokenizer
    first_label_token_ids, next_label_token_ids = get_label_token_ids(tokenizer)

    speech_array, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor.feature_extractor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    predictions: List[Dict] = []
    prefix_parts: List[str] = []

    with torch.no_grad():
        encoded_features = model.get_encoder()(input_features).last_hidden_state

        for word_index, word in enumerate(words):
            prefix_text = " ".join(prefix_parts)
            prefix_ids = encode_prefix(tokenizer, prefix_text)
            decoder_input_ids = torch.tensor([prefix_ids], device=device)

            decoder_output = model.get_decoder()(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoded_features,
            )
            logits = model.proj_out(decoder_output.last_hidden_state)

            candidate_token_ids = first_label_token_ids if word_index == 0 else next_label_token_ids
            label_logits = logits[0, -1, candidate_token_ids]
            label_probs = torch.softmax(label_logits, dim=0)

            prob_0 = float(label_probs[0].item())
            prob_1 = float(label_probs[1].item())
            pred_label = "1" if prob_1 >= threshold else "0"

            predictions.append(
                {
                    "word_index": word_index,
                    "word": word,
                    "pred_label": pred_label,
                    "prob_0": prob_0,
                    "prob_1": prob_1,
                }
            )

            # WPLP-style forced transcript process:
            # append predicted label + known transcript word before predicting the next label.
            prefix_parts.append(pred_label)
            prefix_parts.append(word)

    return predictions


def load_evaluation_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load the fixed split CSV for evaluation.

    Default:
        data/splits/valid_chunks.csv

    Expected columns:
        audio_path = path to wav file
        text       = labeled transcript: 0 word 0 word 1 word ...
        audio_file = optional, used for readable output
    """
    eval_df = pd.read_csv(args.eval_csv)

    required = {"audio_path", "text"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in eval CSV: {sorted(missing)}")

    # Docker/Linux path fix:
    # data\LLMs\file.wav -> data/LLMs/file.wav
    eval_df["audio"] = eval_df["audio_path"].astype(str).str.replace("\\", "/", regex=False)
    eval_df["sentence"] = eval_df["text"]

    # Keep audio_file if it exists. If not, derive a readable name from audio path.
    if "audio_file" not in eval_df.columns:
        eval_df["audio_file"] = eval_df["audio"].apply(lambda p: os.path.basename(str(p)))

    missing_audio = [p for p in eval_df["audio"] if not os.path.exists(p)]
    if missing_audio:
        preview = "\n".join(missing_audio[:20])
        raise FileNotFoundError(
            f"{len(missing_audio)} evaluation audio files are missing. First missing files:\n{preview}"
        )

    print(f"Evaluation CSV: {args.eval_csv}")
    print(f"Evaluation samples: {len(eval_df)}")
    print("Example audio path:", eval_df["audio"].iloc[0])

    return eval_df.reset_index(drop=True)

def run_inference(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    valid_df = load_evaluation_dataframe(args)
    print(f"Evaluation samples: {len(valid_df)}")
    print(f"Threshold: {args.threshold}")

    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model = model.to(device)
    model.eval()

    tokenizer = processor.tokenizer
    print("Tokenization sanity check:")
    print("'0'  ->", tokenizer.encode("0", add_special_tokens=False))
    print("'1'  ->", tokenizer.encode("1", add_special_tokens=False))
    print("' 0' ->", tokenizer.encode(" 0", add_special_tokens=False))
    print("' 1' ->", tokenizer.encode(" 1", add_special_tokens=False))

    all_rows: List[Dict] = []

    for sample_index, row in valid_df.iterrows():
        audio_path = row["audio"]
        audio_file = row["audio_file"]
        true_text = row["sentence"]

        true_labels, words = parse_labeled_text(true_text)
        predictions = predict_labels_for_words(
            model=model,
            processor=processor,
            audio_path=audio_path,
            words=words,
            device=device,
            threshold=args.threshold,
        )

        n = min(len(true_labels), len(predictions))
        truncated = len(true_labels) != len(predictions)

        for i in range(n):
            pred = predictions[i]
            all_rows.append(
                {
                    "sample_index": sample_index,
                    "audio_file": audio_file,
                    "audio_path": audio_path,
                    "word_index": int(pred["word_index"]),
                    "word": pred["word"],
                    "true_label": int(true_labels[i]),
                    "pred_label": int(pred["pred_label"]),
                    "prob_0": float(pred["prob_0"]),
                    "prob_1": float(pred["prob_1"]),
                    "num_true_words_in_sample": len(true_labels),
                    "num_pred_words_in_sample": len(predictions),
                    "sample_was_truncated_for_alignment": truncated,
                }
            )

        if (sample_index + 1) % 10 == 0:
            print(f"Processed {sample_index + 1}/{len(valid_df)} samples")

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No evaluation rows were produced. Check input data format.")

    return df


def add_error_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["true_label"] = df["true_label"].astype(int)
    df["pred_label"] = df["pred_label"].astype(int)
    df["prob_1"] = df["prob_1"].astype(float)

    df["error_type"] = "TN"
    df.loc[(df["true_label"] == 0) & (df["pred_label"] == 1), "error_type"] = "FP"
    df.loc[(df["true_label"] == 1) & (df["pred_label"] == 0), "error_type"] = "FN"
    df.loc[(df["true_label"] == 1) & (df["pred_label"] == 1), "error_type"] = "TP"
    df["margin_from_threshold"] = (df["prob_1"] - 0.5).abs()
    return df


def compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    predicted_boundary_count = tp + fp
    true_boundary_count = tp + fn
    fp_among_predicted_boundaries = fp / predicted_boundary_count if predicted_boundary_count else 0.0

    result = {
        "total_evaluated_words": len(y_true),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "true_boundary_count": int(true_boundary_count),
        "predicted_boundary_count": int(predicted_boundary_count),
        "fp_among_predicted_boundaries": float(fp_among_predicted_boundaries),
    }

    # Threshold-independent metrics.
    # These require probability scores, not binary predictions.
    if y_score is not None and len(set(y_true)) == 2:
        y_score = list(map(float, y_score))
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        result["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None

    return result


def make_threshold_summary(df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    rows: List[Dict] = []
    y_true = df["true_label"].astype(int)
    prob = df["prob_1"].astype(float)

    for threshold in thresholds:
        y_pred = (prob >= threshold).astype(int)
        m = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "threshold": threshold,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "accuracy": round(m["accuracy"], 4),
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
                "tn": m["tn"],
                "predicted_boundary_count": m["predicted_boundary_count"],
                "fp_among_predicted_boundaries": round(m["fp_among_predicted_boundaries"], 4),
            }
        )

    return pd.DataFrame(rows)


def context_for_row(df: pd.DataFrame, row: pd.Series, context_window: int) -> str:
    same_audio = df[df["audio_file"].astype(str) == str(row["audio_file"])].copy()
    word_index = int(row["word_index"])
    context = same_audio[
        (same_audio["word_index"] >= word_index - context_window)
        & (same_audio["word_index"] <= word_index + context_window)
    ].sort_values("word_index")

    pieces: List[str] = []
    for r in context.itertuples():
        token = str(r.word)
        if int(r.word_index) == word_index:
            token = f"[{token}]"
        pieces.append(token)
    return " ".join(pieces)


def with_context(df: pd.DataFrame, rows: pd.DataFrame, context_window: int, review_type: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    rows = rows.copy()
    rows["review_type"] = review_type
    rows["context"] = [context_for_row(df, row, context_window) for _, row in rows.iterrows()]
    columns = [
        "review_type",
        "error_type",
        "audio_file",
        "audio_path",
        "word_index",
        "word",
        "true_label",
        "pred_label",
        "prob_1",
        "context",
    ]
    extra_cols = [c for c in ["original_label", "binary_label"] if c in rows.columns]
    return rows[columns + extra_cols]


def make_manual_review_examples(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    n = args.max_review_per_type
    chunks: List[pd.DataFrame] = []

    fp = df[df["error_type"] == "FP"].sort_values("prob_1", ascending=False).head(n)
    fn = df[df["error_type"] == "FN"].sort_values("prob_1", ascending=True).head(n)
    tp = df[df["error_type"] == "TP"].sort_values("prob_1", ascending=False).head(n)

    # Borderline examples are useful because they show cases where the model is uncertain.
    borderline = df.assign(distance=(df["prob_1"] - args.threshold).abs()).sort_values("distance").head(n)

    chunks.append(with_context(df, fp, args.context_window, "top_false_positives"))
    chunks.append(with_context(df, fn, args.context_window, "top_false_negatives"))
    chunks.append(with_context(df, tp, args.context_window, "confident_true_positives"))
    chunks.append(with_context(df, borderline, args.context_window, "borderline_near_threshold"))

    chunks = [c for c in chunks if not c.empty]
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_and_merge_original_labels(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[Optional[pd.DataFrame], List[str]]:
    notes: List[str] = []
    merlin_path = Path(args.merlin_align_csv)
    sherlock_path = Path(args.sherlock_align_csv)

    if not merlin_path.exists() or not sherlock_path.exists():
        notes.append("Original-label merge skipped: one or both alignment files do not exist.")
        notes.append(f"Missing merlin alignment: {not merlin_path.exists()} ({merlin_path})")
        notes.append(f"Missing sherlock alignment: {not sherlock_path.exists()} ({sherlock_path})")
        return None, notes

    merlin = pd.read_csv(merlin_path)
    sherlock = pd.read_csv(sherlock_path)

    for name, align_df in [("merlin", merlin), ("sherlock", sherlock)]:
        required = {"audio_file", "word_index", "word", "original_label"}
        missing = required - set(align_df.columns)
        if missing:
            notes.append(f"Original-label merge skipped: {name} alignment is missing columns {sorted(missing)}.")
            return None, notes

    merlin["source"] = "merlin"
    sherlock["source"] = "sherlock"
    align = pd.concat([merlin, sherlock], ignore_index=True)

    eval_df = df.copy()
    eval_df["audio_file"] = eval_df["audio_file"].astype(str)
    eval_df["word_index"] = eval_df["word_index"].astype(int)
    align["audio_file"] = align["audio_file"].astype(str)
    align["word_index"] = align["word_index"].astype(int)

    merged = eval_df.merge(
        align,
        on=["audio_file", "word_index"],
        how="left",
        suffixes=("_eval", "_original"),
    )

    word_eval_col = "word_eval" if "word_eval" in merged.columns else "word"
    word_original_col = "word_original" if "word_original" in merged.columns else "word"

    merged["word_match"] = (
        merged[word_eval_col].astype(str).str.lower().str.strip()
        == merged[word_original_col].astype(str).str.lower().str.strip()
    )

    missing_original = int(merged["original_label"].isna().sum())
    mismatches = int((merged["word_match"] == False).sum())
    notes.append(f"Original-label merge rows: {len(merged)}")
    notes.append(f"Rows without original_label: {missing_original}")
    notes.append(f"Rows with word mismatch: {mismatches}")

    clean = merged[(merged["word_match"] == True) & (merged["original_label"].notna())].copy()

    # Normalize main word column for review files.
    if "word_eval" in clean.columns:
        clean["word"] = clean["word_eval"]

    return clean, notes


def make_high_conf_fp_original_label_summary(merged_clean: pd.DataFrame, high_conf_threshold: float) -> pd.DataFrame:
    high_conf_fp = merged_clean[
        (merged_clean["true_label"].astype(int) == 0)
        & (merged_clean["pred_label"].astype(int) == 1)
        & (merged_clean["prob_1"].astype(float) >= high_conf_threshold)
    ].copy()

    counts = high_conf_fp["original_label"].astype(int).value_counts().reindex([0, 1, 2, 3, 4], fill_value=0).sort_index()
    total = int(counts.sum())
    percents = (counts / total * 100).round(2) if total else counts.astype(float)

    summary = pd.DataFrame(
        {
            "original_label": counts.index,
            "count": counts.values,
            "percent": percents.values,
        }
    )
    summary.loc[len(summary)] = {
        "original_label": "2+3 together",
        "count": int(counts.loc[2] + counts.loc[3]),
        "percent": round(((counts.loc[2] + counts.loc[3]) / total * 100), 2) if total else 0.0,
    }
    return summary



def save_threshold_metrics_graph(threshold_df: pd.DataFrame, path: Path) -> None:
    """Save Precision/Recall/F1/Accuracy as a function of the decision threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    ax.plot(threshold_df["threshold"], threshold_df["accuracy"], marker="o", label="Accuracy")
    ax.set_xlabel("Decision threshold for prob_1")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation metrics by threshold")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_graph(metrics: Dict[str, float], path: Path) -> None:
    """Save a simple confusion matrix figure for labels [0, 1]."""
    cm = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")

    max_value = max(max(row) for row in cm) if cm else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_prob_distribution_graph(df: pd.DataFrame, path: Path) -> None:
    """Save distribution of prob_1 grouped by error type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [label for label in ["TN", "FP", "FN", "TP"] if label in set(df["error_type"])]
    data = [df.loc[df["error_type"] == label, "prob_1"].astype(float).values for label in labels]

    if data:
        ax.boxplot(data, labels=[f"{label}\n(n={len(vals)})" for label, vals in zip(labels, data)], showfliers=False)
    ax.axhline(0.5, linestyle="--", linewidth=1, label="0.5 threshold")
    ax.set_xlabel("Error type")
    ax.set_ylabel("prob_1")
    ax.set_title("Boundary probability by error type")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def save_roc_curve_graph(df: pd.DataFrame, path: Path) -> Optional[float]:
    """Save ROC curve and return ROC-AUC."""
    y_true = df["true_label"].astype(int)
    y_score = df["prob_1"].astype(float)

    if y_true.nunique() < 2:
        print("Skipping ROC curve: only one class exists in true_label.")
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / Recall")
    ax.set_title("ROC curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return float(roc_auc)


def save_precision_recall_curve_graph(df: pd.DataFrame, path: Path) -> Optional[float]:
    """Save Precision-Recall curve and return Average Precision."""
    y_true = df["true_label"].astype(int)
    y_score = df["prob_1"].astype(float)

    if y_true.nunique() < 2:
        print("Skipping Precision-Recall curve: only one class exists in true_label.")
        return None

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    positive_rate = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"Average precision = {average_precision:.4f}")
    ax.axhline(positive_rate, linestyle="--", linewidth=1, label=f"Positive-rate baseline = {positive_rate:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return float(average_precision)

def save_high_conf_fp_original_labels_graph(summary_df: pd.DataFrame, path: Path) -> None:
    """Save distribution of original labels among high-confidence FP examples."""
    plot_df = summary_df[summary_df["original_label"].astype(str) != "2+3 together"].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(plot_df["original_label"].astype(str), plot_df["count"].astype(int))
    ax.set_xlabel("Original label")
    ax.set_ylabel("High-confidence FP count")
    ax.set_title("Original labels among high-confidence false positives")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def create_graphs(
    output_dir: Path,
    df: pd.DataFrame,
    metrics: Dict[str, float],
    threshold_df: pd.DataFrame,
    original_label_summary: Optional[pd.DataFrame] = None,
) -> List[Path]:
    """Create all standard evaluation graphs and return paths that were saved."""
    graph_paths: List[Path] = []

    threshold_graph = output_dir / "graph_threshold_metrics.png"
    save_threshold_metrics_graph(threshold_df, threshold_graph)
    graph_paths.append(threshold_graph)

    cm_graph = output_dir / "graph_confusion_matrix.png"
    save_confusion_matrix_graph(metrics, cm_graph)
    graph_paths.append(cm_graph)

    prob_graph = output_dir / "graph_prob1_by_error_type.png"
    save_prob_distribution_graph(df, prob_graph)
    graph_paths.append(prob_graph)

    roc_graph = output_dir / "graph_roc_curve.png"
    roc_auc = save_roc_curve_graph(df, roc_graph)
    if roc_auc is not None and roc_graph.exists():
        graph_paths.append(roc_graph)

    pr_graph = output_dir / "graph_precision_recall_curve.png"
    average_precision = save_precision_recall_curve_graph(df, pr_graph)
    if average_precision is not None and pr_graph.exists():
        graph_paths.append(pr_graph)

    if original_label_summary is not None and not original_label_summary.empty:
        fp_labels_graph = output_dir / "graph_high_conf_fp_original_labels.png"
        save_high_conf_fp_original_labels_graph(original_label_summary, fp_labels_graph)
        if fp_labels_graph.exists():
            graph_paths.append(fp_labels_graph)

    return graph_paths


def write_evaluation_report(
    path: Path,
    args: argparse.Namespace,
    metrics: Dict[str, float],
    threshold_df: pd.DataFrame,
    fp_deep_needed: bool,
    saved_files: List[Path],
    graph_paths: List[Path],
) -> None:
    """Write a Markdown report that links the summary files and embeds PNG graphs."""
    best_f1 = threshold_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]

    lines: List[str] = []
    lines.append("# Boundary evaluation report")
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Model dir: `{args.model_dir}`")
    lines.append(f"- Threshold used: `{args.threshold}`")
    lines.append(f"- Eval CSV: `{args.eval_csv}`")
    lines.append(f"- Random seed: `{args.random_seed}`")
    lines.append("")

    lines.append("## Main metrics")
    lines.append("")
    lines.append(f"- Accuracy: **{metrics['accuracy']:.4f}**")
    lines.append(f"- Precision: **{metrics['precision']:.4f}**")
    lines.append(f"- Recall: **{metrics['recall']:.4f}**")
    lines.append(f"- F1: **{metrics['f1']:.4f}**")
    if metrics.get("roc_auc") is not None:
        lines.append(f"- ROC-AUC: **{metrics['roc_auc']:.4f}**")
    if metrics.get("average_precision") is not None:
        lines.append(f"- PR-AUC / Average Precision: **{metrics['average_precision']:.4f}**")
    lines.append(f"- FP / FN / TP / TN: **{metrics['fp']} / {metrics['fn']} / {metrics['tp']} / {metrics['tn']}**")
    lines.append(f"- Best F1 in threshold scan: **{best_f1['f1']}** at threshold **{best_f1['threshold']}**")
    lines.append(f"- Deep FP analysis triggered: **{fp_deep_needed}**")
    lines.append("")

    if graph_paths:
        lines.append("## Graphs")
        lines.append("")
        descriptions = {
            "graph_threshold_metrics.png": "Precision, recall, F1 and accuracy across decision thresholds.",
            "graph_confusion_matrix.png": "Counts of TN, FP, FN and TP.",
            "graph_prob1_by_error_type.png": "Distribution of boundary probabilities grouped by error type.",
            "graph_high_conf_fp_original_labels.png": "Original labels among high-confidence false positives.",
            "graph_roc_curve.png": "ROC curve: threshold-independent tradeoff between true-positive rate and false-positive rate.",
            "graph_precision_recall_curve.png": "Precision-Recall curve: especially useful for imbalanced boundary/non-boundary data.",
        }
        for graph in graph_paths:
            rel = graph.name
            lines.append(f"### {rel}")
            lines.append("")
            lines.append(descriptions.get(rel, "Evaluation graph."))
            lines.append("")
            lines.append(f"![{rel}]({rel})")
            lines.append("")

    lines.append("## Saved files")
    lines.append("")
    for saved in saved_files:
        lines.append(f"- `{saved.name}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

def write_metrics_summary(
    path: Path,
    args: argparse.Namespace,
    metrics: Dict[str, float],
    error_counts: pd.Series,
    threshold_df: pd.DataFrame,
    fp_deep_needed: bool,
    high_conf_fp_count: int,
    merge_notes: List[str],
    saved_files: List[Path],
) -> None:
    best_f1 = threshold_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    best_precision_with_recall = threshold_df[threshold_df["recall"] >= 0.5]
    if not best_precision_with_recall.empty:
        best_precision_row = best_precision_with_recall.sort_values(["precision", "f1"], ascending=False).iloc[0]
    else:
        best_precision_row = None

    lines: List[str] = []
    lines.append("Boundary Evaluation Summary")
    lines.append("=" * 80)
    lines.append(f"Model dir: {args.model_dir}")
    lines.append(f"Decision threshold used for predictions: {args.threshold}")
    lines.append(f"Eval CSV: {args.eval_csv}")
    lines.append(f"Random seed: {args.random_seed}")
    lines.append("")

    lines.append("Main metrics")
    lines.append("-" * 80)
    lines.append(f"Total evaluated words: {metrics['total_evaluated_words']}")
    lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
    lines.append(f"Precision: {metrics['precision']:.4f}")
    lines.append(f"Recall:    {metrics['recall']:.4f}")
    lines.append(f"F1:        {metrics['f1']:.4f}")
    lines.append(f"Specificity / true-negative rate: {metrics['specificity']:.4f}")
    if metrics.get("roc_auc") is not None:
        lines.append(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    if metrics.get("average_precision") is not None:
        lines.append(f"PR-AUC / Average Precision: {metrics['average_precision']:.4f}")
    lines.append("")

    lines.append("Confusion matrix breakdown, labels=[0, 1]")
    lines.append("-" * 80)
    lines.append(f"TN - true 0 predicted 0: {metrics['tn']}")
    lines.append(f"FP - true 0 predicted 1: {metrics['fp']}")
    lines.append(f"FN - true 1 predicted 0: {metrics['fn']}")
    lines.append(f"TP - true 1 predicted 1: {metrics['tp']}")
    lines.append(f"True boundaries: {metrics['true_boundary_count']}")
    lines.append(f"Predicted boundaries: {metrics['predicted_boundary_count']}")
    lines.append(f"FP among predicted boundaries: {metrics['fp_among_predicted_boundaries']:.4f}")
    lines.append("")

    lines.append("Error counts")
    lines.append("-" * 80)
    for label in ["TN", "FP", "FN", "TP"]:
        lines.append(f"{label}: {int(error_counts.get(label, 0))}")
    lines.append("")

    lines.append("Threshold scan")
    lines.append("-" * 80)
    lines.append(
        f"Best F1 threshold in tested grid: {best_f1['threshold']} "
        f"(F1={best_f1['f1']}, precision={best_f1['precision']}, recall={best_f1['recall']})"
    )
    if best_precision_row is not None:
        lines.append(
            f"Best precision threshold with recall >= 0.5: {best_precision_row['threshold']} "
            f"(precision={best_precision_row['precision']}, recall={best_precision_row['recall']}, F1={best_precision_row['f1']})"
        )
    lines.append("See threshold_summary.csv for the full threshold table.")
    lines.append("")

    lines.append("False-positive deep analysis decision")
    lines.append("-" * 80)
    lines.append(f"High-confidence FP threshold: {args.high_conf_fp_threshold}")
    lines.append(f"High-confidence FP count: {high_conf_fp_count}")
    lines.append(f"Deep FP analysis triggered: {fp_deep_needed}")
    if fp_deep_needed:
        lines.append("Reason: enough high-confidence FP examples and/or high FP share among predicted boundaries.")
    else:
        lines.append("Reason: not enough high-confidence FP examples and FP share is not high enough.")
    lines.append("")

    if merge_notes:
        lines.append("Original-label merge notes")
        lines.append("-" * 80)
        lines.extend(merge_notes)
        lines.append("")

    lines.append("Saved files")
    lines.append("-" * 80)
    for saved in saved_files:
        lines.append(str(saved))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_inference:
        if not args.predictions_csv:
            raise ValueError("--skip-inference requires --predictions-csv")
        df = pd.read_csv(args.predictions_csv)
        required = {"audio_file", "word_index", "word", "true_label", "pred_label", "prob_1"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Predictions CSV is missing required columns: {sorted(missing)}")
    else:
        df = run_inference(args, output_dir)

    df = add_error_type(df)

    predictions_path = output_dir / "predictions_with_errors.csv"
    df.to_csv(predictions_path, index=False)

    metrics = compute_metrics(df["true_label"], df["pred_label"], df["prob_1"])
    error_counts = df["error_type"].value_counts()

    threshold_df = make_threshold_summary(df, DEFAULT_THRESHOLDS)
    threshold_path = output_dir / "threshold_summary.csv"
    threshold_df.to_csv(threshold_path, index=False)

    manual_review = make_manual_review_examples(df, args)
    manual_review_path = output_dir / "manual_review_examples.csv"
    manual_review.to_csv(manual_review_path, index=False)

    fp = df[df["error_type"] == "FP"].copy()
    high_conf_fp = fp[fp["prob_1"] >= args.high_conf_fp_threshold].copy()
    high_conf_fp_count = len(high_conf_fp)

    # The deeper FP files (context around each error, original pre-binary
    # labels) are only worth generating when there's actually a real FP
    # problem to look at - a handful of one-off errors don't need it. Two
    # ways to trip that: enough *confident* false positives on their own
    # (the model being confidently wrong is the more concerning case), or
    # false positives making up a large enough share of everything the
    # model flagged as a boundary.
    predicted_boundary_count = max(metrics["predicted_boundary_count"], 1)
    fp_share_among_predictions = metrics["fp"] / predicted_boundary_count
    fp_deep_needed = (
        high_conf_fp_count >= args.min_high_conf_fp_for_deep_analysis
        or fp_share_among_predictions >= args.fp_rate_trigger
    ) and metrics["fp"] > 0

    saved_files = [predictions_path, threshold_path, manual_review_path]
    merge_notes: List[str] = []
    original_label_summary: Optional[pd.DataFrame] = None

    if fp_deep_needed:
        fp_context = with_context(
            df,
            high_conf_fp.sort_values("prob_1", ascending=False).head(args.max_review_per_type * 2),
            args.context_window,
            "high_conf_false_positives",
        )
        fp_context_path = output_dir / "high_conf_false_positive_context.csv"
        fp_context.to_csv(fp_context_path, index=False)
        saved_files.append(fp_context_path)

        merged_clean, merge_notes = load_and_merge_original_labels(df, args)
        if merged_clean is not None:
            merged_clean_path = output_dir / "predictions_with_original_labels_clean.csv"
            merged_clean.to_csv(merged_clean_path, index=False)
            saved_files.append(merged_clean_path)

            original_label_summary = make_high_conf_fp_original_label_summary(
                merged_clean,
                args.high_conf_fp_threshold,
            )
            original_label_summary_path = output_dir / "high_conf_fp_original_label_summary.csv"
            original_label_summary.to_csv(original_label_summary_path, index=False)
            saved_files.append(original_label_summary_path)

    graph_paths: List[Path] = []
    report_path: Optional[Path] = None
    if not args.no_graphs:
        graph_paths = create_graphs(
            output_dir=output_dir,
            df=df,
            metrics=metrics,
            threshold_df=threshold_df,
            original_label_summary=original_label_summary,
        )
        saved_files.extend(graph_paths)

        report_path = output_dir / "evaluation_report.md"
        write_evaluation_report(
            report_path,
            args,
            metrics,
            threshold_df,
            fp_deep_needed,
            saved_files,
            graph_paths,
        )
        saved_files.append(report_path)

    summary_path = output_dir / "metrics_summary.txt"
    write_metrics_summary(
        summary_path,
        args,
        metrics,
        error_counts,
        threshold_df,
        fp_deep_needed,
        high_conf_fp_count,
        merge_notes,
        saved_files + [summary_path],
    )
    saved_files.append(summary_path)

    print("\n" + "=" * 80)
    print("Evaluation complete")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"FP: {metrics['fp']} | FN: {metrics['fn']} | TP: {metrics['tp']} | TN: {metrics['tn']}")
    print(f"Deep FP analysis triggered: {fp_deep_needed}")
    print("\nSaved files:")
    for path in saved_files:
        print("-", path)


if __name__ == "__main__":
    main()
