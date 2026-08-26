import pandas as pd
import matplotlib.pyplot as plt

# Looks at the model's raw probability (prob_1), not just its 0/1 prediction
# at the default 0.5 cutoff - the actual question here is whether raising the
# decision threshold trades away false positives faster than it costs true
# positives, to find a threshold worth actually using.

# ============================================================
# Input
# ============================================================

INPUT_CSV = "forced_transcript_boundary_predictions.csv"

# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df["true_label"] = df["true_label"].astype(int)
df["pred_label"] = df["pred_label"].astype(int)
df["prob_1"] = df["prob_1"].astype(float)

# ============================================================
# Error types
# ============================================================

df["error_type"] = "TN"
df.loc[(df["true_label"] == 0) & (df["pred_label"] == 1), "error_type"] = "FP"
df.loc[(df["true_label"] == 1) & (df["pred_label"] == 0), "error_type"] = "FN"
df.loc[(df["true_label"] == 1) & (df["pred_label"] == 1), "error_type"] = "TP"

print("=" * 80)
print("BASIC SUMMARY")
print("=" * 80)

print("Total rows:", len(df))
print("True boundaries:", df["true_label"].sum())
print("Predicted boundaries:", df["pred_label"].sum())

print("\nError counts:")
print(df["error_type"].value_counts())

# ============================================================
# Main question: how confident are the false positives?
# ============================================================

print("\n" + "=" * 80)
print("PROBABILITY BY ERROR TYPE")
print("=" * 80)

stats = df.groupby("error_type")["prob_1"].describe()
print(stats)

stats.to_csv("fp_analysis_probability_by_error_type.csv")

# ============================================================
# FP confidence thresholds
# ============================================================

fp = df[df["error_type"] == "FP"].copy()
tp = df[df["error_type"] == "TP"].copy()

thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.985, 0.99, 0.995]

rows = []

for t in thresholds:
    fp_above = fp[fp["prob_1"] >= t]
    tp_above = tp[tp["prob_1"] >= t]

    total_above = len(fp_above) + len(tp_above)

    precision_above = len(tp_above) / total_above if total_above > 0 else 0

    rows.append({
        "threshold": t,
        "FP_above_threshold": len(fp_above),
        "TP_above_threshold": len(tp_above),
        "total_predicted_boundary_above_threshold": total_above,
        "precision_if_using_this_threshold": round(precision_above, 4),
        "percent_of_all_FP_remaining": round(100 * len(fp_above) / len(fp), 2) if len(fp) > 0 else 0,
        "percent_of_all_TP_remaining": round(100 * len(tp_above) / len(tp), 2) if len(tp) > 0 else 0,
    })

threshold_df = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("FP / TP ABOVE CONFIDENCE THRESHOLDS")
print("=" * 80)
print(threshold_df)

threshold_df.to_csv("fp_tp_threshold_confidence_summary.csv", index=False)

# ============================================================
# Top FP examples
# ============================================================

top_fp = fp.sort_values("prob_1", ascending=False)

print("\n" + "=" * 80)
print("TOP 30 FALSE POSITIVES")
print("=" * 80)

print(
    top_fp[
        ["audio_file", "word_index", "word", "true_label", "pred_label", "prob_1"]
    ]
    .head(30)
    .to_string(index=False)
)

top_fp.to_csv("false_positives_sorted_by_prob_1.csv", index=False)

# ============================================================
# Context around top FP
# ============================================================

context_rows = []

for _, row in top_fp.head(50).iterrows():
    audio_file = row["audio_file"]
    word_index = row["word_index"]

    same_audio = df[df["audio_file"] == audio_file].copy()

    context = same_audio[
        (same_audio["word_index"] >= word_index - 5) &
        (same_audio["word_index"] <= word_index + 5)
    ]

    context_text = " ".join(
        [
            f"[{r.word}]" if r.word_index == word_index else str(r.word)
            for r in context.itertuples()
        ]
    )

    context_rows.append({
        "audio_file": audio_file,
        "word_index": word_index,
        "word": row["word"],
        "prob_1": row["prob_1"],
        "context": context_text,
    })

context_df = pd.DataFrame(context_rows)
context_df.to_csv("top_false_positive_context_examples.csv", index=False)

# ============================================================
# Graph 1: distribution of prob_1 for TP and FP
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(fp["prob_1"], bins=30, alpha=0.6, label=f"False Positive (n={len(fp)})")
plt.hist(tp["prob_1"], bins=30, alpha=0.6, label=f"True Positive (n={len(tp)})")

plt.xlabel("prob_1")
plt.ylabel("Count")
plt.title("Boundary probability: False Positives vs True Positives")
plt.legend()
plt.tight_layout()
plt.savefig("graph_fp_vs_tp_probability_distribution.png", dpi=200)
plt.close()

# ============================================================
# Graph 2: how many FP remain as threshold increases
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["FP_above_threshold"],
    marker="o",
    label="False Positives"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["TP_above_threshold"],
    marker="o",
    label="True Positives"
)

plt.xlabel("prob_1 threshold")
plt.ylabel("Count remaining")
plt.title("How FP and TP change when threshold increases")
plt.legend()
plt.tight_layout()
plt.savefig("graph_fp_tp_remaining_by_threshold.png", dpi=200)
plt.close()

# ============================================================
# Graph 3: precision by threshold
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision_if_using_this_threshold"],
    marker="o"
)

plt.xlabel("prob_1 threshold")
plt.ylabel("Precision")
plt.title("Precision when using different probability thresholds")
plt.tight_layout()
plt.savefig("graph_precision_by_threshold.png", dpi=200)
plt.close()

# ============================================================
# Save full evaluation with error type
# ============================================================

df.to_csv("evaluation_with_error_type.csv", index=False)

print("\n" + "=" * 80)
print("SAVED FILES")
print("=" * 80)

print("evaluation_with_error_type.csv")
print("fp_analysis_probability_by_error_type.csv")
print("fp_tp_threshold_confidence_summary.csv")
print("false_positives_sorted_by_prob_1.csv")
print("top_false_positive_context_examples.csv")
print("graph_fp_vs_tp_probability_distribution.png")
print("graph_fp_tp_remaining_by_threshold.png")
print("graph_precision_by_threshold.png")

print("\nDone.")