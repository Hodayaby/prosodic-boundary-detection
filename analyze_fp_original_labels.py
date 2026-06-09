import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Input: clean file created by merge_eval_with_original_labels.py
# ============================================================

INPUT_CSV = "evaluation_with_original_labels_clean.csv"

# FP שהמודל בטוח בהם מאוד
HIGH_CONF_THRESHOLD = 0.985

# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df["true_label"] = df["true_label"].astype(int)
df["pred_label"] = df["pred_label"].astype(int)
df["prob_1"] = df["prob_1"].astype(float)
df["original_label"] = df["original_label"].astype(int)

# ============================================================
# Select high-confidence False Positives
# true_label=0, pred_label=1, prob_1 high
# ============================================================

high_conf_fp = df[
    (df["true_label"] == 0) &
    (df["pred_label"] == 1) &
    (df["prob_1"] >= HIGH_CONF_THRESHOLD)
].copy()

total = len(high_conf_fp)

# ============================================================
# Count original labels 0/1/2/3/4
# ============================================================

counts = (
    high_conf_fp["original_label"]
    .value_counts()
    .reindex([0, 1, 2, 3, 4], fill_value=0)
    .sort_index()
)

percents = (counts / total * 100).round(2) if total > 0 else counts * 0

summary = pd.DataFrame({
    "count": counts,
    "percent": percents
})

num_2_3 = counts.loc[2] + counts.loc[3]
percent_2_3 = (num_2_3 / total * 100) if total > 0 else 0

# ============================================================
# Print clean result
# ============================================================

print("=" * 80)
print("HIGH-CONFIDENCE FP ORIGINAL LABEL ANALYSIS")
print("=" * 80)

print(f"Input file: {INPUT_CSV}")
print(f"Threshold: {HIGH_CONF_THRESHOLD}")
print(f"Total high-confidence FP: {total}\n")

for label in [0, 1, 2, 3, 4]:
    print(f"original_label={label}: {counts.loc[label]} ({percents.loc[label]:.2f}%)")

print(f"\n2+3 together: {num_2_3} / {total} ({percent_2_3:.2f}%)")

# ============================================================
# Save one summary table
# ============================================================

summary.to_csv("high_conf_fp_original_label_summary.csv")

# ============================================================
# Save one graph
# ============================================================

plt.figure(figsize=(7, 5))
counts.plot(kind="bar")

plt.xlabel("Original label")
plt.ylabel("High-confidence FP count")
plt.title(f"Original labels among high-confidence FP\nprob_1 >= {HIGH_CONF_THRESHOLD}")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("high_conf_fp_original_labels.png", dpi=200)
plt.close()

print("\nSaved:")
print("high_conf_fp_original_label_summary.csv")
print("high_conf_fp_original_labels.png")
print("\nDone.")