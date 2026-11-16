"""
Figures for the Wavelet vs. our model comparison (report section 7).
Run compare.py first; it writes the per-word CSVs this script reads.
Output: figures/*.png (300 dpi, light background for the report).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

OURS, WAV = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})

valid = pd.read_csv(os.path.join(HERE, "valid_ours_vs_wavelet.csv"))
test = pd.read_csv(os.path.join(HERE, "test_ours_vs_wavelet.csv"))
y = test["true_label"]
SOURCES = {"How_I_use_LLMs": "Karpathy lecture", "merlin_audio": "Merlin", "sherlock_audio": "Sherlock"}
WAV_THRESHOLD = 1.031  # best validation F1, see results.txt


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300)
    plt.close(fig)


# 1. ROC curves
fig, ax = plt.subplots(figsize=(5, 4.5))
for col, color, label in [("prob_1", OURS, "Our model"), ("boundary", WAV, "Wavelet toolkit")]:
    fpr, tpr, _ = roc_curve(y, test[col])
    ax.plot(fpr, tpr, color=color, label=f"{label} (AUC = {roc_auc_score(y, test[col]):.3f})")
ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Chance")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curve, test set ({len(test):,} words)",
       xlim=(0, 1), ylim=(0, 1.01))
ax.legend(loc="lower right")
save(fig, "fig1_roc_test.png")

# 2. Precision-recall curves
fig, ax = plt.subplots(figsize=(5, 4.5))
for col, color, label in [("prob_1", OURS, "Our model"), ("boundary", WAV, "Wavelet toolkit")]:
    p, r, _ = precision_recall_curve(y, test[col])
    ax.plot(r, p, color=color, label=f"{label} (AP = {average_precision_score(y, test[col]):.3f})")
ax.axhline(y.mean(), color=MUTED, linestyle="--", linewidth=1, label=f"Base rate ({y.mean():.3f})")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve, test set", xlim=(0, 1), ylim=(0, 1.01))
ax.legend(loc="lower left")
save(fig, "fig2_pr_test.png")

# 3. F1 per source, both models at their thresholds
groups = [("All test", test)] + [(SOURCES[s], g) for s, g in test.groupby("source_file")]
f_ours = [f1_score(g["true_label"], g["prob_1"] >= 0.5) for _, g in groups]
f_wav = [f1_score(g["true_label"], g["boundary"] >= WAV_THRESHOLD) for _, g in groups]
x = np.arange(len(groups))
w = 0.36
fig, ax = plt.subplots(figsize=(6.5, 4))
b1 = ax.bar(x - w / 2 - 0.01, f_ours, w, color=OURS, label="Our model (threshold 0.5)")
b2 = ax.bar(x + w / 2 + 0.01, f_wav, w, color=WAV, label=f"Wavelet toolkit (threshold {WAV_THRESHOLD}, from validation)")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x, [f"{n}\n({int(g['true_label'].sum())} boundaries)" for n, g in groups])
ax.set(ylabel="F1", ylim=(0, 1), title="F1 on the test set, by recording")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
save(fig, "fig3_f1_by_source.png")

# 4. Threshold sweep on validation (each model on its own scale, so two panels)
sweep = pd.read_csv(os.path.join(HERE, "threshold_sweep.csv"))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, model, color, title, chosen in [
    (axes[0], "ours", OURS, "Our model: prob_1 threshold", 0.5),
    (axes[1], "wavelet", WAV, "Wavelet: boundary-strength threshold", WAV_THRESHOLD),
]:
    g = sweep[sweep["model"] == model].sort_values("threshold")
    ax.plot(g["threshold"], g["valid_F1"], color=color, label="F1")
    ax.plot(g["threshold"], g["valid_P"], color=color, linestyle=":", linewidth=1.5, label="Precision")
    ax.plot(g["threshold"], g["valid_R"], color=color, linestyle="--", linewidth=1.5, label="Recall")
    ax.axvline(chosen, color=MUTED, linewidth=1)
    ax.text(chosen, 1.02, f" used: {chosen}", color=MUTED, fontsize=9, va="bottom")
    ax.set(title=title, xlabel="Threshold", ylim=(0, 1.08))
    ax.legend(loc="lower left" if model == "ours" else "upper right")
axes[0].set_ylabel("Score on validation set")
save(fig, "fig4_threshold_sweep_validation.png")

# 5. Calibration after Platt scaling (10 equal-size bins)
fig, ax = plt.subplots(figsize=(5, 4.5))
for col, color, label in [("prob_1", OURS, "Our model"), ("wavelet_prob", WAV, "Wavelet after Platt scaling")]:
    bins = pd.qcut(test[col].rank(method="first"), 10, labels=False)
    g = test.groupby(bins).agg(pred=(col, "mean"), obs=("true_label", "mean"))
    ax.plot(g["pred"], g["obs"], color=color, marker="o", markersize=6, label=label)
ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
ax.set(xlabel="Mean predicted probability", ylabel="Observed boundary rate",
       title="Calibration on the test set", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="upper left")
save(fig, "fig5_calibration_test.png")

# 6. Score distributions for boundary vs. non-boundary words
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, col, color, title in [(axes[0], "prob_1", OURS, "Our model: prob_1"),
                              (axes[1], "boundary", WAV, "Wavelet: boundary strength")]:
    data = [test.loc[y == 0, col], test.loc[y == 1, col]]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for patch, alpha in zip(bp["boxes"], (0.35, 0.9)):
        patch.set(facecolor=color, alpha=alpha, edgecolor=color)
    for part in ("medians", "whiskers", "caps"):
        for line in bp[part]:
            line.set(color=INK, linewidth=1)
    ax.set_xticks([1, 2], [f"No boundary\n(n={len(data[0])})", f"Boundary\n(n={len(data[1])})"])
    ax.set(title=title)
    ax.grid(axis="x", visible=False)
save(fig, "fig6_score_distributions_test.png")

print("Saved:", ", ".join(sorted(os.listdir(OUT))))
