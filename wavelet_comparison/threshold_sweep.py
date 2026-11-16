"""
Threshold sweep for both models: P/R/F1 on validation and on test at a grid
of thresholds. The Wavelet threshold used in results.txt is the one with the
best validation F1; the test columns are here only to show what happens, they
were not used to pick anything. Run compare.py first.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
valid = pd.read_csv(os.path.join(HERE, "valid_ours_vs_wavelet.csv"))
test = pd.read_csv(os.path.join(HERE, "test_ours_vs_wavelet.csv"))

GRIDS = {
    "wavelet": ("boundary", np.round(np.arange(0.0, 2.51, 0.05), 2).tolist() + [1.031]),
    "ours": ("prob_1", np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()),
}

rows = []
for model, (col, grid) in GRIDS.items():
    for t in sorted(set(grid)):
        row = {"model": model, "threshold": t}
        for split, df in [("valid", valid), ("test", test)]:
            y, pred = df["true_label"], df[col] >= t
            row[f"{split}_P"] = round(precision_score(y, pred, zero_division=0), 4)
            row[f"{split}_R"] = round(recall_score(y, pred, zero_division=0), 4)
            row[f"{split}_F1"] = round(f1_score(y, pred, zero_division=0), 4)
            row[f"{split}_predicted_boundaries"] = int(pred.sum())
        rows.append(row)

sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(HERE, "threshold_sweep.csv"), index=False)
for model, g in sweep.groupby("model"):
    best = g.loc[g["valid_F1"].idxmax()]
    print(f"{model}: best validation F1 {best['valid_F1']} at threshold {best['threshold']} "
          f"-> test F1 {best['test_F1']}")
