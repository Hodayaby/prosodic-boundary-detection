"""
Compare the Wavelet Prosody Toolkit boundary strength with our Whisper model
(whisper-binary-boundary-model2) on the same validation and test words.

1. Merge each chunk's .prom rows to our per-word predictions and check that
   word counts, word text and times agree.
2. Ranking metrics on TEST with raw scores: ROC-AUC, PR-AUC (no conversion
   needed, both depend only on the order of the scores).
3. Thresholded P/R/F1 on TEST, Wavelet threshold chosen on VALIDATION
   (best F1). Our model keeps its reported threshold of 0.5.
4. Platt scaling: logistic regression fitted on VALIDATION Wavelet scores,
   applied to TEST, so both models give probabilities; Brier score.
5. Chunk-level bootstrap CI for the AUC differences.

The last word of every chunk is dropped for both models: the toolkit does not
compute a boundary for it and always writes 1.0.

WAVELET_OUT=out_default_config scores a run made with another config; its
outputs get that name as a suffix so the main results are not overwritten.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

# Folder that holds data/data/splits/ and the results_eval_* folders (not in
# the repo, see the root .gitignore). Set WAVELET_DATA_DIR to it.
PROJECT_DIR = os.environ.get("WAVELET_DATA_DIR")
if not PROJECT_DIR:
    raise SystemExit("Set WAVELET_DATA_DIR to the folder that holds data/data/splits and results_eval_*")
PRED_FILES = {
    "valid": os.path.join(PROJECT_DIR, "results_eval_16866867", "results_eval_16866867", "predictions_with_errors.csv"),
    "test": os.path.join(PROJECT_DIR, "results_eval_16866883", "evaluation_outputs2_test_t05", "predictions_with_errors.csv"),
}
WORD_FILES = {s: os.path.join(PROJECT_DIR, "data", "data", "splits", f"{s}_words.csv") for s in PRED_FILES}
HERE = os.path.dirname(os.path.abspath(__file__))
OUR_THRESHOLD = 0.5
N_BOOT = 1000
WAVELET_OUT = os.environ.get("WAVELET_OUT", "out")
SUFFIX = "" if WAVELET_OUT == "out" else f"_{WAVELET_OUT}"


def load_prom(split):
    rows = []
    for path in glob.glob(os.path.join(HERE, WAVELET_OUT, split, "*.prom")):
        p = pd.read_csv(path, sep="\t", header=None, quoting=3, keep_default_na=False,
                        names=["chunk", "w_start", "w_end", "w_word", "prominence", "boundary"])
        p["audio_file"] = p["chunk"] + ".wav"
        p["word_index"] = range(len(p))
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


def merged(split):
    pred = pd.read_csv(PRED_FILES[split])
    words = pd.read_csv(WORD_FILES[split])[["audio_file", "word_index", "start_s", "source_file"]]
    pred = pred.merge(words, on=["audio_file", "word_index"], how="left", validate="one_to_one")
    prom = load_prom(split)

    n_ours = pred.groupby("audio_file").size()
    n_wav = prom.groupby("audio_file").size()
    bad = (n_ours - n_wav.reindex(n_ours.index)).fillna(-999)
    bad = bad[bad != 0]
    if len(bad):
        raise SystemExit(f"{split}: word count mismatch in {len(bad)} chunks:\n{bad}")

    m = pred.merge(prom[["audio_file", "word_index", "w_word", "w_start", "boundary"]],
                   on=["audio_file", "word_index"], how="left", validate="one_to_one")
    word_mismatch = (m["w_word"].astype(str) != m["word"].astype(str)).sum()
    chunk_offset = m.groupby("audio_file")["start_s"].transform("first")
    time_err = (m["w_start"] - (m["start_s"] - chunk_offset)).abs()
    print(f"{split}: {len(m)} words, {m['audio_file'].nunique()} chunks, "
          f"word-text mismatches={word_mismatch}, max start-time diff={time_err.max():.3f}s, "
          f"missing scores={m['boundary'].isna().sum()}")

    last = m.groupby("audio_file")["word_index"].transform("max") == m["word_index"]
    print(f"{split}: dropping {last.sum()} chunk-final words ({int(m.loc[last, 'true_label'].sum())} true boundaries), "
          f"keeping {(~last).sum()}")
    return m[~last].reset_index(drop=True)


def best_threshold(y, score):
    best = (-1, None)
    for t in np.unique(score):
        f = f1_score(y, score >= t, zero_division=0)
        if f > best[0]:
            best = (f, t)
    return best[1], best[0]


def prf(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(P=precision_score(y, pred, zero_division=0), R=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0), TN=tn, FP=fp, FN=fn, TP=tp)


def bootstrap_auc_diff(df, rng):
    chunks = df["audio_file"].unique()
    groups = {c: g for c, g in df.groupby("audio_file")}
    d_roc, d_pr = [], []
    for _ in range(N_BOOT):
        s = pd.concat([groups[c] for c in rng.choice(chunks, len(chunks), replace=True)])
        if s["true_label"].nunique() < 2:
            continue
        y = s["true_label"]
        d_roc.append(roc_auc_score(y, s["prob_1"]) - roc_auc_score(y, s["boundary"]))
        d_pr.append(average_precision_score(y, s["prob_1"]) - average_precision_score(y, s["boundary"]))
    return np.percentile(d_roc, [2.5, 97.5]), np.percentile(d_pr, [2.5, 97.5])


def main():
    valid, test = merged("valid"), merged("test")
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("\n=== TEST ranking metrics (raw scores) ===")
    for name, df in [("ALL", test)] + list(test.groupby("source_file")):
        y = df["true_label"]
        out(f"{name:15s} n={len(df):5d} boundaries={int(y.sum()):4d} | "
            f"ours ROC-AUC={roc_auc_score(y, df['prob_1']):.4f} PR-AUC={average_precision_score(y, df['prob_1']):.4f} | "
            f"wavelet ROC-AUC={roc_auc_score(y, df['boundary']):.4f} PR-AUC={average_precision_score(y, df['boundary']):.4f} | "
            f"base rate={y.mean():.4f}")

    roc_ci, pr_ci = bootstrap_auc_diff(test, np.random.default_rng(42))
    out(f"\nBootstrap over test chunks ({N_BOOT}x), ours minus wavelet: "
        f"ROC-AUC diff 95% CI [{roc_ci[0]:.3f}, {roc_ci[1]:.3f}], PR-AUC diff 95% CI [{pr_ci[0]:.3f}, {pr_ci[1]:.3f}]")

    t_wav, f_val = best_threshold(valid["true_label"].values, valid["boundary"].values)
    out(f"\n=== Thresholded metrics on TEST ===")
    out(f"Wavelet threshold chosen on validation (best F1={f_val:.4f}): {t_wav:.4f}")
    for name, df in [("ALL", test)] + list(test.groupby("source_file")):
        y = df["true_label"]
        o = prf(y, df["prob_1"] >= OUR_THRESHOLD)
        w = prf(y, df["boundary"] >= t_wav)
        out(f"{name:15s} ours    P={o['P']:.4f} R={o['R']:.4f} F1={o['F1']:.4f} (TN={o['TN']} FP={o['FP']} FN={o['FN']} TP={o['TP']})")
        out(f"{'':15s} wavelet P={w['P']:.4f} R={w['R']:.4f} F1={w['F1']:.4f} (TN={w['TN']} FP={w['FP']} FN={w['FN']} TP={w['TP']})")

    out("\n=== Platt scaling (fit on validation, applied to test) ===")
    lr = LogisticRegression().fit(valid[["boundary"]], valid["true_label"])
    test["wavelet_prob"] = lr.predict_proba(test[["boundary"]])[:, 1]
    a, b = lr.coef_[0][0], lr.intercept_[0]
    out(f"p = sigmoid({a:.4f} * boundary_strength + {b:.4f})")
    y = test["true_label"]
    out(f"Brier score (lower is better): ours={brier_score_loss(y, test['prob_1']):.4f} "
        f"wavelet={brier_score_loss(y, test['wavelet_prob']):.4f} "
        f"always-base-rate={brier_score_loss(y, np.full(len(y), valid['true_label'].mean())):.4f}")
    out(f"ROC-AUC after Platt (should equal raw): wavelet={roc_auc_score(y, test['wavelet_prob']):.4f}")
    w = prf(y, test["wavelet_prob"] >= 0.5)
    out(f"Wavelet at calibrated p>=0.5: P={w['P']:.4f} R={w['R']:.4f} F1={w['F1']:.4f}")

    out("\nCalibration on test (mean predicted vs observed boundary rate, 10 equal-size bins):")
    for col in ["prob_1", "wavelet_prob"]:
        bins = pd.qcut(test[col].rank(method="first"), 10, labels=False)
        g = test.groupby(bins).agg(pred=(col, "mean"), obs=("true_label", "mean"))
        out(f"  {col:13s} " + "  ".join(f"{p:.2f}/{o:.2f}" for p, o in zip(g["pred"], g["obs"])))

    out(f"\nSpearman correlation ours vs wavelet on test: "
        f"{test['prob_1'].rank().corr(test['boundary'].rank()):.4f}")
    out(f"Share of test words with wavelet boundary strength exactly 0: {(test['boundary'] == 0).mean():.4f}")

    test.to_csv(os.path.join(HERE, f"test_ours_vs_wavelet{SUFFIX}.csv"), index=False)
    valid.to_csv(os.path.join(HERE, f"valid_ours_vs_wavelet{SUFFIX}.csv"), index=False)
    with open(os.path.join(HERE, f"results{SUFFIX}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
