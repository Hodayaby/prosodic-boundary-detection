import pandas as pd

# ============================================================
# Files
# ============================================================

EVAL_CSV = "forced_transcript_boundary_predictions.csv"

MERLIN_ALIGN_CSV = "data/data_with_numbers2341/merlin_audio_word_alignment.csv"
SHERLOCK_ALIGN_CSV = "data/data_with_numbers2341/sherlock_audio_word_alignment.csv"

OUTPUT_CSV = "evaluation_with_original_labels.csv"
CLEAN_OUTPUT_CSV = "evaluation_with_original_labels_clean.csv"

# ============================================================
# Load
# ============================================================

eval_df = pd.read_csv(EVAL_CSV)
merlin = pd.read_csv(MERLIN_ALIGN_CSV)
sherlock = pd.read_csv(SHERLOCK_ALIGN_CSV)

# ============================================================
# Clean types
# ============================================================

for df in [eval_df, merlin, sherlock]:
    df["audio_file"] = df["audio_file"].astype(str)
    df["word_index"] = df["word_index"].astype(int)

eval_df["true_label"] = eval_df["true_label"].astype(int)
eval_df["pred_label"] = eval_df["pred_label"].astype(int)
eval_df["prob_1"] = eval_df["prob_1"].astype(float)

# ============================================================
# Add source
# ============================================================

merlin["source"] = "merlin"
sherlock["source"] = "sherlock"

# ============================================================
# Combine alignment files
# ============================================================

align = pd.concat([merlin, sherlock], ignore_index=True)

print("=" * 80)
print("ALIGNMENT LABEL DISTRIBUTION")
print("=" * 80)

print(
    align["original_label"]
    .value_counts(dropna=False)
    .sort_index()
)

# ============================================================
# Merge by audio_file + word_index
# ============================================================

merged = eval_df.merge(
    align,
    on=["audio_file", "word_index"],
    how="left",
    suffixes=("_eval", "_original")
)

# ============================================================
# Word match check
# ============================================================

merged["word_match"] = (
    merged["word_eval"]
    .astype(str)
    .str.lower()
    .str.strip()
    ==
    merged["word_original"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# ============================================================
# Diagnostics
# ============================================================

print("\n" + "=" * 80)
print("MERGE CHECK")
print("=" * 80)

print("Evaluation rows:", len(eval_df))
print("Merged rows:", len(merged))
print("Rows without original_label:", merged["original_label"].isna().sum())

print("\nWord match counts:")
print(merged["word_match"].value_counts(dropna=False))

print("\nOriginal label distribution in merged evaluation:")
print(
    merged["original_label"]
    .value_counts(dropna=False)
    .sort_index()
)

# ============================================================
# Show bad rows
# ============================================================

bad = merged[
    (merged["word_match"] == False) |
    (merged["original_label"].isna())
].copy()

if len(bad) > 0:
    print("\nWARNING: problematic rows found:")
    print("Problematic rows:", len(bad))

    print(
        bad[
            [
                "audio_file",
                "word_index",
                "word_eval",
                "word_original",
                "true_label",
                "pred_label",
                "prob_1",
                "original_label",
                "binary_label",
                "word_match",
            ]
        ]
        .head(80)
        .to_string(index=False)
    )
else:
    print("\nMerge looks clean: all rows matched and all words match.")

# ============================================================
# Save full merged file
# ============================================================

merged.to_csv(OUTPUT_CSV, index=False)

# ============================================================
# Save clean matched file only
# ============================================================

merged_clean = merged[
    (merged["word_match"] == True) &
    (merged["original_label"].notna())
].copy()

merged_clean.to_csv(CLEAN_OUTPUT_CSV, index=False)

print("\nSaved full file:", OUTPUT_CSV)
print("Saved clean file:", CLEAN_OUTPUT_CSV)
print("Clean rows:", len(merged_clean))
print("Removed rows:", len(merged) - len(merged_clean))

print("\nDone.")