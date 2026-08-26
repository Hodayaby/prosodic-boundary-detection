import os
import pandas as pd


def clean_words_dataframe(df, csv_path=None, output_dir="."):
    """
    Clean and standardize a word-level transcription dataframe.

    Expected output columns:
    - word
    - start_s
    - end_s
    - boundary
    - original_boundary

    This function:
    - renames onset/offset to start_s/end_s if needed
    - validates required columns
    - keeps original_boundary before binary conversion
    - converts labels: 2->0, 3->0, 4->1
    - fills missing boundary with 0
    - drops rows without timing, without shifting labels
    - drops rows with invalid timing
    - sorts rows by start time
    """

    # --------------------------------------------------
    # Standardize column names
    # --------------------------------------------------
    if "start_s" not in df.columns and "onset" in df.columns:
        df = df.rename(columns={"onset": "start_s"})

    if "end_s" not in df.columns and "offset" in df.columns:
        df = df.rename(columns={"offset": "end_s"})

    # --------------------------------------------------
    # Validate required columns
    # --------------------------------------------------
    required_columns = ["word", "start_s", "end_s", "boundary"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}. "
                f"CSV columns are: {list(df.columns)}"
            )

    # --------------------------------------------------
    # Drop rows without words
    # --------------------------------------------------
    missing_word = df["word"].isna()

    if missing_word.sum() > 0:
        print(f"Warning: dropped {missing_word.sum()} rows without word.")
        df = df[~missing_word].copy()

    # --------------------------------------------------
    # Keep original label for evaluation/debugging
    # --------------------------------------------------
    if "original_boundary" not in df.columns:
        df["original_boundary"] = df["boundary"]

    # --------------------------------------------------
    # Clean labels
    # 2 -> 0
    # 3 -> 0
    # 4 -> 1
    # --------------------------------------------------
    df["boundary"] = df["boundary"].fillna(0)

    df["boundary"] = df["boundary"].replace({
        2: 0,
        3: 0,
        4: 1
    })

    df["boundary"] = df["boundary"].astype(int)

    # --------------------------------------------------
    # Drop rows without timing, without moving labels
    # --------------------------------------------------
    missing_time = df["start_s"].isna() | df["end_s"].isna()

    if missing_time.sum() > 0:
        if csv_path is not None:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            dropped_path = os.path.join(
                output_dir,
                f"{base_name}_dropped_no_timing.csv"
            )
            os.makedirs(output_dir, exist_ok=True)
            df[missing_time].to_csv(dropped_path, index=False)
            print(f"Warning: dropped {missing_time.sum()} rows without timing.")
            print(f"Dropped rows saved to: {dropped_path}")
        else:
            print(f"Warning: dropped {missing_time.sum()} rows without timing.")

        df = df[~missing_time].copy()

    # --------------------------------------------------
    # Convert timing columns to float
    # --------------------------------------------------
    df["start_s"] = df["start_s"].astype(float)
    df["end_s"] = df["end_s"].astype(float)

    # --------------------------------------------------
    # Drop invalid timing rows
    # --------------------------------------------------
    invalid_time = df["end_s"] <= df["start_s"]

    if invalid_time.sum() > 0:
        print(f"Warning: dropped {invalid_time.sum()} rows with invalid timing.")
        df = df[~invalid_time].copy()

    # --------------------------------------------------
    # Sort by time
    # --------------------------------------------------
    df = df.sort_values("start_s").reset_index(drop=True)

    print("Cleaned dataframe:")
    print(f"Rows after cleaning: {len(df)}")
    print("Boundary distribution:")
    print(df["boundary"].value_counts().sort_index())

    return df