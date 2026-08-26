import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Paths
# ============================================================

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "splits")

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_all_training_data():
    """
    Loads all *_training_data.csv files created by prepare_data_with_alignment.py.
    Each row is one audio chunk.
    """
    training_files = glob.glob(
        os.path.join(DATA_DIR, "**", "*_training_data.csv"),
        recursive=True
    )

    if not training_files:
        raise FileNotFoundError(f"No *_training_data.csv files found in {DATA_DIR}")

    dfs = []

    for path in training_files:
        df = pd.read_csv(path)
        source_name = os.path.basename(path).replace("_training_data.csv", "")

        folder = os.path.dirname(path)

        df["source_file"] = source_name
        df["audio_path"] = df["audio_file"].apply(
            lambda filename: os.path.join(folder, filename)
        )
        
        dfs.append(df)

        print(f"Loaded training data: {path}, rows={len(df)}")

    all_chunks = pd.concat(dfs, ignore_index=True)

    if "audio_file" not in all_chunks.columns:
        raise ValueError("Expected column 'audio_file' in training data")

    if "text" not in all_chunks.columns:
        raise ValueError("Expected column 'text' in training data")

    return all_chunks


def load_all_word_alignment():
    """
    Loads all *_word_alignment.csv files created by prepare_data_with_alignment.py.
    Each row is one word.
    """
    alignment_files = glob.glob(
        os.path.join(DATA_DIR, "**", "*_word_alignment.csv"),
        recursive=True
    )

    if not alignment_files:
        raise FileNotFoundError(f"No *_word_alignment.csv files found in {DATA_DIR}")

    dfs = []

    for path in alignment_files:
        df = pd.read_csv(path)
        source_name = os.path.basename(path).replace("_word_alignment.csv", "")

        folder = os.path.dirname(path)

        df["source_file"] = source_name
        df["audio_path"] = df["audio_file"].apply(
            lambda filename: os.path.join(folder, filename)
        )
        
        dfs.append(df)

        print(f"Loaded word alignment: {path}, rows={len(df)}")

    all_words = pd.concat(dfs, ignore_index=True)

    if "audio_file" not in all_words.columns:
        raise ValueError("Expected column 'audio_file' in word alignment")

    if "word_index" not in all_words.columns:
        raise ValueError("Expected column 'word_index' in word alignment")

    if "binary_label" not in all_words.columns:
        raise ValueError("Expected column 'binary_label' in word alignment")

    return all_words


def add_chunk_statistics(chunks_df, words_df):
    """
    Adds useful statistics per audio chunk:
    number of words, number of boundary labels, whether the chunk contains boundary.
    This helps us stratify the split.
    """
    stats = (
        words_df
        .groupby("audio_file")
        .agg(
            num_words=("word", "count"),
            num_boundaries=("binary_label", "sum"),
        )
        .reset_index()
    )

    stats["has_boundary"] = (stats["num_boundaries"] > 0).astype(int)

    chunks_df = chunks_df.merge(stats, on="audio_file", how="left")

    chunks_df["num_words"] = chunks_df["num_words"].fillna(0).astype(int)
    chunks_df["num_boundaries"] = chunks_df["num_boundaries"].fillna(0).astype(int)
    chunks_df["has_boundary"] = chunks_df["has_boundary"].fillna(0).astype(int)

    return chunks_df


def split_chunks(chunks_df):
    """
    Splits chunks into train / validation / test.

    Splitting is done at the chunk level, not the word level - words inside
    the same chunk are neighbors in the same recording and highly
    correlated, so splitting by word would leak context between train and
    test (a model could partly "recognize" a test recording it already saw
    most of during training). filter_words_by_chunks() below then pulls the
    matching words for whichever chunks ended up in each split.

    We stratify by has_boundary when possible, so every split has
    a reasonable amount of chunks with boundaries.
    """
    stratify_col = chunks_df["has_boundary"]

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        chunks_df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify_col if stratify_col.nunique() > 1 else None
    )

    # Second split: validation vs test
    relative_valid_ratio = VALID_RATIO / (VALID_RATIO + TEST_RATIO)

    temp_stratify_col = temp_df["has_boundary"]

    valid_df, test_df = train_test_split(
        temp_df,
        train_size=relative_valid_ratio,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=temp_stratify_col if temp_stratify_col.nunique() > 1 else None
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def filter_words_by_chunks(words_df, chunks_df):
    """
    Keeps only words whose audio_file appears in the given chunk split.
    """
    audio_files = set(chunks_df["audio_file"])
    return words_df[words_df["audio_file"].isin(audio_files)].reset_index(drop=True)


def print_split_summary(name, chunks_df, words_df):
    total_words = len(words_df)
    total_boundaries = int(words_df["binary_label"].sum())
    boundary_ratio = total_boundaries / total_words if total_words > 0 else 0

    print("\n" + "=" * 80)
    print(f"{name} summary")
    print("=" * 80)
    print(f"Chunks: {len(chunks_df)}")
    print(f"Words: {total_words}")
    print(f"Boundaries: {total_boundaries}")
    print(f"Boundary ratio: {boundary_ratio:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    chunks_df = load_all_training_data()
    words_df = load_all_word_alignment()

    chunks_df = add_chunk_statistics(chunks_df, words_df)

    print("\nFull dataset summary")
    print("Chunks:", len(chunks_df))
    print("Words:", len(words_df))
    print("Boundaries:", int(words_df["binary_label"].sum()))
    print("Boundary ratio:", words_df["binary_label"].mean())

    train_chunks, valid_chunks, test_chunks = split_chunks(chunks_df)

    train_words = filter_words_by_chunks(words_df, train_chunks)
    valid_words = filter_words_by_chunks(words_df, valid_chunks)
    test_words = filter_words_by_chunks(words_df, test_chunks)

    print_split_summary("Train", train_chunks, train_words)
    print_split_summary("Validation", valid_chunks, valid_words)
    print_split_summary("Test", test_chunks, test_words)

    train_chunks.to_csv(os.path.join(OUTPUT_DIR, "train_chunks.csv"), index=False)
    valid_chunks.to_csv(os.path.join(OUTPUT_DIR, "valid_chunks.csv"), index=False)
    test_chunks.to_csv(os.path.join(OUTPUT_DIR, "test_chunks.csv"), index=False)

    train_words.to_csv(os.path.join(OUTPUT_DIR, "train_words.csv"), index=False)
    valid_words.to_csv(os.path.join(OUTPUT_DIR, "valid_words.csv"), index=False)
    test_words.to_csv(os.path.join(OUTPUT_DIR, "test_words.csv"), index=False)

    print("\nSaved split files to:", OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
