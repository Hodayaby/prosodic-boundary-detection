"""Convert MARSEC's phoneme-level tabular annotation into per-word data,
for a fair head-to-head against Oz's model (see KAN-65/KAN-66) - MARSEC is
British BBC speech from the 1980s, which neither model has trained on,
unlike our own LLMs/sherlock/merlin data.

Source: https://www.ortolang.fr/market/corpora/sldr000033 (free, no signup),
one CSV per session under §doc/Tabular/ (A.csv, B.csv, ..., M.csv - no I/L).
Each row is one phoneme, tagged with the word it belongs to and a set of
"is this the first/last unit of X" position flags. We only need two of them:

- word/start_s/end_s: not given directly. Reconstructed by summing the
  `duration` column in row order - verified against the real audio
  (downloaded Sounds/A/A01/A01.wav: 8,854,512 bytes, 16kHz mono 16-bit PCM
  -> 276.702s, exactly matching the reconstructed total for recording A01).
  Silence between words is its own "#"-word row with a real duration, so
  summing through it (not skipping it) is what keeps the timeline correct -
  it's dropped from the final word-level output, but only after its
  duration has already been folded into the running total.
- boundary: MARSEC's `wfi` column ("word is final of intonation-unit") is
  constant across every phoneme of a word (verified: 0 mismatches across
  ~8,900 words sampled), so it's already a clean per-word 0/1 label. It
  marks intonation-unit boundaries, which are more frequent than the
  clause/sentence boundaries our model was trained on (~23% of words here
  vs. ~9% in our own data) - a related but not identical concept, so
  treat results on this data as approximate.

Critically: `recording` (e.g. "A01") is not a sub-segment of one bigger
file - it IS the file (one recording = one standalone .wav, its own
timeline starting at 0). That's what made the duration check above
possible, and it's why timing is reset per recording below rather than
accumulated across an entire session's CSV.
"""

import argparse
from pathlib import Path

import pandas as pd

PAUSE_WORD = "#"


def load_tabular_csv(csv_path: Path) -> pd.DataFrame:
    """Load one MARSEC §doc/Tabular/<letter>.csv file.

    Input: csv_path - path to the downloaded session CSV (e.g. A.csv).
    Output: the raw phoneme-level DataFrame, unmodified.
    """
    df = pd.read_csv(csv_path)
    required = {"recording", "word", "duration", "wfi", "piw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}. Found: {list(df.columns)}")
    return df


def words_for_recording(phoneme_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse one recording's phoneme rows into per-word start_s/end_s/boundary.

    A new word token starts wherever `piw` ("phoneme is initial of word") is
    1 - not wherever the word text changes, since two consecutive word
    tokens can share the same text (a stutter like "the the", or two
    pauses in a row) and would otherwise get wrongly merged into one.

    Input: phoneme_rows - rows for a single `recording` value, in original
        (file) order - that order is what the cumulative-duration timeline
        depends on, so callers must not have re-sorted it.
    Output: DataFrame with word/start_s/end_s/boundary, pause ("#") rows
        dropped, one row per real word.
    """
    words = []
    t = 0.0
    current = None

    def flush():
        if current is not None and current["word"] != PAUSE_WORD:
            words.append({"word": current["word"], "start_s": current["start_s"], "end_s": current["end_s"], "boundary": current["boundary"]})

    for _, row in phoneme_rows.iterrows():
        word = row["word"]
        duration = float(row["duration"])
        wfi = int(row["wfi"])
        is_word_start = int(row["piw"]) == 1

        if current is None or is_word_start:
            flush()
            current = {"word": word, "start_s": t, "end_s": t, "boundary": wfi}
        elif word != current["word"]:
            raise ValueError(
                f"Word text changed mid-token without piw=1 at t={t:.3f}s: "
                f"'{current['word']}' -> '{word}' - the piw flag and word column disagree."
            )
        elif wfi != current["boundary"]:
            raise ValueError(
                f"Inconsistent wfi within word '{word}' at t={t:.3f}s - expected the same value "
                "across every phoneme of one word (see module docstring)."
            )

        t += duration
        current["end_s"] = t

    flush()

    return pd.DataFrame(words, columns=["word", "start_s", "end_s", "boundary"])


def convert_session(csv_path: Path, output_dir: Path) -> list:
    """Convert one session CSV into one word-level CSV per recording.

    Input: csv_path - a MARSEC Tabular session CSV; output_dir - where to
        write the per-recording CSVs (created if missing).
    Output: list of (recording_name, output_path, num_words) written.
    """
    df = load_tabular_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for recording, group in df.groupby("recording", sort=False):
        words_df = words_for_recording(group)
        out_path = output_dir / f"{recording}.csv"
        words_df.to_csv(out_path, index=False)
        written.append((recording, out_path, len(words_df)))

    return written


def write_oz_input(word_csvs: list, output_path: Path) -> int:
    """Concatenate several of our word-level CSVs into Oz's word/boundary
    input shape (his LLAMA_training.py's load_word_sequence expects exactly
    these two columns, no timing).

    Input: word_csvs - paths to word-level CSVs (as produced by
        convert_session); output_path - where to write the combined file.
    Output: total number of words written.
    """
    parts = [pd.read_csv(p)[["word", "boundary"]] for p in word_csvs]
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(output_path, index=False)
    return len(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MARSEC tabular annotation into per-word CSVs.")
    parser.add_argument("session_csvs", nargs="+", type=Path, help="One or more §doc/Tabular/<letter>.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path("marsec_words"), help="Where to write per-recording word CSVs.")
    parser.add_argument(
        "--oz-input", type=Path, default=None,
        help="If given, also write a combined word/boundary CSV (all recordings, all sessions) at this path, ready for Oz's PREDICT_FILES.",
    )
    args = parser.parse_args()

    all_written = []
    for session_csv in args.session_csvs:
        written = convert_session(session_csv, args.output_dir)
        all_written.extend(written)
        for recording, out_path, num_words in written:
            print(f"{recording}: {num_words} words -> {out_path}")

    print(f"\n{len(all_written)} recordings converted, {sum(n for _, _, n in all_written)} words total.")

    if args.oz_input:
        n = write_oz_input([p for _, p, _ in all_written], args.oz_input)
        print(f"Combined Oz-format input ({n} words) -> {args.oz_input}")
