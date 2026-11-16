"""
Build Wavelet Prosody Toolkit inputs for the validation and test chunks.

For every chunk in valid_words.csv / test_words.csv this copies the chunk WAV
and writes a same-named Praat TextGrid with a "words" tier. Word times in the
split tables are absolute (relative to the full recording), while each chunk
WAV starts at its first word's start_s (see prepare_data_with_alignment.py),
so times are shifted by that offset. Gaps between words become empty
intervals, which the toolkit treats as silence.
"""
import os
import shutil
import sys

import pandas as pd
import soundfile as sf

# Folder that holds data/data/splits/ and the chunk audio (not in the repo, see
# the root .gitignore). Set WAVELET_DATA_DIR to it.
PROJECT_DIR = os.environ.get("WAVELET_DATA_DIR")
if not PROJECT_DIR:
    raise SystemExit("Set WAVELET_DATA_DIR to the folder that holds data/data/splits")
SPLITS_DIR = os.path.join(PROJECT_DIR, "data", "data", "splits")
AUDIO_ROOT = os.path.join(PROJECT_DIR, "data")  # audio_path is like data\merlin\x.wav
OUT_ROOT = os.path.dirname(os.path.abspath(__file__))


def textgrid(intervals, xmax):
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0.0",
        f"xmax = {xmax:.4f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "\titem [1]:",
        '\t\tclass = "IntervalTier"',
        '\t\tname = "words"',
        "\t\txmin = 0.0",
        f"\t\txmax = {xmax:.4f}",
        f"\t\tintervals: size = {len(intervals)}",
    ]
    for i, (s, e, text) in enumerate(intervals, 1):
        text = str(text).replace('"', '""')
        lines += [
            f"\t\t\tintervals [{i}]:",
            f"\t\t\t\txmin = {s:.4f}",
            f"\t\t\t\txmax = {e:.4f}",
            f'\t\t\t\ttext = "{text}"',
        ]
    return "\n".join(lines) + "\n"


def build(split):
    words = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}_words.csv"))
    out_dir = os.path.join(OUT_ROOT, "inputs", split)
    os.makedirs(out_dir, exist_ok=True)
    problems = []

    for audio_file, chunk in words.groupby("audio_file", sort=False):
        chunk = chunk.sort_values("word_index")
        src = os.path.join(AUDIO_ROOT, *chunk["audio_path"].iloc[0].replace("\\", "/").split("/"))
        dst = os.path.join(out_dir, audio_file)
        shutil.copyfile(src, dst)
        duration = sf.info(dst).duration

        offset = chunk["start_s"].iloc[0]
        next_starts = list(chunk["start_s"].iloc[1:] - offset) + [duration]
        intervals, cursor = [], 0.0
        for (_, row), next_start in zip(chunk.iterrows(), next_starts):
            s = max(row["start_s"] - offset, cursor)  # never overlap the previous word
            e = min(row["end_s"] - offset, duration)
            if e > next_start:
                # e.g. sherlock_audio_chunk_22 "forgot": its start time in the source table is 26 s too early
                problems.append((audio_file, row["word_index"], row["word"], f"end clipped {e:.2f}->{next_start:.2f}"))
                e = next_start
            if e <= s:
                problems.append((audio_file, row["word_index"], row["word"], "non-positive duration"))
                e = s + 0.01
            if s > cursor + 1e-4:
                intervals.append((cursor, s, ""))
            intervals.append((s, e, row["word"]))
            cursor = e
        if cursor < duration - 1e-4:
            intervals.append((cursor, duration, ""))
        xmax = max(duration, cursor)

        tg_path = os.path.join(out_dir, audio_file.replace(".wav", ".TextGrid"))
        with open(tg_path, "w", encoding="utf-8") as f:
            f.write(textgrid(intervals, xmax))

    print(f"{split}: {words['audio_file'].nunique()} chunks, {len(words)} words, {len(problems)} timing fixes")
    for p in problems[:20]:
        print("  ", p)


if __name__ == "__main__":
    for split in sys.argv[1:] or ["valid", "test"]:
        build(split)
