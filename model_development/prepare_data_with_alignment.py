import pandas as pd
from pydub import AudioSegment
import argparse
import os
from data_cleaning import clean_words_dataframe

OUTPUT_DIR = "data/generated_training_data"
PADDING_MS = 200

def process_audio(audio_path, csv_path):
    print(f"Processing audio: {audio_path}")
    print(f"Reading data from: {csv_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    audio = AudioSegment.from_wav(audio_path)
    df = pd.read_csv(csv_path)

    df = clean_words_dataframe(df, csv_path=csv_path, output_dir=OUTPUT_DIR)

    MAX_DURATION_MS = 30000

    chunks_data = []
    alignment_rows = []

    current_words = []
    current_word_rows = []

    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    chunk_start_ms = df.iloc[0]["start_s"] * 1000
    chunk_index = 0

    def save_current_chunk(chunk_index, chunk_start_ms, chunk_end_ms, current_words, current_word_rows):
        """
        Save one audio chunk, one row in the training CSV,
        and one alignment row per word in the chunk.
        """
        if not current_words:
            return

        chunk_filename = f"{base_name}_chunk_{chunk_index}.wav"
        chunk_path = os.path.join(OUTPUT_DIR, chunk_filename)

        audio_chunk = audio[chunk_start_ms:chunk_end_ms]
        audio_chunk.export(chunk_path, format="wav")

        chunks_data.append({
            "audio_file": chunk_filename,
            "text": " ".join(current_words)
        })

        for word_index_in_chunk, word_row in enumerate(current_word_rows):
            alignment_rows.append({
                "audio_file": chunk_filename,
                "chunk_index": chunk_index,
                "word_index": word_index_in_chunk,
                "global_word_index": word_row["global_word_index"],
                "word": word_row["word"],
                "start_s": word_row["start_s"],
                "end_s": word_row["end_s"],
                "original_label": word_row["original_boundary"],
                "binary_label": int(word_row["boundary"])
            })

    for index, row in df.iterrows():
        tag = row["boundary"]
        word = row["word"]

        current_start_ms = row["start_s"] * 1000
        current_end_ms = row["end_s"] * 1000

        potential_duration = current_end_ms - chunk_start_ms

        if potential_duration > MAX_DURATION_MS:
            # Save the previous chunk before starting a new one
            chunk_end_ms = current_word_rows[-1]["end_s"] * 1000 + PADDING_MS
            chunk_end_ms = min(chunk_end_ms, len(audio))

            save_current_chunk(
                chunk_index=chunk_index,
                chunk_start_ms=chunk_start_ms,
                chunk_end_ms=chunk_end_ms,
                current_words=current_words,
                current_word_rows=current_word_rows
            )

            # Start new chunk
            chunk_index += 1
            chunk_start_ms = current_start_ms

            current_words = [f"{tag} {word}"]
            current_word_rows = [{
                "global_word_index": index,
                "word": word,
                "start_s": row["start_s"],
                "end_s": row["end_s"],
                "boundary": tag,
                "original_boundary": row["original_boundary"]
            }]

        else:
            current_words.append(f"{tag} {word}")
            current_word_rows.append({
                "global_word_index": index,
                "word": word,
                "start_s": row["start_s"],
                "end_s": row["end_s"],
                "boundary": tag,
                "original_boundary": row["original_boundary"]
            })

    # Save last chunk
    if current_words:
        chunk_end_ms = df.iloc[-1]["end_s"] * 1000

        save_current_chunk(
            chunk_index=chunk_index,
            chunk_start_ms=chunk_start_ms,
            chunk_end_ms=chunk_end_ms,
            current_words=current_words,
            current_word_rows=current_word_rows
        )

    final_dataset = pd.DataFrame(chunks_data)
    output_csv = os.path.join(OUTPUT_DIR, f"{base_name}_training_data.csv")
    final_dataset.to_csv(output_csv, index=False)

    alignment_df = pd.DataFrame(alignment_rows)
    alignment_csv = os.path.join(OUTPUT_DIR, f"{base_name}_word_alignment.csv")
    alignment_df.to_csv(alignment_csv, index=False)

    print(f"Done! Created {len(chunks_data)} chunks.")
    print(f"Saved training data to: {output_csv}")
    print(f"Saved word alignment to: {alignment_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut audio and create training data with word alignment")
    parser.add_argument("audio_file", help="Path to the full audio file WAV")
    parser.add_argument("csv_file", help="Path to the transcription CSV file")

    args = parser.parse_args()

    process_audio(args.audio_file, args.csv_file)