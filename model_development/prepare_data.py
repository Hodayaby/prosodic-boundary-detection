import pandas as pd
from pydub import AudioSegment
import argparse
import os

def process_audio(audio_path, csv_path):
    print(f"Processing audio: {audio_path}")
    print(f"Reading data from: {csv_path}")
    
    audio = AudioSegment.from_wav(audio_path)
    df = pd.read_csv(csv_path)

    MAX_DURATION_MS = 30000  
    
    chunks_data = []
    current_words = []
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # משתמשים בזמנים המקוריים והמדויקים מהטבלה כדי למצוא את המיקום הנכון באודיו המלא
    chunk_start_ms = df.iloc[0]['start_s'] * 1000
    chunk_index = 0

    for index, row in df.iterrows():
        tag = row['boundary']
        word = row['word']
        
        current_start_ms = row['start_s'] * 1000
        current_end_ms = row['end_s'] * 1000
        
        potential_duration = current_end_ms - chunk_start_ms
        
        if potential_duration > MAX_DURATION_MS:
            chunk_end_ms = current_start_ms
            
            # החיתוך כאן ניגש לזמנים האבסולוטיים בקובץ הארוך, ויוצר קובץ קצר וחדש
            audio_chunk = audio[chunk_start_ms:chunk_end_ms]
            chunk_filename = f"{base_name}_chunk_{chunk_index}.wav"
            audio_chunk.export(chunk_filename, format="wav")
            
            chunks_data.append({
                'audio_file': chunk_filename,
                'text': " ".join(current_words)
            })
            
            chunk_index += 1
            chunk_start_ms = current_start_ms
            current_words = [f"{tag} {word}"]
            
        else:
            current_words.append(f"{tag} {word}")

    if current_words:
        chunk_end_ms = df.iloc[-1]['end_s'] * 1000 
        audio_chunk = audio[chunk_start_ms:chunk_end_ms]
        chunk_filename = f"{base_name}_chunk_{chunk_index}.wav"
        audio_chunk.export(chunk_filename, format="wav")
        
        chunks_data.append({
            'audio_file': chunk_filename,
            'text': " ".join(current_words)
        })

    final_dataset = pd.DataFrame(chunks_data)
    output_csv = f"{base_name}_training_data.csv"
    final_dataset.to_csv(output_csv, index=False)
    
    print(f"Done! Created {len(chunks_data)} chunks.")
    print(f"Saved training data to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut audio and transcription strictly under 30s")
    parser.add_argument("audio_file", help="Path to the full audio file (WAV)")
    parser.add_argument("csv_file", help="Path to the transcription CSV file")
    
    args = parser.parse_args()
    process_audio(args.audio_file, args.csv_file)