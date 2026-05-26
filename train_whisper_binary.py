import os
import torch
import pandas as pd
import librosa

from datasets import Dataset, Audio, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ============================================================
# Basic settings
# ============================================================

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "english"
TASK = "transcribe"

SHERLOCK_CSV = "data/sherlock_audio_training_data.csv"
MERLIN_CSV = "data/merlin_audio_training_data.csv"

AUDIO_DIR = "data/audio"
OUTPUT_DIR = "whisper-binary-boundary-model"

TRAIN_RATIO = 0.85
RANDOM_SEED = 42
# ============================================================
# Load CSV files
# ============================================================

sherlock_df = pd.read_csv(SHERLOCK_CSV)
merlin_df = pd.read_csv(MERLIN_CSV)

df = pd.concat([sherlock_df, merlin_df], ignore_index=True)

print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))
print(df.head())


# ============================================================
# Prepare paths and target text
# ============================================================

# CSV columns:
# audio_file = wav file name
# text       = target text with binary labels before words
#
# Example:
# 1 okay 0 so 0 they 0 began ...

df["audio"] = df["audio_file"].apply(
    lambda filename: os.path.join(AUDIO_DIR, filename)
)

df["sentence"] = df["text"]

df = df[["audio", "sentence"]]


# ============================================================
# Check missing audio files
# ============================================================

missing_files = []

for audio_path in df["audio"]:
    if not os.path.exists(audio_path):
        missing_files.append(audio_path)

if missing_files:
    print("\nMissing audio files:")
    for path in missing_files[:30]:
        print(path)

    raise FileNotFoundError(
        f"Found {len(missing_files)} missing audio files. "
        "Please make sure all wav files are inside data/audio/"
    )

print(f"\nAll audio files found. Total samples: {len(df)}")


# ============================================================
# Train / validation split
# ============================================================

train_df = df.sample(frac=TRAIN_RATIO, random_state=RANDOM_SEED)
valid_df = df.drop(train_df.index)

print(f"Train ratio: {TRAIN_RATIO:.0%}")
print(f"Validation ratio: {1 - TRAIN_RATIO:.0%}")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(valid_df)}")


# ============================================================
# Convert to HuggingFace Dataset
# ============================================================

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(valid_df.reset_index(drop=True)),
})

# We load audio manually with librosa inside prepare_dataset,
# because cast_column(Audio) may fail with some datasets/pyarrow versions.


# ============================================================
# Load Whisper components
# ============================================================

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)

tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK
)

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=LANGUAGE,
    task=TASK
)

model.config.suppress_tokens = []


# ============================================================
# Check how 0/1 labels are tokenized
# ============================================================

print("\nTokenization check:")
print("'0'  ->", tokenizer.encode("0", add_special_tokens=False))
print("'1'  ->", tokenizer.encode("1", add_special_tokens=False))
print("' 0' ->", tokenizer.encode(" 0", add_special_tokens=False))
print("' 1' ->", tokenizer.encode(" 1", add_special_tokens=False))


# ============================================================
# Preprocessing
# ============================================================

def prepare_dataset(batch):
    """
    Loads one audio file and prepares it for Whisper fine-tuning.

    Input:
        batch["audio"]    = path to wav file
        batch["sentence"] = target text with 0/1 labels before words

    Output:
        input_features = processed audio features for Whisper
        labels         = tokenized target text
    """

    audio_path = batch["audio"]

    # Load audio manually.
    # sr=16000 makes sure the audio sampling rate matches Whisper's expected rate.
    # mono=True converts stereo to mono if needed.
    speech_array, sampling_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # Convert raw audio into Whisper log-mel input features.
    batch["input_features"] = feature_extractor(
        speech_array,
        sampling_rate=sampling_rate
    ).input_features[0]

    # Convert target text into token IDs.
    # Example target:
    # "1 okay 0 so 0 they ..."
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch


dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    num_proc=1
)


# ============================================================
# Data collator
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# ============================================================
# Training settings
# ============================================================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,

    learning_rate=1e-5,
    warmup_steps=20,
    max_steps=200,

    fp16=torch.cuda.is_available(),

    eval_strategy="steps",
    eval_steps=50,

    save_strategy="steps",
    save_steps=50,

    logging_steps=10,

    predict_with_generate=True,
    generation_max_length=300,

    report_to=[],

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


# ============================================================
# Trainer
# ============================================================

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)


# ============================================================
# Train and save
# ============================================================

trainer.train()

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")
