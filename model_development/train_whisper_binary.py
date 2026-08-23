import os
import torch
import pandas as pd
import librosa

from datasets import Dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

# ============================================================
# Basic settings
# ============================================================

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "english"
TASK = "transcribe"

TRAIN_CSV = "data/splits/train_chunks.csv"
VALID_CSV = "data/splits/valid_chunks.csv"
TEST_CSV = "data/splits/test_chunks.csv"

OUTPUT_DIR = "whisper-binary-boundary-model2"

NUM_TRAIN_EPOCHS = 5
RANDOM_SEED = 42

# ============================================================
# Load split CSV files
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Train columns:", train_df.columns.tolist())
print("Validation columns:", valid_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())

print("Train samples:", len(train_df))
print("Validation samples:", len(valid_df))
print("Test samples:", len(test_df))

# ============================================================
# Prepare paths and target text
# ============================================================

# Expected columns:
# audio_path = path to wav file
# text       = target text with binary labels before words

train_df["audio"] = train_df["audio_path"].str.replace("\\", "/", regex=False)
valid_df["audio"] = valid_df["audio_path"].str.replace("\\", "/", regex=False)
test_df["audio"] = test_df["audio_path"].str.replace("\\", "/", regex=False)

train_df["sentence"] = train_df["text"]
valid_df["sentence"] = valid_df["text"]
test_df["sentence"] = test_df["text"]

print("\nExample paths after Linux conversion:")
print("Train:", train_df["audio"].iloc[0])
print("Validation:", valid_df["audio"].iloc[0])
print("Test:", test_df["audio"].iloc[0])

train_df = train_df[["audio", "sentence"]]
valid_df = valid_df[["audio", "sentence"]]
test_df = test_df[["audio", "sentence"]]

# ============================================================
# Check missing audio files
# ============================================================

all_audio_paths = (
    train_df["audio"].tolist()
    + valid_df["audio"].tolist()
    + test_df["audio"].tolist()
)

missing_files = []

for audio_path in all_audio_paths:
    if not os.path.exists(audio_path):
        missing_files.append(audio_path)

if missing_files:
    print("\nMissing audio files:")
    for path in missing_files[:30]:
        print(path)

    raise FileNotFoundError(
        f"Found {len(missing_files)} missing audio files. "
        "Please make sure all wav files are in the correct location."
    )

print(f"\nAll audio files found. Total samples: {len(all_audio_paths)}")

# ============================================================
# Convert to HuggingFace Dataset
# ============================================================

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(valid_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
})

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
    audio_path = batch["audio"]

    speech_array, sampling_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    batch["input_features"] = feature_extractor(
        speech_array,
        sampling_rate=sampling_rate
    ).input_features[0]

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
    num_train_epochs=NUM_TRAIN_EPOCHS,

    fp16=torch.cuda.is_available(),

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=10,

    predict_with_generate=True,
    generation_max_length=448,

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
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)
    ],
)

# ============================================================
# Train and save
# ============================================================

trainer.train()

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")
