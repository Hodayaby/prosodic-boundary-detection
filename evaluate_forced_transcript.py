import os
import re
import torch
import librosa
import pandas as pd

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

# Avoid cuDNN initialization crash that happened on this pod
torch.backends.cudnn.enabled = False


MODEL_DIR = "whisper-binary-boundary-model"

SHERLOCK_CSV = "data/sherlock_audio_training_data.csv"
MERLIN_CSV = "data/merlin_audio_training_data.csv"
AUDIO_DIR = "data/audio"

TRAIN_RATIO = 0.85
RANDOM_SEED = 42

# If prob_1 >= THRESHOLD, predict boundary = 1
# Start with 0.5. Later we can test 0.3 / 0.4 because the data is imbalanced.
THRESHOLD = 0.5

OUTPUT_CSV = "forced_transcript_boundary_predictions.csv"
THRESHOLDS_CSV = "threshold_results.csv"
ROC_PLOT_PATH = "roc_curve.png"
PR_PLOT_PATH = "precision_recall_curve.png"

# Thresholds to test after probabilities are collected
THRESHOLDS_TO_TEST = np.arange(0.05, 1.00, 0.05)


def parse_labeled_text(text):
    """
    Converts:
        '1 okay 0 so 0 they 1 stopped'

    Into:
        labels = ['1', '0', '0', '1']
        words  = ['okay', 'so', 'they', 'stopped']

    Assumption:
        the data format is always: label word label word ...
    """
    tokens = text.strip().split()

    labels = []
    words = []

    i = 0
    while i < len(tokens) - 1:
        label = tokens[i]
        word = tokens[i + 1]

        if label not in {"0", "1"}:
            # If something unexpected happens, skip one token and continue.
            # This keeps the script from crashing on a malformed row.
            i += 1
            continue

        labels.append(label)
        words.append(word)
        i += 2

    return labels, words


def encode_prefix(tokenizer, prefix_text):
    """
    Tokenizes the current decoder prefix.

    prefix_text is the already known/generated sequence:
        '0 okay 0 so 1 stopped'

    We include Whisper's special start/language/task tokens,
    but remove EOS because we are not done generating yet.
    """
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)

    if len(prefix_ids) > 0 and prefix_ids[-1] == tokenizer.eos_token_id:
        prefix_ids = prefix_ids[:-1]

    return prefix_ids


def predict_labels_for_words(model, processor, audio_path, words, device):
    """
    Main forced-transcript inference.

    Input:
        audio_path = wav file
        words      = known transcript words, without labels

    Output:
        predictions per word:
        word, pred_label, prob_0, prob_1
    """

    tokenizer = processor.tokenizer

    # Candidate tokens for the first label in the sequence:
    # The sequence starts with "0" or "1", without a leading space.
    first_label_token_ids = [
        tokenizer.encode("0", add_special_tokens=False)[0],
        tokenizer.encode("1", add_special_tokens=False)[0],
    ]

    # Candidate tokens for labels after a word:
    # Later labels appear as " 0" or " 1", with a leading space.
    next_label_token_ids = [
        tokenizer.encode(" 0", add_special_tokens=False)[0],
        tokenizer.encode(" 1", add_special_tokens=False)[0],
    ]

    # Load audio
    speech_array, _ = librosa.load(audio_path, sr=16000, mono=True)

    inputs = processor.feature_extractor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(device)

    predictions = []
    prefix_parts = []

    with torch.no_grad():
        # Encode audio once
        encoded_features = model.get_encoder()(input_features).last_hidden_state

        for word_index, word in enumerate(words):
            prefix_text = " ".join(prefix_parts)

            prefix_ids = encode_prefix(tokenizer, prefix_text)
            decoder_input_ids = torch.tensor([prefix_ids], device=device)

            decoder_output = model.get_decoder()(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoded_features
            )

            logits = model.proj_out(decoder_output.last_hidden_state)

            if word_index == 0:
                candidate_token_ids = first_label_token_ids
            else:
                candidate_token_ids = next_label_token_ids

            # Take only logits for label 0 and label 1
            label_logits = logits[0, -1, candidate_token_ids]

            # Softmax over just [0, 1]
            label_probs = torch.softmax(label_logits, dim=0)

            prob_0 = label_probs[0].item()
            prob_1 = label_probs[1].item()

            pred_label = "1" if prob_1 >= THRESHOLD else "0"

            predictions.append({
                "word_index": word_index,
                "word": word,
                "pred_label": pred_label,
                "prob_0": prob_0,
                "prob_1": prob_1,
            })

            # Important:
            # We append the predicted label, but the known transcript word.
            # This is the WPLP-style forced transcript process:
            # audio + known words, model predicts only labels.
            prefix_parts.append(pred_label)
            prefix_parts.append(word)

    return predictions

def evaluate_with_threshold(y_true, y_prob, threshold):
    """
    Compute binary classification metrics for a given threshold.
    """
    y_pred = (np.array(y_prob) >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        pos_label=1,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_thresholds(y_true, y_prob, thresholds):
    """
    Evaluate many thresholds and return:
    1. DataFrame with results for each threshold
    2. Best row according to F1
    """
    rows = []

    for threshold in thresholds:
        rows.append(evaluate_with_threshold(y_true, y_prob, threshold))

    results_df = pd.DataFrame(rows)
    best_row = results_df.sort_values(
        by=["f1", "recall", "precision"],
        ascending=False
    ).iloc[0]

    return results_df, best_row


def plot_roc_curve(y_true, y_prob, output_path):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title("ROC Curve - Boundary Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return auc


def plot_precision_recall_curve(y_true, y_prob, output_path):
    from sklearn.metrics import precision_recall_curve

    pr_auc = average_precision_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Boundary Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return pr_auc


def main():
    # Load data
    sherlock_df = pd.read_csv(SHERLOCK_CSV)
    merlin_df = pd.read_csv(MERLIN_CSV)
    df = pd.concat([sherlock_df, merlin_df], ignore_index=True)

    df["audio"] = df["audio_file"].apply(lambda filename: os.path.join(AUDIO_DIR, filename))
    df["sentence"] = df["text"]

    # Same split as training
    train_df = df.sample(frac=TRAIN_RATIO, random_state=RANDOM_SEED)
    valid_df = df.drop(train_df.index).reset_index(drop=True)

    print(f"Validation samples: {len(valid_df)}")
    print(f"Threshold: {THRESHOLD}")

    # Load model
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = model.to(device)
    model.eval()

    # Show tokenization sanity check
    tokenizer = processor.tokenizer
    print("\nTokenization sanity check:")
    print("'0'  ->", tokenizer.encode("0", add_special_tokens=False))
    print("'1'  ->", tokenizer.encode("1", add_special_tokens=False))
    print("' 0' ->", tokenizer.encode(" 0", add_special_tokens=False))
    print("' 1' ->", tokenizer.encode(" 1", add_special_tokens=False))

    all_rows = []
    all_true = []
    all_pred = []
    all_prob_1 = []
    
    examples_to_print = 3

    for sample_index, row in valid_df.iterrows():
        audio_path = row["audio"]
        audio_file = row["audio_file"]
        true_text = row["sentence"]

        true_labels, words = parse_labeled_text(true_text)

        predictions = predict_labels_for_words(
            model=model,
            processor=processor,
            audio_path=audio_path,
            words=words,
            device=device
        )

        # Align just in case
        n = min(len(true_labels), len(predictions))

        for i in range(n):
            true_label = true_labels[i]
            pred = predictions[i]

            all_true.append(int(true_label))
            all_pred.append(int(pred["pred_label"]))
            all_prob_1.append(float(pred["prob_1"]))

            all_rows.append({
                "sample_index": sample_index,
                "audio_file": audio_file,
                "word_index": pred["word_index"],
                "word": pred["word"],
                "true_label": true_label,
                "pred_label": pred["pred_label"],
                "prob_0": pred["prob_0"],
                "prob_1": pred["prob_1"],
            })

        if sample_index < examples_to_print:
            print("\n" + "=" * 80)
            print(f"Example {sample_index}")
            print("Audio:", audio_path)
            print("\nFirst 40 words:")
            print(words[:40])
            print("\nTrue labels:")
            print(true_labels[:40])
            print("\nPred labels:")
            print([p["pred_label"] for p in predictions[:40]])
            print("\nProb_1:")
            print([round(p["prob_1"], 3) for p in predictions[:40]])

    # Save detailed predictions
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    # Convert to numpy arrays
    all_true = np.array(all_true).astype(int)
    all_prob_1 = np.array(all_prob_1).astype(float)

    # Metrics for the default threshold
    default_metrics = evaluate_with_threshold(
        y_true=all_true,
        y_prob=all_prob_1,
        threshold=THRESHOLD
    )

    # Evaluate multiple thresholds
    threshold_results_df, best_threshold_row = evaluate_thresholds(
        y_true=all_true,
        y_prob=all_prob_1,
        thresholds=THRESHOLDS_TO_TEST
    )

    threshold_results_df.to_csv(THRESHOLDS_CSV, index=False)

    # ROC AUC and PR AUC
    roc_auc = roc_auc_score(all_true, all_prob_1)
    pr_auc = average_precision_score(all_true, all_prob_1)

    plot_roc_curve(all_true, all_prob_1, ROC_PLOT_PATH)
    plot_precision_recall_curve(all_true, all_prob_1, PR_PLOT_PATH)

    print("\n" + "=" * 80)
    print("Forced-transcript boundary evaluation")
    print("=" * 80)

    print(f"Total evaluated labels: {len(all_true)}")

    print("\nDefault threshold evaluation")
    print(f"Threshold: {THRESHOLD}")
    print(f"Accuracy:  {default_metrics['accuracy']:.4f}")
    print(f"Precision: {default_metrics['precision']:.4f}")
    print(f"Recall:    {default_metrics['recall']:.4f}")
    print(f"F1:        {default_metrics['f1']:.4f}")

    print("\nConfusion matrix breakdown:")
    print(f"TN - true 0 predicted 0: {default_metrics['tn']}")
    print(f"FP - true 0 predicted 1: {default_metrics['fp']}")
    print(f"FN - true 1 predicted 0: {default_metrics['fn']}")
    print(f"TP - true 1 predicted 1: {default_metrics['tp']}")

    print("\nThreshold search - best threshold by F1")
    print(f"Best threshold: {best_threshold_row['threshold']:.2f}")
    print(f"Accuracy:       {best_threshold_row['accuracy']:.4f}")
    print(f"Precision:      {best_threshold_row['precision']:.4f}")
    print(f"Recall:         {best_threshold_row['recall']:.4f}")
    print(f"F1:             {best_threshold_row['f1']:.4f}")

    print("\nAUC metrics")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")

    print(f"\nSaved detailed predictions to: {OUTPUT_CSV}")
    print(f"Saved threshold results to: {THRESHOLDS_CSV}")
    print(f"Saved ROC curve to: {ROC_PLOT_PATH}")
    print(f"Saved Precision-Recall curve to: {PR_PLOT_PATH}")


if __name__ == "__main__":
    main()
