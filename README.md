# Prosodic Boundary Detection

Detects sentence/prosodic boundaries in speech audio. A user uploads an
audio file and a word-level transcript (CSV), the pipeline validates and
preprocesses the audio, splits it into chunks, aligns the transcript to
those chunks, and runs a trained classifier (on the BIU SLURM cluster) to
predict a boundary probability for every word.

## Requirements

- Python 3.10+
- A BIU (Bar-Ilan University) lab account, for the parts that need the
  remote server (running the classifier)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If `pip` and `python` point to different installations on your machine,
use `python -m pip install -r requirements.txt` instead.

## Running the web app

The app is a local web page (Streamlit). It runs entirely on your own
computer - nothing is uploaded to the internet unless you explicitly
deploy it somewhere.

**1. Connect to BIU's VPN first** (in your browser, a separate tab):

```
https://access.biu.ac.il
```

This step is required before clicking "Run" in the app below - without
it, the app cannot reach the BIU server. It is not required just to open
the app or fill in the form.

**2. Start the app:**

```bash
cd path/to/prosodic-boundary-detection
python -m streamlit run app/app.py
```

(`python -m streamlit` is used instead of plain `streamlit` because the
`streamlit` command isn't always on PATH after installation.)

**3. Open the link the terminal prints**, normally:

```
http://localhost:8501
```

**4. Click "Start"** on the welcome screen.

**5. Connect:** enter your BIU login as a single `username@host` field
(e.g. `agmonlab@slurm-login1.lnx.biu.ac.il`), your lab password, and
optionally a notification email (BIU will email you when the job starts,
finishes, or fails - leave this blank to skip it). Click **"Test
connection"** - this opens and immediately closes a real SSH connection,
so a wrong host, username, or password is caught here rather than after
you've uploaded files. Once it succeeds, click "Continue to upload".

**6. Upload and run:** upload an audio file and a transcript CSV, then
click **Run**. This submits a job to BIU's SLURM cluster and can take a
while on a busy queue. When it finishes, the results table appears with
a "Download results (CSV)" button.

**To stop the app:** go back to the terminal and press `Ctrl+C`, or close
the terminal window. The page stops working once the terminal that's
running it is closed.

## Running the tests

```bash
python -m pytest
```

Some tests are skipped automatically if `data/` (real training data,
excluded from git) isn't present on your machine - that's expected.

## Project structure

```
pipeline/               Core, reusable pipeline logic (no UI code)
  input_validation.py    Checks the uploaded audio + transcript CSV are usable
  audio_preprocessing.py Resamples audio to the format the model was trained on
  chunker.py              Splits long audio into <=30s pieces, cutting only between words
  transcript_alignment.py Checks the transcript's timestamps actually match the audio/chunks
  schema.py                Defines the final output table's columns
  biu_sync.py              Connects to the BIU server (SSH/SFTP), submits and tracks the job
  orchestrator.py           Runs every stage above in order for one job

app/                     The Streamlit web UI
  app.py                   The page itself: connect/upload/results screens, Run button
  pipeline_runner.py        Thin adapter between the UI's raw inputs and pipeline.orchestrator
  style.css                 Visual styling only - no logic

run_pipeline_job.py     Entry point that runs on the BIU side, inside the SLURM job:
                          classifies each chunk, merges the predictions, applies the
                          threshold, and runs quality control before returning results

model_development/      How the model itself was built - training, evaluation, SLURM
                          launchers for both. Not part of the production pipeline above;
                          run_pipeline_job.py just reuses its decoding logic.
  train_whisper_binary.py    Fine-tunes Whisper on the labeled boundary data
  train_whisper2.slurm        SLURM launcher for the above
  evaluate_boundary_final2.py Forced-transcript evaluation, metrics, error analysis
  evaluate_test_model2.slurm  SLURM launcher for the above
  create_splits.py            Builds the fixed train/validation/test split
  prepare_data.py              Data prep
  prepare_data_with_alignment.py Data prep, keeping the word-to-chunk alignment
  data_cleaning.py             Shared label/word cleaning used while preparing data
  fix_llms_csv.py              One-off cleanup for the Karpathy transcript's timings
  analyze_false_positives.py   Manual error-analysis tooling

tests/                   Automated tests, one file per pipeline/ and app/ module
```

Submitting a `model_development/*.slurm` job on BIU: run `sbatch` from the
repo root, e.g. `sbatch model_development/train_whisper2.slurm` - not from
inside `model_development/` itself, since the scripts assume the repo root
as their working directory.

## Transcript CSV format

The transcript CSV must have (or contain columns recognizable as) these
three columns:

| column    | meaning                          |
|-----------|-----------------------------------|
| `word`    | the spoken word                   |
| `start_s` | when the word starts, in seconds  |
| `end_s`   | when the word ends, in seconds    |
