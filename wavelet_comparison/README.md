# Comparison with the Wavelet Prosody Toolkit (KAN-81)

Compares our model (`whisper-binary-boundary-model2`) with the
[Wavelet Prosody Toolkit](https://github.com/asuni/wavelet_prosody_toolkit)
(Suni et al., 2017), a non-neural method that scores each word's boundary
strength from F0, energy and word duration. Both systems are scored on the
exact same words of our fixed validation and test splits.

## Results (test split, 3,296 words, 297 boundaries)

| Metric | Our model | Wavelet |
|---|---|---|
| ROC-AUC | 0.979 | 0.868 |
| PR-AUC | 0.863 | 0.360 |
| Precision | 0.820 | 0.374 |
| Recall | 0.768 | 0.741 |
| F1 | 0.793 | 0.497 |
| Brier score | 0.028 | 0.065 |

Our model uses its reported threshold (0.5). The Wavelet threshold (1.031) is
the one with the best F1 on the validation split. A chunk-level bootstrap
(1,000 resamples) gives a 95% CI of [0.087, 0.142] for the ROC-AUC difference
and [0.447, 0.549] for the PR-AUC difference. Full numbers, per-recording
results and calibration are in `results.txt`.

## Method

1. `build_inputs.py` copies each validation/test chunk WAV and writes a
   same-named TextGrid with a `words` tier from the split word tables. Times
   are shifted to start at the chunk start.
2. The toolkit's `prosody_labeller` scores every word with the
   `libritts_boundary.yaml` config (the toolkit's boundary-detection config).
   Its output is in `out/`.
3. `compare.py` joins the scores to our per-word predictions, checks word
   counts, word text and times, and computes ROC-AUC, PR-AUC, thresholded
   P/R/F1, Platt scaling (fit on validation), Brier score and the bootstrap CI.
4. `threshold_sweep.py` writes P/R/F1 per threshold for both models.
5. `make_figures.py` draws the six figures in `figures/`.

Notes:

- The last word of every chunk is dropped for both models: the toolkit does
  not compute a boundary for it and always writes 1.0.
- In the original Sherlock table, "I forgot" (said at about 676 s) has start
  times 26 s too early, so in our split tables "I" sits at the end of
  `sherlock_audio_chunk_21` and "forgot" at the start of
  `sherlock_audio_chunk_22`. For the toolkit, "forgot" is clipped to end
  where the next word starts. This is one non-boundary word in the test split.
- The toolkit gets the gold word times from the transcript, which our model
  does not use, and was run with its stock config, not tuned on our data.

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r wavelet_comparison/requirements.txt
```

Set `WAVELET_DATA_DIR` to the folder that holds `data/data/splits/`, the chunk
audio and the `results_eval_16866867` / `results_eval_16866883` folders (none
of these are in the repo), then from `wavelet_comparison/`:

```bash
python build_inputs.py
prosody_labeller inputs/valid -c <toolkit>/wavelet_prosody_toolkit/configs/libritts_boundary.yaml -o out/valid -j 4
prosody_labeller inputs/test -c <toolkit>/wavelet_prosody_toolkit/configs/libritts_boundary.yaml -o out/test -j 4
python compare.py
python threshold_sweep.py
python make_figures.py
```

`inputs/` is generated and not committed. `out/` is committed, so
`compare.py` onwards can be rerun without installing the toolkit.
