"""The single connection point to the BIU lab server.

TODO: replace run_on_server() with the real flow: SSH/SFTP to the BIU
cluster using the lab account, submit a job (same convention as
train_whisper2.slurm / evaluate_test_model2.slurm at the repo root),
wait for it to finish, and pull back the per-word predictions.

Until that's wired up, this returns deterministic fake predictions so the
rest of the app can be built and tested end-to-end without a live server.
Every other module in app/ should keep working unchanged once this
function is replaced - it's the only thing that needs to know about the
server.
"""

from typing import Optional

import pandas as pd

CHUNK_SIZE_WORDS = 20  # placeholder chunking, until the real chunker module is wired in here
FAKE_THRESHOLD = 0.5


def run_on_server(transcript_df: pd.DataFrame, credentials: Optional[dict] = None) -> pd.DataFrame:
    """Given the validated transcript, return per-word boundary predictions.

    Real implementation will use `credentials` (BIU lab email/password) to
    open an SSH/SFTP connection, send audio + transcript to the BIU cluster,
    and run the trained model there - never write the password to disk or logs.
    For now, `credentials` is unused and this returns fake but schema-shaped
    output so app.py has something real to display.
    """
    n = len(transcript_df)
    rng = pd.Series(range(n))

    boundary_probability = ((rng * 37 + 11) % 100) / 100.0
    chunk_id = (rng // CHUNK_SIZE_WORDS).map(lambda i: f"chunk_{i}")

    return pd.DataFrame(
        {
            "boundary_probability": boundary_probability,
            "boundary_prediction": (boundary_probability >= FAKE_THRESHOLD).astype(int),
            "threshold": FAKE_THRESHOLD,
            "chunk_id": chunk_id,
        }
    )
