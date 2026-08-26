"""Per-stage status events for one pipeline run.

Tracks each stage's start/completion/failure (with elapsed time and any
metrics the stage records) and reports them through a callback - e.g. to
drive a live status display in the UI while a run is in progress.

An earlier version of this also wrote each job's events to a file for
later debugging, but that file lived in the same short-lived temp
directory as everything else in a run and was never actually reachable
after the fact - and worse, the open file handle blocked that temp
directory's own cleanup on Windows (PermissionError: WinError 32) right
as a run finished. Removed rather than fixed, since the live callback
already covers the real need.
"""

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

StageEventCallback = Callable[[str, str, Dict[str, Any]], None]
"""Called as (stage, event, fields) on "start"/"completed"/"failed" - and,
for biu_sync specifically, "polling" while waiting on SLURM (see
pipeline.biu_sync.run_biu_job's on_status)."""


@contextmanager
def log_stage(stage: str, on_event: Optional[StageEventCallback] = None) -> Iterator[Dict[str, Any]]:
    """Track one pipeline stage's start, then its completion or failure, with elapsed time.

    Yields a dict the caller can fill in with stage-specific metrics
    (e.g. metrics["num_chunks"] = len(chunks)) - whatever's in it by the
    end is passed to on_event alongside the completed/failed event.

    Input: stage - the stage's name; on_event - optional callback, invoked
        on start/completed/failed (see StageEventCallback) - e.g. to drive
        a live status display.
    Output: none. Re-raises whatever the wrapped code raised, after reporting it as a failure.
    """
    def _notify(event: str, **fields: Any) -> None:
        if on_event:
            on_event(stage, event, fields)

    start = time.monotonic()  # not time.time() - immune to the system clock changing mid-stage (NTP sync, DST)
    _notify("start")
    metrics: Dict[str, Any] = {}
    try:
        yield metrics
    except Exception as exc:
        _notify("failed", elapsed_s=round(time.monotonic() - start, 3), error=str(exc), **metrics)
        raise
    else:
        _notify("completed", elapsed_s=round(time.monotonic() - start, 3), **metrics)
