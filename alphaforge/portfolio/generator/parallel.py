"""
parallel.py — Interrupt-safe ProcessPoolExecutor context manager.

On Windows, Ctrl+C sends SIGINT to every process in the console group —
including worker processes — causing them to print ugly tracebacks and
then hang while the main process waits for them to finish.

Two fixes applied here:
  1. Workers ignore SIGINT (SIG_IGN) — only the main process handles it.
  2. On KeyboardInterrupt in the main process, worker processes are
     forcefully terminated before shutdown, so the pool exits immediately.
"""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor


def resolve_workers(n_workers: int) -> int:
    """Resolve -1 to cpu_count(), clamp to at least 1."""
    return os.cpu_count() or 1 if n_workers == -1 else max(1, n_workers)


# ── Worker initializer wrapper ────────────────────────────────────────────────

def _worker_initializer(user_init, user_args: tuple) -> None:
    """
    Called once per worker process on startup.
    Disables SIGINT in the worker so Ctrl+C only interrupts the main process,
    then calls the user's own initializer (e.g. _init_filter_workers).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if user_init is not None:
        user_init(*user_args)


# ── Context manager ───────────────────────────────────────────────────────────

@contextmanager
def pool_executor(n_workers: int, initializer=None, initargs: tuple = ()):
    """
    Interrupt-safe ProcessPoolExecutor context manager.

    - Workers ignore SIGINT so only the main process handles Ctrl+C.
    - On KeyboardInterrupt, workers are forcefully terminated and the pool
      shuts down without waiting — terminal returns immediately.
    - Prints a clean cancellation message instead of a traceback.
    """
    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_initializer,
        initargs=(initializer, initargs),
    )
    try:
        yield pool
    except KeyboardInterrupt:
        # Forcefully kill all worker processes — no waiting
        if hasattr(pool, "_processes"):
            for p in pool._processes.values():
                p.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        print("\n  Cancelled by user.\n")
        raise SystemExit(0)
    except Exception:
        if hasattr(pool, "_processes"):
            for p in pool._processes.values():
                p.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
