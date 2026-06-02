"""Process-wide concurrency limiter for local Ollama LLM calls.

Local Ollama instances do not parallelise like cloud APIs.  Flooding them with
concurrent requests causes context-switching overhead, VRAM thrashing, and
out-of-memory crashes.  Every ``ainvoke`` call in this codebase goes through
:func:`get_ollama_semaphore` so the process never exceeds
``AppSettings.ollama_concurrency`` simultaneous in-flight requests.

Tuning guide
------------
* ``OLLAMA_CONCURRENCY=1`` (default) – strict serial execution; safest for
  consumer hardware with a single small model loaded.
* ``OLLAMA_CONCURRENCY=2`` – allows mild pipelining if Ollama is running on a
  machine with ample VRAM (e.g. 24 GB+) or when offloading to a remote host.

The semaphore is created lazily on first access so it is always bound to the
event loop that is actually running at call time.
"""

from __future__ import annotations

import asyncio
import logging


logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def get_ollama_semaphore() -> asyncio.Semaphore:
    """Return the process-wide semaphore throttling concurrent Ollama calls.

    Thread-safe under a single event-loop (the normal asyncio model).  The
    concurrency limit is read once from :func:`factcheck.config.get_settings`
    and cached for the lifetime of the process.
    """
    global _semaphore
    if _semaphore is None:
        # Local import avoids circular dependency at module load time.
        from factcheck.config import get_settings  # noqa: PLC0415

        limit = get_settings().ollama_concurrency
        logger.info(
            "Ollama concurrency semaphore initialised with limit=%d "
            "(set OLLAMA_CONCURRENCY in .env to change)",
            limit,
        )
        _semaphore = asyncio.Semaphore(limit)
    return _semaphore


def reset_ollama_semaphore_for_tests() -> None:
    """Reset the cached semaphore so tests can vary ``OLLAMA_CONCURRENCY``."""

    global _semaphore
    _semaphore = None
