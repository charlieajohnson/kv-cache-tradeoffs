from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    elapsed_ms: float = 0.0
    _start: float | None = None

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end = time.perf_counter()
        if self._start is not None:
            self.elapsed_ms += (end - self._start) * 1000


@contextmanager
def timed() -> Generator[Timer, None, None]:
    t = Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed_ms = (time.perf_counter() - start) * 1000
