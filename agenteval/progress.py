from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from typing import Any, TextIO

from agenteval.types import Outcome, TaskResult

logger = logging.getLogger("agenteval")


class ProgressReporter:
    """Live progress for long evaluations.

    Writes to stderr so piping stdout to a file still yields clean JSON.
    Silently disables itself when the stream is not a TTY, so CI logs do
    not fill with carriage returns.
    """

    def __init__(
        self,
        total: int,
        stream: TextIO | None = None,
        enabled: bool | None = None,
        width: int = 28,
        min_interval_s: float = 0.1,
    ):
        self._total = max(0, total)
        self._stream = stream or sys.stderr
        self._width = width
        self._min_interval = min_interval_s
        self._done = 0
        self._passed = 0
        self._failed = 0
        self._errored = 0
        self._start = time.perf_counter()
        self._last_draw = 0.0

        if enabled is None:
            enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._enabled = enabled and self._total > 0

    def update(self, result: TaskResult) -> None:
        self._done += 1
        if result.outcome == Outcome.PASS:
            self._passed += 1
        elif result.outcome == Outcome.FAIL:
            self._failed += 1
        else:
            self._errored += 1

        logger.debug("task %s -> %s (%.1fms)", result.task_id, result.outcome.value, result.elapsed_ms)

        now = time.perf_counter()
        if self._done >= self._total or (now - self._last_draw) >= self._min_interval:
            self._last_draw = now
            self._draw()

    def _draw(self) -> None:
        if not self._enabled:
            return

        fraction = self._done / self._total if self._total else 1.0
        filled = int(fraction * self._width)
        bar = "#" * filled + "-" * (self._width - filled)

        elapsed = time.perf_counter() - self._start
        rate = self._done / elapsed if elapsed > 0 else 0.0
        remaining = (self._total - self._done) / rate if rate > 0 else 0.0

        graded = self._passed + self._failed
        pass_rate = self._passed / graded if graded else 0.0

        line = (
            f"\r  [{bar}] {self._done}/{self._total} "
            f"| pass {pass_rate:.0%} | err {self._errored} "
            f"| {rate:.1f}/s | eta {_duration(remaining)}"
        )
        try:
            self._stream.write(line.ljust(96)[:96])
            self._stream.flush()
        except (OSError, ValueError):
            self._enabled = False

    def finish(self) -> None:
        if not self._enabled:
            return
        elapsed = time.perf_counter() - self._start
        graded = self._passed + self._failed
        pass_rate = self._passed / graded if graded else 0.0
        try:
            self._stream.write(
                f"\r  {self._done}/{self._total} tasks | pass {pass_rate:.1%} "
                f"| {self._errored} errors | {_duration(elapsed)}".ljust(96)[:96] + "\n"
            )
            self._stream.flush()
        except (OSError, ValueError):
            pass

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()

    @property
    def callback(self) -> Callable[[TaskResult], None]:
        return self.update


def _duration(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def configure_logging(level: int = logging.INFO, stream: TextIO | None = None) -> None:
    """Attach a handler to the agenteval logger.

    The library never configures logging on import; applications decide.
    Call this for quick setup, or configure the 'agenteval' logger yourself.
    """
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
