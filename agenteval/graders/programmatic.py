from __future__ import annotations

from typing import Any, Callable

from agenteval.graders.base import Grader
from agenteval.types import Score, Task


class PredicateGrader(Grader):
    name = "predicate"

    def __init__(self, predicate: Callable[[Any, Task], bool], label: str = ""):
        self._predicate = predicate
        self._label = label or "predicate"

    def grade(self, prediction: Any, task: Task) -> Score:
        try:
            ok = bool(self._predicate(prediction, task))
        except Exception as e:
            return self._score(0.0, passed=False, detail=f"{self._label} raised {type(e).__name__}: {e}")
        return self._score(1.0 if ok else 0.0, passed=ok,
                           detail="" if ok else f"{self._label} returned False")


class CallableGrader(Grader):
    name = "callable"

    def __init__(self, fn: Callable[[Any, Task], Any], threshold: float = 1.0, label: str = ""):
        self._fn = fn
        self._threshold = threshold
        self._label = label or "callable"

    def grade(self, prediction: Any, task: Task) -> Score:
        try:
            raw = self._fn(prediction, task)
        except Exception as e:
            return self._score(0.0, passed=False, detail=f"{self._label} raised {type(e).__name__}: {e}")

        if isinstance(raw, Score):
            return raw

        if isinstance(raw, tuple) and len(raw) == 2:
            value, detail = raw
            value = float(value)
            return self._score(value, passed=value >= self._threshold, detail=str(detail))

        if isinstance(raw, bool):
            return self._score(1.0 if raw else 0.0, passed=raw)

        try:
            value = float(raw)
        except (TypeError, ValueError):
            return self._score(0.0, passed=False, detail=f"{self._label} returned non-numeric {raw!r}")

        return self._score(value, passed=value >= self._threshold)
