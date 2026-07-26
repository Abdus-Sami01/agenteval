from __future__ import annotations

import re
from typing import Any

from agenteval.graders.base import Grader
from agenteval.types import Score, Task

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    match = NUMBER_RE.search(str(value).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


class NumericGrader(Grader):
    name = "numeric"

    def __init__(self, tolerance: float = 0.0, relative: bool = False, extract: bool = True):
        self._tolerance = tolerance
        self._relative = relative
        self._extract = extract

    def grade(self, prediction: Any, task: Task) -> Score:
        pred = extract_number(prediction) if self._extract else _coerce(prediction)
        gold = extract_number(task.expected) if self._extract else _coerce(task.expected)

        if pred is None:
            return self._score(0.0, passed=False, detail=f"no number found in {str(prediction)[:60]!r}")
        if gold is None:
            return self._score(0.0, passed=False, detail="expected value is not numeric")

        delta = abs(pred - gold)
        limit = abs(gold) * self._tolerance if self._relative else self._tolerance
        ok = delta <= limit

        return self._score(
            1.0 if ok else 0.0,
            passed=ok,
            detail="" if ok else f"got {pred}, expected {gold} (delta {delta:.6g} > {limit:.6g})",
            delta=delta,
        )


class RangeGrader(Grader):
    name = "range"

    def __init__(self, low: float | None = None, high: float | None = None, inclusive: bool = True):
        self._low = low
        self._high = high
        self._inclusive = inclusive

    def grade(self, prediction: Any, task: Task) -> Score:
        pred = extract_number(prediction)
        if pred is None:
            return self._score(0.0, passed=False, detail="no number found")

        low, high = self._low, self._high
        if low is None and high is None and isinstance(task.expected, list | tuple) and len(task.expected) == 2:
            low, high = float(task.expected[0]), float(task.expected[1])

        if self._inclusive:
            ok = (low is None or pred >= low) and (high is None or pred <= high)
        else:
            ok = (low is None or pred > low) and (high is None or pred < high)

        bounds = f"[{low}, {high}]" if self._inclusive else f"({low}, {high})"
        return self._score(
            1.0 if ok else 0.0,
            passed=ok,
            detail="" if ok else f"{pred} outside {bounds}",
        )


def _coerce(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
