from __future__ import annotations

import re
from typing import Any, Callable

from agenteval.graders.base import Grader
from agenteval.types import Score, Task


class RubricGrader(Grader):
    name = "rubric"

    def __init__(self, criteria: dict[str, Callable[[Any, Task], bool]], weights: dict[str, float] | None = None,
                 threshold: float = 1.0):
        self._criteria = criteria
        self._weights = weights or {}
        self._threshold = threshold

    def grade(self, prediction: Any, task: Task) -> Score:
        if not self._criteria:
            return self._score(0.0, passed=False, detail="no criteria configured")

        earned = 0.0
        possible = 0.0
        subscores: dict[str, float] = {}
        failed: list[str] = []

        for label, check in self._criteria.items():
            weight = self._weights.get(label, 1.0)
            possible += weight
            try:
                ok = bool(check(prediction, task))
            except Exception as e:
                subscores[label] = 0.0
                failed.append(f"{label} (error: {e})")
                continue
            subscores[label] = 1.0 if ok else 0.0
            if ok:
                earned += weight
            else:
                failed.append(label)

        value = earned / possible if possible else 0.0
        return self._score(
            value,
            passed=value >= self._threshold,
            detail="" if not failed else f"failed: {', '.join(failed[:5])}",
            **subscores,
        )


class WeightedGrader(Grader):
    name = "weighted"

    def __init__(self, graders: dict[str, Grader], weights: dict[str, float] | None = None,
                 threshold: float = 1.0):
        self._graders = graders
        self._weights = weights or {}
        self._threshold = threshold

    def grade(self, prediction: Any, task: Task) -> Score:
        if not self._graders:
            return self._score(0.0, passed=False, detail="no graders configured")

        total_weight = 0.0
        weighted_sum = 0.0
        subscores: dict[str, float] = {}
        details: list[str] = []

        for label, grader in self._graders.items():
            weight = self._weights.get(label, 1.0)
            sub = grader.grade(prediction, task)
            subscores[label] = sub.value
            weighted_sum += sub.value * weight
            total_weight += weight
            if sub.detail:
                details.append(f"{label}: {sub.detail}")

        value = weighted_sum / total_weight if total_weight else 0.0
        return self._score(
            value,
            passed=value >= self._threshold,
            detail="; ".join(details[:3]),
            **subscores,
        )


SCORE_PATTERNS = [
    re.compile(r"\bscore\s*[:=]\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"\brating\s*[:=]\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"),
    re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$"),
]

VERDICT_YES = re.compile(r"\b(yes|correct|pass|true|acceptable)\b", re.I)
VERDICT_NO = re.compile(r"\b(no|incorrect|fail|false|unacceptable)\b", re.I)


class LLMJudgeGrader(Grader):
    name = "llm_judge"

    def __init__(
        self,
        judge_fn: Callable[[str], str],
        prompt_template: str = "",
        scale: float = 1.0,
        threshold: float = 0.5,
        retries: int = 1,
    ):
        self._judge = judge_fn
        self._template = prompt_template or DEFAULT_JUDGE_PROMPT
        self._scale = scale
        self._threshold = threshold
        self._retries = max(1, retries)

    def grade(self, prediction: Any, task: Task) -> Score:
        prompt = (
            self._template
            .replace("{{input}}", str(task.input))
            .replace("{{expected}}", str(task.expected))
            .replace("{{prediction}}", str(prediction))
        )

        last_raw = ""
        for _ in range(self._retries):
            try:
                last_raw = str(self._judge(prompt))
            except Exception as e:
                return self._score(0.0, passed=False, detail=f"judge call failed: {e}")

            parsed = parse_judge_score(last_raw, self._scale)
            if parsed is not None:
                return self._score(
                    parsed,
                    passed=parsed >= self._threshold,
                    detail=last_raw.strip()[:160],
                )

        return self._score(0.0, passed=False,
                           detail=f"could not parse a score from judge output: {last_raw.strip()[:120]!r}")


def parse_judge_score(raw: str, scale: float = 1.0) -> float | None:
    text = raw.strip()

    for pattern in SCORE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if match.re.groups == 2:
            numerator, denominator = float(match.group(1)), float(match.group(2))
            return numerator / denominator if denominator else None
        value = float(match.group(1))
        return value / scale if scale else value

    has_yes = VERDICT_YES.search(text) is not None
    has_no = VERDICT_NO.search(text) is not None
    if has_yes and not has_no:
        return 1.0
    if has_no and not has_yes:
        return 0.0
    return None


DEFAULT_JUDGE_PROMPT = """You are grading a model's answer.

Question:
{{input}}

Reference answer:
{{expected}}

Model answer:
{{prediction}}

Is the model answer correct? Reply with "Score: 1" if correct or "Score: 0" if incorrect, then one sentence of justification."""
