from __future__ import annotations

import re
import string
import unicodedata
from typing import Any

from agenteval.graders.base import Grader
from agenteval.types import Score, Task

ARTICLES = {"a", "an", "the"}


def normalize_text(
    text: Any,
    lowercase: bool = True,
    strip_punctuation: bool = True,
    strip_articles: bool = True,
    collapse_space: bool = True,
) -> str:
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    if lowercase:
        s = s.lower()
    if strip_punctuation:
        s = s.translate(str.maketrans("", "", string.punctuation))
    if strip_articles:
        s = " ".join(w for w in s.split() if w not in ARTICLES)
    if collapse_space:
        s = " ".join(s.split())
    return s.strip()


class ExactMatchGrader(Grader):
    name = "exact"

    def __init__(self, normalize: bool = True, case_sensitive: bool = False):
        self._normalize = normalize
        self._case_sensitive = case_sensitive

    def grade(self, prediction: Any, task: Task) -> Score:
        if self._normalize:
            pred = normalize_text(prediction, lowercase=not self._case_sensitive)
            gold = normalize_text(task.expected, lowercase=not self._case_sensitive)
        else:
            pred, gold = str(prediction), str(task.expected)
            if not self._case_sensitive:
                pred, gold = pred.lower(), gold.lower()

        match = pred == gold
        return self._score(1.0 if match else 0.0, detail="" if match else f"expected {gold!r}, got {pred!r}")


class ContainsGrader(Grader):
    name = "contains"

    def __init__(self, normalize: bool = True, require_all: bool = True):
        self._normalize = normalize
        self._require_all = require_all

    def grade(self, prediction: Any, task: Task) -> Score:
        expected = task.expected
        needles = expected if isinstance(expected, list | tuple | set) else [expected]

        hay = normalize_text(prediction) if self._normalize else str(prediction)
        found = [n for n in needles if (normalize_text(n) if self._normalize else str(n)) in hay]

        ratio = len(found) / len(needles) if needles else 0.0
        passed = ratio == 1.0 if self._require_all else ratio > 0.0
        missing = [n for n in needles if n not in found]
        return self._score(
            ratio,
            passed=passed,
            detail="" if passed else f"missing {missing[:5]}",
        )


class RegexGrader(Grader):
    name = "regex"

    def __init__(self, pattern: str = "", flags: int = 0, use_expected: bool = False):
        self._pattern = pattern
        self._flags = flags
        self._use_expected = use_expected

    def grade(self, prediction: Any, task: Task) -> Score:
        pattern = str(task.expected) if self._use_expected else self._pattern
        if not pattern:
            return self._score(0.0, detail="no pattern configured")
        try:
            hit = re.search(pattern, str(prediction), self._flags) is not None
        except re.error as e:
            return self._score(0.0, detail=f"invalid regex: {e}")
        return self._score(1.0 if hit else 0.0, detail="" if hit else f"no match for {pattern!r}")


class EditDistanceGrader(Grader):
    name = "edit_distance"

    def __init__(self, threshold: float = 0.8, normalize: bool = True):
        self._threshold = threshold
        self._normalize = normalize

    def grade(self, prediction: Any, task: Task) -> Score:
        pred = normalize_text(prediction) if self._normalize else str(prediction)
        gold = normalize_text(task.expected) if self._normalize else str(task.expected)

        if not pred and not gold:
            return self._score(1.0)

        distance = _levenshtein(pred, gold)
        longest = max(len(pred), len(gold)) or 1
        similarity = 1.0 - (distance / longest)
        return self._score(
            similarity,
            passed=similarity >= self._threshold,
            detail=f"similarity {similarity:.3f} vs threshold {self._threshold:.2f}",
            distance=float(distance),
        )


class F1TokenGrader(Grader):
    name = "f1"

    def __init__(self, threshold: float = 1.0, normalize: bool = True):
        self._threshold = threshold
        self._normalize = normalize

    def grade(self, prediction: Any, task: Task) -> Score:
        pred_tokens = (normalize_text(prediction) if self._normalize else str(prediction)).split()
        gold_tokens = (normalize_text(task.expected) if self._normalize else str(task.expected)).split()

        if not pred_tokens and not gold_tokens:
            return self._score(1.0, precision=1.0, recall=1.0)
        if not pred_tokens or not gold_tokens:
            return self._score(0.0, passed=False, detail="one side empty", precision=0.0, recall=0.0)

        common: dict[str, int] = {}
        gold_counts: dict[str, int] = {}
        for t in gold_tokens:
            gold_counts[t] = gold_counts.get(t, 0) + 1
        for t in pred_tokens:
            if gold_counts.get(t, 0) > 0:
                gold_counts[t] -= 1
                common[t] = common.get(t, 0) + 1

        overlap = sum(common.values())
        if overlap == 0:
            return self._score(0.0, passed=False, detail="no token overlap", precision=0.0, recall=0.0)

        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return self._score(
            f1,
            passed=f1 >= self._threshold,
            detail=f"f1 {f1:.3f} (p={precision:.3f} r={recall:.3f})",
            precision=precision,
            recall=recall,
        )


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]
