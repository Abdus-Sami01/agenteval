from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from agenteval.compare import Comparison, compare
from agenteval.types import EvalRun


@dataclass(frozen=True)
class AdjustedTest:
    label: str
    raw_p: float
    adjusted_p: float
    significant: bool
    rank: int = 0


@dataclass
class MultipleComparisonReport:
    method: str
    alpha: float
    tests: list[AdjustedTest] = field(default_factory=list)

    @property
    def significant(self) -> list[AdjustedTest]:
        return [t for t in self.tests if t.significant]

    @property
    def any_significant(self) -> bool:
        return any(t.significant for t in self.tests)

    def summary(self) -> str:
        if not self.tests:
            return "No comparisons."

        lines = [
            f"{self.method} correction at alpha={self.alpha} over {len(self.tests)} comparisons",
            "",
            f"  {'comparison':<34}{'raw p':>10}{'adj p':>10}  verdict",
        ]
        for t in sorted(self.tests, key=lambda x: x.raw_p):
            verdict = "significant" if t.significant else "not significant"
            lines.append(f"  {t.label[:33]:<34}{t.raw_p:>10.4g}{t.adjusted_p:>10.4g}  {verdict}")

        naive = sum(1 for t in self.tests if t.raw_p < self.alpha)
        corrected = len(self.significant)
        if naive != corrected:
            lines.append("")
            lines.append(f"  {naive} comparison(s) look significant uncorrected, "
                         f"{corrected} survive correction")
        return "\n".join(lines)


def bonferroni(p_values: Sequence[float], alpha: float = 0.05, labels: Sequence[str] | None = None) -> MultipleComparisonReport:
    n = len(p_values)
    labels = list(labels or [f"test_{i}" for i in range(n)])
    tests = [
        AdjustedTest(labels[i], p, min(1.0, p * n), min(1.0, p * n) < alpha, i)
        for i, p in enumerate(p_values)
    ]
    return MultipleComparisonReport("Bonferroni", alpha, tests)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05, labels: Sequence[str] | None = None) -> MultipleComparisonReport:
    n = len(p_values)
    labels = list(labels or [f"test_{i}" for i in range(n)])
    if n == 0:
        return MultipleComparisonReport("Holm-Bonferroni", alpha, [])

    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 0.0

    for rank, idx in enumerate(order):
        candidate = (n - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)

    tests = [
        AdjustedTest(labels[i], p_values[i], adjusted[i], adjusted[i] < alpha, order.index(i))
        for i in range(n)
    ]
    return MultipleComparisonReport("Holm-Bonferroni", alpha, tests)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05, labels: Sequence[str] | None = None) -> MultipleComparisonReport:
    n = len(p_values)
    labels = list(labels or [f"test_{i}" for i in range(n)])
    if n == 0:
        return MultipleComparisonReport("Benjamini-Hochberg", alpha, [])

    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 1.0

    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        candidate = p_values[idx] * n / (rank + 1)
        running = min(running, candidate)
        adjusted[idx] = min(1.0, running)

    tests = [
        AdjustedTest(labels[i], p_values[i], adjusted[i], adjusted[i] < alpha, order.index(i))
        for i in range(n)
    ]
    return MultipleComparisonReport("Benjamini-Hochberg (FDR)", alpha, tests)


CORRECTIONS = {
    "bonferroni": bonferroni,
    "holm": holm_bonferroni,
    "bh": benjamini_hochberg,
    "fdr": benjamini_hochberg,
}


def adjust(p_values: Sequence[float], method: str = "holm", alpha: float = 0.05,
           labels: Sequence[str] | None = None) -> MultipleComparisonReport:
    fn = CORRECTIONS.get(method.lower())
    if fn is None:
        raise ValueError(f"unknown correction {method!r}. Available: {sorted(CORRECTIONS)}")
    return fn(p_values, alpha, labels)


@dataclass
class SystemMatrix:
    comparisons: dict[tuple[str, str], Comparison] = field(default_factory=dict)
    correction: MultipleComparisonReport | None = None

    def summary(self) -> str:
        if not self.comparisons:
            return "No pairwise comparisons."

        lines = ["Pairwise comparisons", ""]
        for (a, b), c in self.comparisons.items():
            raw = c.significance.p_value if c.significance else 1.0
            lines.append(f"  {a} vs {b}: delta={c.delta:+.4f} raw p={raw:.4g} -> {c.verdict()}")

        if self.correction:
            lines.append("")
            lines.append(self.correction.summary())
        return "\n".join(lines)

    def corrected_verdict(self, a: str, b: str) -> str:
        key = (a, b)
        comparison = self.comparisons.get(key)
        if comparison is None or self.correction is None:
            return "UNKNOWN"

        label = f"{a} vs {b}"
        for t in self.correction.tests:
            if t.label == label:
                if not t.significant:
                    return "INCONCLUSIVE"
                return "IMPROVEMENT" if comparison.delta > 0 else "REGRESSION"
        return "UNKNOWN"


def compare_all(
    runs: dict[str, EvalRun],
    method: str = "holm",
    alpha: float = 0.05,
    iterations: int = 5_000,
    seed: int = 0,
    baseline: str | None = None,
) -> SystemMatrix:
    names = list(runs)
    pairs: list[tuple[str, str]] = []

    if baseline:
        if baseline not in runs:
            raise ValueError(f"baseline {baseline!r} not among runs: {names}")
        pairs = [(baseline, n) for n in names if n != baseline]
    else:
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    matrix = SystemMatrix()
    p_values: list[float] = []
    labels: list[str] = []

    for a, b in pairs:
        c = compare(runs[a], runs[b], iterations=iterations, seed=seed)
        matrix.comparisons[(a, b)] = c
        p_values.append(c.significance.p_value if c.significance else 1.0)
        labels.append(f"{a} vs {b}")

    if p_values:
        matrix.correction = adjust(p_values, method, alpha, labels)

    return matrix
