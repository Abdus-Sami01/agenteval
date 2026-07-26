from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from agenteval.exceptions import PairedLengthError


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    level: float = 0.95
    method: str = ""

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def margin(self) -> float:
        return self.width / 2

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    test: str
    significant: bool
    detail: str = ""

    def __str__(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return f"{self.test}: stat={self.statistic:.4f}, p={self.p_value:.4g} ({verdict})"


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> Interval:
    if trials == 0:
        return Interval(0.0, 0.0, 1.0, level, "wilson")

    z = _z_for(level)
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom

    return Interval(
        point=p,
        low=max(0.0, center - spread),
        high=min(1.0, center + spread),
        level=level,
        method="wilson",
    )


def bootstrap_interval(
    values: Sequence[float],
    level: float = 0.95,
    iterations: int = 10_000,
    seed: int = 0,
) -> Interval:
    if not values:
        return Interval(0.0, 0.0, 0.0, level, "bootstrap")
    if len(values) == 1:
        v = float(values[0])
        return Interval(v, v, v, level, "bootstrap")

    means = _bootstrap_means(values, iterations, seed)

    alpha = (1 - level) / 2
    low = means[max(0, int(alpha * iterations))]
    high = means[min(iterations - 1, int((1 - alpha) * iterations))]

    return Interval(point=mean(values), low=low, high=high, level=level, method="bootstrap")


def _bootstrap_means(values: Sequence[float], iterations: int, seed: int) -> list[float]:
    """Sorted bootstrap means.

    Resamples with random.choices, whose inner loop runs in C, rather than
    drawing indices one at a time in Python.
    """
    rng = random.Random(seed)
    n = len(values)
    choices = rng.choices
    pool = list(values)

    means = [sum(choices(pool, k=n)) / n for _ in range(iterations)]
    means.sort()
    return means


def paired_bootstrap_diff(
    baseline: Sequence[float],
    candidate: Sequence[float],
    level: float = 0.95,
    iterations: int = 10_000,
    seed: int = 0,
) -> Interval:
    if len(baseline) != len(candidate):
        raise PairedLengthError(len(baseline), len(candidate), "paired bootstrap")
    if not baseline:
        return Interval(0.0, 0.0, 0.0, level, "paired-bootstrap")

    diffs = [c - b for b, c in zip(baseline, candidate, strict=False)]
    means = _bootstrap_means(diffs, iterations, seed)

    alpha = (1 - level) / 2
    return Interval(
        point=mean(diffs),
        low=means[max(0, int(alpha * iterations))],
        high=means[min(iterations - 1, int((1 - alpha) * iterations))],
        level=level,
        method="paired-bootstrap",
    )


def bca_interval(
    values: Sequence[float],
    level: float = 0.95,
    iterations: int = 10_000,
    seed: int = 0,
) -> Interval:
    n = len(values)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, level, "bca")
    if n < 3:
        return bootstrap_interval(values, level, iterations, seed)

    observed = mean(values)
    replicates = _bootstrap_means(values, iterations, seed)

    below = sum(1 for r in replicates if r < observed)
    if below == 0 or below == iterations:
        return bootstrap_interval(values, level, iterations, seed)
    z0 = _inv_norm_cdf(below / iterations)

    total = sum(values)
    jackknife = [(total - v) / (n - 1) for v in values]
    jack_mean = mean(jackknife)
    deviations = [jack_mean - j for j in jackknife]

    numerator = sum(d ** 3 for d in deviations)
    denominator = 6 * (sum(d ** 2 for d in deviations) ** 1.5)
    acceleration = numerator / denominator if denominator else 0.0

    alpha = (1 - level) / 2
    z_lo, z_hi = _inv_norm_cdf(alpha), _inv_norm_cdf(1 - alpha)

    def adjust(z: float) -> float:
        denom = 1 - acceleration * (z0 + z)
        if denom == 0:
            return 0.5
        return _norm_cdf(z0 + (z0 + z) / denom)

    lo_pct, hi_pct = adjust(z_lo), adjust(z_hi)
    lo_idx = min(iterations - 1, max(0, int(lo_pct * iterations)))
    hi_idx = min(iterations - 1, max(0, int(hi_pct * iterations)))
    if lo_idx > hi_idx:
        lo_idx, hi_idx = hi_idx, lo_idx

    return Interval(
        point=observed,
        low=replicates[lo_idx],
        high=replicates[hi_idx],
        level=level,
        method="bca",
    )


def stratified_rates(
    groups: dict[str, tuple[int, int]],
    level: float = 0.95,
) -> dict[str, Interval]:
    return {name: wilson_interval(passed, total, level) for name, (passed, total) in groups.items()}


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def permutation_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    iterations: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> TestResult:
    if len(baseline) != len(candidate):
        raise PairedLengthError(len(baseline), len(candidate), "permutation test")
    if not baseline:
        return TestResult(0.0, 1.0, "paired-permutation", False, "no samples")

    diffs = [c - b for b, c in zip(baseline, candidate, strict=False)]
    observed = mean(diffs)
    rng = random.Random(seed)

    extreme = 0
    for _ in range(iterations):
        flipped = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        if abs(flipped) >= abs(observed):
            extreme += 1

    p = (extreme + 1) / (iterations + 1)
    return TestResult(
        statistic=observed,
        p_value=p,
        test="paired-permutation",
        significant=p < alpha,
        detail=f"mean difference {observed:+.4f} over {len(diffs)} paired items",
    )


def mcnemar_test(baseline: Sequence[bool], candidate: Sequence[bool], alpha: float = 0.05) -> TestResult:
    if len(baseline) != len(candidate):
        raise PairedLengthError(len(baseline), len(candidate), "mcnemar test")

    only_candidate = sum(1 for b, c in zip(baseline, candidate, strict=False) if c and not b)
    only_baseline = sum(1 for b, c in zip(baseline, candidate, strict=False) if b and not c)
    discordant = only_candidate + only_baseline

    if discordant == 0:
        return TestResult(0.0, 1.0, "mcnemar", False, "no discordant pairs")

    smaller = min(only_candidate, only_baseline)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
    p = min(1.0, 2 * tail)

    return TestResult(
        statistic=float(only_candidate - only_baseline),
        p_value=p,
        test="mcnemar",
        significant=p < alpha,
        detail=f"{only_candidate} fixed, {only_baseline} broken ({discordant} discordant)",
    )


def cohens_d(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    if len(baseline) < 2 or len(candidate) < 2:
        return 0.0
    n1, n2 = len(baseline), len(candidate)
    s1, s2 = stdev(baseline), stdev(candidate)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (mean(candidate) - mean(baseline)) / pooled


def cliffs_delta(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    if not baseline or not candidate:
        return 0.0
    greater = sum(1 for c in candidate for b in baseline if c > b)
    less = sum(1 for c in candidate for b in baseline if c < b)
    return (greater - less) / (len(baseline) * len(candidate))


def interpret_effect(delta: float) -> str:
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def required_sample_size(baseline_rate: float, detectable_delta: float, level: float = 0.95, power: float = 0.8) -> int:
    if detectable_delta <= 0:
        return 0
    z_alpha = _z_for(level)
    z_beta = _z_for(2 * power - 1)
    p1 = min(max(baseline_rate, 1e-6), 1 - 1e-6)
    p2 = min(max(p1 + detectable_delta, 1e-6), 1 - 1e-6)
    p_bar = (p1 + p2) / 2

    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / (detectable_delta ** 2))


def minimum_detectable_effect(n: int, baseline_rate: float = 0.5, level: float = 0.95, power: float = 0.8) -> float:
    if n <= 0:
        return 1.0
    z_alpha = _z_for(level)
    z_beta = _z_for(2 * power - 1)
    p = min(max(baseline_rate, 1e-6), 1 - 1e-6)
    return (z_alpha + z_beta) * math.sqrt(2 * p * (1 - p) / n)


def _z_for(level: float) -> float:
    level = min(max(level, 0.0), 0.999999)
    return _inv_norm_cdf(1 - (1 - level) / 2)


def _inv_norm_cdf(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
