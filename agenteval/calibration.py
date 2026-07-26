from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Bin:
    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return self.accuracy - self.mean_confidence


@dataclass
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    bins: list[Bin] = field(default_factory=list)
    n: int = 0
    mean_confidence: float = 0.0
    accuracy: float = 0.0

    @property
    def overconfident(self) -> bool:
        return self.mean_confidence - self.accuracy > 0.05

    @property
    def underconfident(self) -> bool:
        return self.accuracy - self.mean_confidence > 0.05

    @property
    def well_calibrated(self) -> bool:
        return self.ece < 0.05

    def summary(self) -> str:
        lines = [
            f"  items                {self.n}",
            f"  mean confidence      {self.mean_confidence:.1%}",
            f"  actual accuracy      {self.accuracy:.1%}",
            f"  ECE (expected error) {self.ece:.4f}",
            f"  MCE (worst bin)      {self.mce:.4f}",
            f"  Brier score          {self.brier:.4f}",
            "",
            f"  {'confidence bin':<18}{'n':>6}{'conf':>9}{'acc':>9}{'gap':>9}",
        ]
        for b in self.bins:
            if b.count == 0:
                continue
            lines.append(
                f"  [{b.low:.1f}, {b.high:.1f})".ljust(18)
                + f"{b.count:>6}{b.mean_confidence:>9.1%}{b.accuracy:>9.1%}{b.gap:>+9.1%}"
            )
        lines.append("")
        if self.overconfident:
            lines.append("  OVERCONFIDENT: stated confidence exceeds real accuracy.")
        elif self.underconfident:
            lines.append("  UNDERCONFIDENT: system is more accurate than it claims.")
        elif self.well_calibrated:
            lines.append("  Well calibrated (ECE < 0.05).")
        else:
            lines.append("  Roughly calibrated overall, but check individual bins.")
        return "\n".join(lines)


def calibration(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> CalibrationReport:
    if len(confidences) != len(correct):
        raise ValueError(f"length mismatch: {len(confidences)} confidences, {len(correct)} outcomes")

    n = len(confidences)
    if n == 0:
        return CalibrationReport(0.0, 0.0, 0.0, [], 0, 0.0, 0.0)

    clipped = [min(1.0, max(0.0, float(c))) for c in confidences]
    outcomes = [bool(c) for c in correct]

    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[Bin] = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == n_bins - 1:
            members = [(c, o) for c, o in zip(clipped, outcomes) if low <= c <= high]
        else:
            members = [(c, o) for c, o in zip(clipped, outcomes) if low <= c < high]

        if not members:
            bins.append(Bin(low, high, 0, 0.0, 0.0))
            continue

        count = len(members)
        mean_conf = sum(c for c, _ in members) / count
        accuracy = sum(1 for _, o in members if o) / count
        gap = abs(accuracy - mean_conf)

        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append(Bin(low, high, count, mean_conf, accuracy))

    brier = sum((c - (1.0 if o else 0.0)) ** 2 for c, o in zip(clipped, outcomes)) / n

    return CalibrationReport(
        ece=ece,
        mce=mce,
        brier=brier,
        bins=bins,
        n=n,
        mean_confidence=sum(clipped) / n,
        accuracy=sum(1 for o in outcomes if o) / n,
    )


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not confidences:
        return 0.0
    return sum(
        (min(1.0, max(0.0, c)) - (1.0 if o else 0.0)) ** 2
        for c, o in zip(confidences, correct)
    ) / len(confidences)


def brier_skill_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not correct:
        return 0.0
    base_rate = sum(1 for c in correct if c) / len(correct)
    reference = sum((base_rate - (1.0 if o else 0.0)) ** 2 for o in correct) / len(correct)
    if reference == 0:
        return 0.0
    return 1.0 - brier_score(confidences, correct) / reference


def log_loss(confidences: Sequence[float], correct: Sequence[bool], epsilon: float = 1e-15) -> float:
    if not confidences:
        return 0.0
    total = 0.0
    for c, o in zip(confidences, correct):
        p = min(1 - epsilon, max(epsilon, float(c)))
        total += -math.log(p) if o else -math.log(1 - p)
    return total / len(confidences)


def reliability_diagram_text(report: CalibrationReport, width: int = 40) -> str:
    lines = ["Reliability diagram (| = perfect calibration)", ""]
    for b in report.bins:
        if b.count == 0:
            continue
        conf_pos = int(b.mean_confidence * width)
        acc_pos = int(b.accuracy * width)
        row = [" "] * (width + 1)
        row[conf_pos] = "|"
        row[acc_pos] = "#" if acc_pos != conf_pos else "*"
        lines.append(f"  [{b.low:.1f},{b.high:.1f}) {''.join(row)}  n={b.count}")
    lines.append("")
    lines.append("  | = stated confidence, # = actual accuracy, * = aligned")
    return "\n".join(lines)
