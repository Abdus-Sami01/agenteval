from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agenteval.stats import Interval, wilson_interval
from agenteval.types import Task


@dataclass(frozen=True)
class AgreementReport:
    n: int
    raw_agreement: float
    kappa: float
    expected_agreement: float
    positive_agreement: float = 0.0
    negative_agreement: float = 0.0
    agreement_ci: Interval | None = None

    @property
    def interpretation(self) -> str:
        k = self.kappa
        if k < 0.0:
            return "worse than chance"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"

    @property
    def trustworthy(self) -> bool:
        return self.kappa >= 0.60

    def summary(self) -> str:
        lines = [
            f"  items compared      {self.n}",
            f"  raw agreement       {self.raw_agreement:.1%}"
            + (f"  (95% CI {self.agreement_ci.low:.1%} - {self.agreement_ci.high:.1%})" if self.agreement_ci else ""),
            f"  chance agreement    {self.expected_agreement:.1%}",
            f"  Cohen's kappa       {self.kappa:.4f}  ({self.interpretation})",
        ]
        if self.positive_agreement or self.negative_agreement:
            lines.append(f"  positive agreement  {self.positive_agreement:.1%}")
            lines.append(f"  negative agreement  {self.negative_agreement:.1%}")
        lines.append("")
        if self.trustworthy:
            lines.append("  Judge agrees with reference labels well enough to rely on.")
        else:
            lines.append("  WARNING: agreement is weak - scores from this judge should not be")
            lines.append("  treated as ground truth without further validation.")
        return "\n".join(lines)


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> AgreementReport:
    if len(a) != len(b):
        raise ValueError(f"rater sequences must be equal length, got {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        return AgreementReport(0, 0.0, 0.0, 0.0)

    both_yes = sum(1 for x, y in zip(a, b) if x and y)
    both_no = sum(1 for x, y in zip(a, b) if not x and not y)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if not x and y)

    observed = (both_yes + both_no) / n
    p_a_yes = (both_yes + a_only) / n
    p_b_yes = (both_yes + b_only) / n
    expected = p_a_yes * p_b_yes + (1 - p_a_yes) * (1 - p_b_yes)

    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)

    pos_denom = 2 * both_yes + a_only + b_only
    neg_denom = 2 * both_no + a_only + b_only

    return AgreementReport(
        n=n,
        raw_agreement=observed,
        kappa=kappa,
        expected_agreement=expected,
        positive_agreement=(2 * both_yes / pos_denom) if pos_denom else 0.0,
        negative_agreement=(2 * both_no / neg_denom) if neg_denom else 0.0,
        agreement_ci=wilson_interval(both_yes + both_no, n),
    )


def percent_agreement(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def krippendorff_alpha(ratings: Sequence[Sequence[Any]]) -> float:
    columns = [[r for r in col if r is not None] for col in zip(*ratings)] if ratings else []
    usable = [c for c in columns if len(c) >= 2]
    if not usable:
        return 0.0

    observed_disagreement = 0.0
    pairs = 0
    values: list[Any] = []

    for col in usable:
        values.extend(col)
        m = len(col)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                observed_disagreement += 0.0 if col[i] == col[j] else 1.0
                pairs += 1

    if pairs == 0:
        return 1.0
    observed = observed_disagreement / pairs

    total = len(values)
    expected_disagreement = 0.0
    for i in range(total):
        for j in range(total):
            if i == j:
                continue
            expected_disagreement += 0.0 if values[i] == values[j] else 1.0
    expected = expected_disagreement / (total * (total - 1)) if total > 1 else 0.0

    if expected == 0:
        return 1.0
    return 1.0 - observed / expected


@dataclass
class JudgeValidation:
    report: AgreementReport
    disagreements: list[tuple[str, bool, bool]] = field(default_factory=list)
    judge_bias: float = 0.0

    @property
    def judge_is_lenient(self) -> bool:
        return self.judge_bias > 0.05

    @property
    def judge_is_strict(self) -> bool:
        return self.judge_bias < -0.05

    def summary(self) -> str:
        lines = ["Judge validation against reference labels", ""]
        lines.append(self.report.summary())
        lines.append("")

        if self.judge_is_lenient:
            lines.append(f"  BIAS: judge passes {self.judge_bias:+.1%} more often than the reference (lenient)")
        elif self.judge_is_strict:
            lines.append(f"  BIAS: judge passes {self.judge_bias:+.1%} vs reference (strict)")
        else:
            lines.append(f"  bias: {self.judge_bias:+.1%} - judge and reference pass at similar rates")

        if self.disagreements:
            lines.append("")
            lines.append(f"  disagreements ({len(self.disagreements)}), first few:")
            for task_id, judge, reference in self.disagreements[:6]:
                direction = "judge passed, reference failed" if judge else "judge failed, reference passed"
                lines.append(f"    {task_id}: {direction}")
        return "\n".join(lines)


def validate_judge(
    judge_fn: Callable[[Any, Task], bool],
    labeled: Sequence[tuple[Task, Any, bool]],
) -> JudgeValidation:
    judge_calls: list[bool] = []
    reference: list[bool] = []
    disagreements: list[tuple[str, bool, bool]] = []

    for task, prediction, human_label in labeled:
        try:
            verdict = bool(judge_fn(prediction, task))
        except Exception:
            verdict = False
        judge_calls.append(verdict)
        reference.append(bool(human_label))
        if verdict != bool(human_label):
            disagreements.append((task.id, verdict, bool(human_label)))

    report = cohens_kappa(reference, judge_calls)
    n = len(judge_calls) or 1
    bias = (sum(judge_calls) - sum(reference)) / n

    return JudgeValidation(report=report, disagreements=disagreements, judge_bias=bias)
