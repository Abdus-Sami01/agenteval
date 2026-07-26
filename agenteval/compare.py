from __future__ import annotations

from dataclasses import dataclass, field

from agenteval.stats import (
    Interval,
    TestResult,
    cliffs_delta,
    interpret_effect,
    mcnemar_test,
    paired_bootstrap_diff,
    permutation_test,
    wilson_interval,
)
from agenteval.types import EvalRun, GateReport, GateResult, Outcome


@dataclass
class Comparison:
    baseline_name: str
    candidate_name: str
    paired_ids: list[str] = field(default_factory=list)
    baseline_rate: Interval | None = None
    candidate_rate: Interval | None = None
    score_delta: Interval | None = None
    significance: TestResult | None = None
    mcnemar: TestResult | None = None
    effect_size: float = 0.0
    fixed: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.score_delta.point if self.score_delta else 0.0

    @property
    def is_improvement(self) -> bool:
        if not self.score_delta:
            return False
        return self.score_delta.low > 0

    @property
    def is_regression(self) -> bool:
        if not self.score_delta:
            return False
        return self.score_delta.high < 0

    @property
    def inconclusive(self) -> bool:
        return not self.is_improvement and not self.is_regression

    def verdict(self) -> str:
        if self.is_improvement:
            return "IMPROVEMENT"
        if self.is_regression:
            return "REGRESSION"
        return "INCONCLUSIVE"

    def summary(self) -> str:
        lines = [
            f"{self.baseline_name} -> {self.candidate_name}  ({len(self.paired_ids)} paired tasks)",
            "",
        ]
        if self.baseline_rate and self.candidate_rate:
            lines.append(f"  pass rate  baseline : {self.baseline_rate}")
            lines.append(f"  pass rate  candidate: {self.candidate_rate}")
        if self.score_delta:
            lines.append(f"  score delta         : {self.score_delta}")
        if self.significance:
            lines.append(f"  {self.significance}")
        if self.mcnemar:
            lines.append(f"  {self.mcnemar}")
        lines.append(f"  effect size (cliff) : {self.effect_size:+.4f} ({interpret_effect(self.effect_size)})")
        lines.append("")
        if self.fixed:
            lines.append(f"  fixed  ({len(self.fixed)}): {', '.join(self.fixed[:8])}")
        if self.broken:
            lines.append(f"  broken ({len(self.broken)}): {', '.join(self.broken[:8])}")
        lines.append("")
        lines.append(f"  VERDICT: {self.verdict()}")
        if self.inconclusive and self.score_delta:
            lines.append("  (confidence interval spans zero - not enough evidence to call it either way)")
        return "\n".join(lines)


def compare(
    baseline: EvalRun,
    candidate: EvalRun,
    level: float = 0.95,
    iterations: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Comparison:
    base_by_id = {r.task_id: r for r in baseline.results}
    cand_by_id = {r.task_id: r for r in candidate.results}
    shared = [tid for tid in base_by_id if tid in cand_by_id]

    comparison = Comparison(
        baseline_name=baseline.metadata.system_name or "baseline",
        candidate_name=candidate.metadata.system_name or "candidate",
        paired_ids=shared,
    )

    if not shared:
        return comparison

    base_scores = [base_by_id[t].numeric for t in shared]
    cand_scores = [cand_by_id[t].numeric for t in shared]
    base_pass = [base_by_id[t].is_pass for t in shared]
    cand_pass = [cand_by_id[t].is_pass for t in shared]

    comparison.baseline_rate = wilson_interval(sum(base_pass), len(base_pass), level)
    comparison.candidate_rate = wilson_interval(sum(cand_pass), len(cand_pass), level)
    comparison.score_delta = paired_bootstrap_diff(base_scores, cand_scores, level, iterations, seed)
    comparison.significance = permutation_test(base_scores, cand_scores, iterations, seed, alpha)
    comparison.mcnemar = mcnemar_test(base_pass, cand_pass, alpha)
    comparison.effect_size = cliffs_delta(base_scores, cand_scores)

    comparison.fixed = [t for t in shared if not base_by_id[t].is_pass and cand_by_id[t].is_pass]
    comparison.broken = [t for t in shared if base_by_id[t].is_pass and not cand_by_id[t].is_pass]

    return comparison


def gate(
    run: EvalRun,
    min_pass_rate: float | None = None,
    min_mean_score: float | None = None,
    max_error_rate: float | None = None,
    max_mean_latency_ms: float | None = None,
    min_tag_pass_rate: dict[str, float] | None = None,
    require_ci_above: float | None = None,
    level: float = 0.95,
) -> GateReport:
    report = GateReport()

    if min_pass_rate is not None:
        observed = run.pass_rate
        report.gates.append(GateResult(
            "min_pass_rate", observed >= min_pass_rate, observed, min_pass_rate,
            f"{run.passed}/{run.passed + run.failed} graded tasks passed",
        ))

    if min_mean_score is not None:
        observed = run.mean_score
        report.gates.append(GateResult(
            "min_mean_score", observed >= min_mean_score, observed, min_mean_score,
        ))

    if max_error_rate is not None:
        observed = run.errored / len(run) if len(run) else 0.0
        report.gates.append(GateResult(
            "max_error_rate", observed <= max_error_rate, observed, max_error_rate,
            f"{run.errored} errored or timed out",
        ))

    if max_mean_latency_ms is not None:
        latencies = [r.elapsed_ms for r in run.results]
        observed = sum(latencies) / len(latencies) if latencies else 0.0
        report.gates.append(GateResult(
            "max_mean_latency_ms", observed <= max_mean_latency_ms, observed, max_mean_latency_ms,
        ))

    if require_ci_above is not None:
        ci = wilson_interval(run.passed, run.passed + run.failed, level)
        report.gates.append(GateResult(
            "ci_lower_bound", ci.low >= require_ci_above, ci.low, require_ci_above,
            f"{int(level * 100)}% CI {ci}",
        ))

    for tag, threshold in (min_tag_pass_rate or {}).items():
        tagged = run.by_tag().get(tag, [])
        graded = [r for r in tagged if r.outcome in (Outcome.PASS, Outcome.FAIL)]
        observed = sum(1 for r in graded if r.is_pass) / len(graded) if graded else 0.0
        report.gates.append(GateResult(
            f"tag:{tag}", observed >= threshold, observed, threshold,
            f"{len(graded)} tasks tagged {tag!r}",
        ))

    return report


def regression_gate(
    comparison: Comparison,
    max_broken: int = 0,
    max_score_drop: float = 0.0,
    block_on_significant_regression: bool = True,
) -> GateReport:
    report = GateReport()

    report.gates.append(GateResult(
        "max_broken_tasks",
        len(comparison.broken) <= max_broken,
        float(len(comparison.broken)),
        float(max_broken),
        f"broken: {', '.join(comparison.broken[:5])}" if comparison.broken else "none broken",
    ))

    drop = -comparison.delta if comparison.delta < 0 else 0.0
    report.gates.append(GateResult(
        "max_score_drop", drop <= max_score_drop, drop, max_score_drop,
        f"mean score moved {comparison.delta:+.4f}",
    ))

    if block_on_significant_regression:
        regressed = comparison.is_regression
        report.gates.append(GateResult(
            "no_significant_regression", not regressed, 1.0 if regressed else 0.0, 0.0,
            f"CI {comparison.score_delta}" if comparison.score_delta else "no interval",
        ))

    return report
