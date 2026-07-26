from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from agenteval.stats import Interval, mean, stdev, wilson_interval, _z_for
from agenteval.types import EvalRun, Outcome


@dataclass(frozen=True)
class TaskStability:
    task_id: str
    runs: int
    passes: int

    @property
    def pass_rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0

    @property
    def is_flaky(self) -> bool:
        return 0 < self.passes < self.runs

    @property
    def entropy(self) -> float:
        p = self.pass_rate
        if p in (0.0, 1.0):
            return 0.0
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass
class StabilityReport:
    runs: int = 0
    tasks: int = 0
    pass_rates: list[float] = field(default_factory=list)
    task_stability: list[TaskStability] = field(default_factory=list)

    @property
    def mean_pass_rate(self) -> float:
        return mean(self.pass_rates)

    @property
    def pass_rate_stdev(self) -> float:
        return stdev(self.pass_rates)

    @property
    def spread(self) -> float:
        return max(self.pass_rates) - min(self.pass_rates) if self.pass_rates else 0.0

    @property
    def flaky(self) -> list[TaskStability]:
        return [t for t in self.task_stability if t.is_flaky]

    @property
    def always_pass(self) -> list[TaskStability]:
        return [t for t in self.task_stability if t.runs and t.passes == t.runs]

    @property
    def always_fail(self) -> list[TaskStability]:
        return [t for t in self.task_stability if t.runs and t.passes == 0]

    @property
    def flake_rate(self) -> float:
        return len(self.flaky) / self.tasks if self.tasks else 0.0

    def combined_interval(self, level: float = 0.95) -> Interval:
        """CI that accounts for run-to-run variance, not just sampling error.

        A single run's Wilson interval assumes the system is deterministic.
        For a nondeterministic system the between-run variance has to be
        added, otherwise the interval is too narrow and differences look
        more certain than they are.
        """
        if not self.pass_rates:
            return Interval(0.0, 0.0, 1.0, level, "combined")

        point = self.mean_pass_rate
        n_runs = len(self.pass_rates)

        within = point * (1 - point) / self.tasks if self.tasks else 0.0
        between = (self.pass_rate_stdev ** 2) / n_runs if n_runs > 1 else 0.0

        se = math.sqrt(within + between)
        z = _z_for(level)
        return Interval(
            point=point,
            low=max(0.0, point - z * se),
            high=min(1.0, point + z * se),
            level=level,
            method="combined-variance",
        )

    def naive_interval(self, level: float = 0.95) -> Interval:
        if not self.pass_rates:
            return Interval(0.0, 0.0, 1.0, level, "wilson")
        passes = round(self.mean_pass_rate * self.tasks)
        return wilson_interval(passes, self.tasks, level)

    @property
    def reliable(self) -> bool:
        return self.spread <= 0.05 and self.flake_rate <= 0.1

    def summary(self, show_flaky: int = 8) -> str:
        combined = self.combined_interval()
        naive = self.naive_interval()

        lines = [
            f"  runs                {self.runs}",
            f"  tasks per run       {self.tasks}",
            f"  pass rates          {', '.join(f'{r:.1%}' for r in self.pass_rates)}",
            f"  mean pass rate      {self.mean_pass_rate:.1%}",
            f"  run-to-run stdev    {self.pass_rate_stdev:.4f}",
            f"  spread (max-min)    {self.spread:.1%}",
            "",
            f"  single-run CI       [{naive.low:.1%}, {naive.high:.1%}]  (assumes determinism)",
            f"  variance-aware CI   [{combined.low:.1%}, {combined.high:.1%}]  (includes run-to-run noise)",
        ]

        if combined.width > naive.width * 1.05:
            inflation = combined.width / naive.width if naive.width else 0
            lines.append(f"  -> nondeterminism widens the interval {inflation:.2f}x; "
                         "reporting a single run would overstate certainty")

        lines += [
            "",
            f"  always pass         {len(self.always_pass)}",
            f"  always fail         {len(self.always_fail)}",
            f"  flaky               {len(self.flaky)} ({self.flake_rate:.1%} of tasks)",
        ]

        if self.flaky:
            lines.append("")
            lines.append("  flakiest tasks:")
            for t in sorted(self.flaky, key=lambda x: -x.entropy)[:show_flaky]:
                lines.append(f"    {t.task_id:<20}{t.passes}/{t.runs} passes  (entropy {t.entropy:.3f})")

        lines.append("")
        if self.reliable:
            lines.append("  Results look stable enough to compare against other systems.")
        else:
            lines.append("  WARNING: results vary noticeably between runs. Increase repeats or")
            lines.append("  reduce sampling temperature before drawing conclusions from a comparison.")
        return "\n".join(lines)


def analyze_stability(runs: Sequence[EvalRun]) -> StabilityReport:
    if not runs:
        return StabilityReport()

    counts: dict[str, list[int]] = {}
    for run in runs:
        for r in run.results:
            if r.outcome in (Outcome.PASS, Outcome.FAIL):
                counts.setdefault(r.task_id, []).append(1 if r.is_pass else 0)

    stability = [
        TaskStability(task_id=tid, runs=len(observations), passes=sum(observations))
        for tid, observations in sorted(counts.items())
    ]

    return StabilityReport(
        runs=len(runs),
        tasks=len(stability),
        pass_rates=[r.pass_rate for r in runs],
        task_stability=stability,
    )


def intraclass_correlation(runs: Sequence[EvalRun]) -> float:
    """Fraction of total variance explained by real task difficulty.

    Near 1 means tasks are consistently easy or hard across runs. Near 0
    means outcomes are mostly noise, so per-task results carry little signal.
    """
    if len(runs) < 2:
        return 0.0

    counts: dict[str, list[float]] = {}
    for run in runs:
        for r in run.results:
            if r.outcome in (Outcome.PASS, Outcome.FAIL):
                counts.setdefault(r.task_id, []).append(1.0 if r.is_pass else 0.0)

    usable = [v for v in counts.values() if len(v) >= 2]
    if not usable:
        return 0.0

    k = min(len(v) for v in usable)
    trimmed = [v[:k] for v in usable]
    n = len(trimmed)

    grand = mean([x for v in trimmed for x in v])
    task_means = [mean(v) for v in trimmed]

    between = k * sum((m - grand) ** 2 for m in task_means) / (n - 1) if n > 1 else 0.0
    within_total = sum((x - task_means[i]) ** 2 for i, v in enumerate(trimmed) for x in v)
    within = within_total / (n * (k - 1)) if n * (k - 1) > 0 else 0.0

    if between + (k - 1) * within == 0:
        return 0.0
    return max(0.0, (between - within) / (between + (k - 1) * within))


def required_repeats(observed_stdev: float, target_precision: float = 0.02, level: float = 0.95) -> int:
    if observed_stdev <= 0 or target_precision <= 0:
        return 1
    z = _z_for(level)
    return max(1, math.ceil((z * observed_stdev / target_precision) ** 2))
