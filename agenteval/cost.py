from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from agenteval.types import EvalRun, Outcome, Task


@dataclass(frozen=True)
class CostEntry:
    task_id: str
    amount: float
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class CostReport:
    total: float = 0.0
    entries: list[CostEntry] = field(default_factory=list)
    unit: str = "usd"
    budget: float = 0.0
    stopped_early: bool = False

    @property
    def tasks(self) -> int:
        return len(self.entries)

    @property
    def per_task(self) -> float:
        return self.total / self.tasks if self.tasks else 0.0

    @property
    def tokens_in(self) -> int:
        return sum(e.tokens_in for e in self.entries)

    @property
    def tokens_out(self) -> int:
        return sum(e.tokens_out for e in self.entries)

    def cost_per_pass(self, run: EvalRun) -> float:
        return self.total / run.passed if run.passed else float("inf")

    def project(self, n_tasks: int) -> float:
        return self.per_task * n_tasks

    def summary(self, run: EvalRun | None = None) -> str:
        lines = [
            f"  tasks billed        {self.tasks}",
            f"  total cost          {self.total:.4f} {self.unit}",
            f"  cost per task       {self.per_task:.6f} {self.unit}",
        ]
        if self.tokens_in or self.tokens_out:
            lines.append(f"  tokens in / out     {self.tokens_in:,} / {self.tokens_out:,}")
        if self.budget:
            pct = self.total / self.budget * 100
            lines.append(f"  budget              {self.budget:.4f} ({pct:.1f}% used)")
        if run:
            per_pass = self.cost_per_pass(run)
            shown = f"{per_pass:.6f}" if per_pass != float("inf") else "n/a (no passes)"
            lines.append(f"  cost per passing    {shown} {self.unit}")
        if self.stopped_early:
            lines.append("")
            lines.append("  RUN STOPPED EARLY: budget was exhausted before the suite finished.")
            lines.append("  Reported scores cover only the tasks that ran.")
        lines.append("")
        for n in (100, 1_000, 10_000):
            lines.append(f"  projected for {n:>6,} tasks: {self.project(n):.2f} {self.unit}")
        return "\n".join(lines)


class BudgetExceeded(Exception):
    pass


class CostTracker:
    def __init__(self, budget: float = 0.0, unit: str = "usd", hard_stop: bool = True):
        self._report = CostReport(unit=unit, budget=budget)
        self._budget = budget
        self._hard_stop = hard_stop
        self._lock = threading.Lock()

    def charge(self, task_id: str, amount: float, tokens_in: int = 0, tokens_out: int = 0) -> None:
        with self._lock:
            self._report.entries.append(CostEntry(task_id, amount, tokens_in, tokens_out))
            self._report.total += amount

    def would_exceed(self, amount: float) -> bool:
        if self._budget <= 0:
            return False
        return self._report.total + amount > self._budget

    @property
    def exhausted(self) -> bool:
        return self._budget > 0 and self._report.total >= self._budget

    @property
    def remaining(self) -> float:
        return max(0.0, self._budget - self._report.total) if self._budget > 0 else float("inf")

    def wrap(
        self,
        system: Callable[[Task], Any],
        cost_fn: Callable[[Task, Any], float] | None = None,
        fixed_cost: float = 0.0,
        estimate_fn: Callable[[Task], float] | None = None,
    ) -> Callable[[Task], Any]:
        def costed(task: Task) -> Any:
            estimate = estimate_fn(task) if estimate_fn else fixed_cost

            if self._hard_stop and self.would_exceed(estimate):
                self._report.stopped_early = True
                raise BudgetExceeded(
                    f"budget {self._budget:.4f} {self._report.unit} would be exceeded by "
                    f"{task.id!r} (estimated {estimate:.6f}, remaining {self.remaining:.6f})"
                )

            prediction = system(task)
            actual = cost_fn(task, prediction) if cost_fn else fixed_cost
            if actual:
                self.charge(task.id, actual)
            return prediction

        return costed

    @property
    def report(self) -> CostReport:
        return self._report


TOKEN_PRICES = {
    "input": 0.003 / 1000,
    "output": 0.015 / 1000,
}


def estimate_tokens(text: Any) -> int:
    return max(1, len(str(text)) // 4)


def token_cost(prompt: Any, completion: Any,
               input_price: float | None = None,
               output_price: float | None = None) -> float:
    pin = input_price if input_price is not None else TOKEN_PRICES["input"]
    pout = output_price if output_price is not None else TOKEN_PRICES["output"]
    return estimate_tokens(prompt) * pin + estimate_tokens(completion) * pout


def cost_efficiency(runs: dict[str, EvalRun], costs: dict[str, CostReport]) -> str:
    rows = []
    for name, run in runs.items():
        report = costs.get(name)
        if not report:
            continue
        per_pass = report.cost_per_pass(run)
        rows.append((name, run.pass_rate, report.total, report.per_task, per_pass))

    if not rows:
        return "No cost data."

    rows.sort(key=lambda r: r[4])
    width = max(len(r[0]) for r in rows) + 2
    lines = [
        f"{'system':<{width}}{'pass rate':>11}{'total':>11}{'per task':>12}{'per pass':>12}",
        "-" * (width + 46),
    ]
    for name, rate, total, per_task, per_pass in rows:
        shown = f"{per_pass:.5f}" if per_pass != float("inf") else "n/a"
        lines.append(f"{name:<{width}}{rate:>10.1%}{total:>11.4f}{per_task:>12.6f}{shown:>12}")

    lines.append("")
    lines.append(f"  cheapest per passing task: {rows[0][0]}")
    return "\n".join(lines)
