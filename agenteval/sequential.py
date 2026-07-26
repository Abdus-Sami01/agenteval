from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agenteval.graders.base import Grader
from agenteval.stats import wilson_interval
from agenteval.types import EvalRun, Outcome, Task, TaskSuite


@dataclass
class StoppingDecision:
    stop: bool
    reason: str
    observed: int = 0
    successes: int = 0
    lower: float = 0.0
    upper: float = 1.0

    def __str__(self) -> str:
        state = "STOP" if self.stop else "CONTINUE"
        return f"[{state}] after {self.observed}: {self.reason}"


class SequentialGate:
    """Early stopping for a pass-rate threshold using an alpha-spending bound.

    The bound widens the required evidence at small n, so peeking after every
    task does not inflate the false-positive rate the way a naive fixed-alpha
    check would.
    """

    def __init__(
        self,
        threshold: float,
        alpha: float = 0.05,
        min_samples: int = 20,
        max_samples: int = 0,
    ):
        self._threshold = threshold
        self._alpha = alpha
        self._min = max(1, min_samples)
        self._max = max_samples

    def check(self, successes: int, observed: int) -> StoppingDecision:
        if observed < self._min:
            return StoppingDecision(False, f"need at least {self._min} samples", observed, successes)

        level = 1 - self._spending(observed)
        ci = wilson_interval(successes, observed, level)

        if ci.low > self._threshold:
            return StoppingDecision(
                True,
                f"pass rate is above {self._threshold:.1%} "
                f"(lower bound {ci.low:.1%} at adjusted level {level:.3f})",
                observed, successes, ci.low, ci.high,
            )

        if ci.high < self._threshold:
            return StoppingDecision(
                True,
                f"pass rate is below {self._threshold:.1%} "
                f"(upper bound {ci.high:.1%} at adjusted level {level:.3f})",
                observed, successes, ci.low, ci.high,
            )

        if self._max and observed >= self._max:
            return StoppingDecision(
                True,
                f"budget of {self._max} tasks exhausted without a conclusive result",
                observed, successes, ci.low, ci.high,
            )

        return StoppingDecision(
            False,
            f"interval [{ci.low:.1%}, {ci.high:.1%}] still spans {self._threshold:.1%}",
            observed, successes, ci.low, ci.high,
        )

    def _spending(self, observed: int) -> float:
        horizon = self._max or max(observed * 2, self._min * 4)
        fraction = min(1.0, observed / horizon)
        if fraction <= 0:
            return self._alpha * 1e-6
        return self._alpha * (fraction ** 2)


@dataclass
class SequentialRun:
    run: EvalRun
    decision: StoppingDecision
    evaluated: int = 0
    total_available: int = 0
    saved: int = 0

    @property
    def saved_fraction(self) -> float:
        return self.saved / self.total_available if self.total_available else 0.0

    def summary(self) -> str:
        return "\n".join([
            f"  evaluated   {self.evaluated} of {self.total_available} tasks",
            f"  skipped     {self.saved} ({self.saved_fraction:.0%} of the suite)",
            f"  decision    {self.decision.reason}",
            f"  pass rate   {self.run.pass_rate:.1%}",
        ])


def evaluate_sequential(
    system: Callable[[Task], Any],
    suite: TaskSuite,
    grader: Grader,
    threshold: float,
    alpha: float = 0.05,
    min_samples: int = 20,
    max_samples: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    system_name: str = "",
) -> SequentialRun:
    import random as _random

    from agenteval.runner import evaluate

    tasks = list(suite.tasks)
    if shuffle:
        _random.Random(seed).shuffle(tasks)

    gate = SequentialGate(threshold, alpha, min_samples, max_samples or len(tasks))
    processed: list[Task] = []
    successes = 0
    decision = StoppingDecision(False, "not started")
    partial_results = []

    for task in tasks:
        single = TaskSuite(name=suite.name, tasks=[task])
        step = evaluate(system, single, grader, seed=seed, system_name=system_name)
        partial_results.extend(step.results)
        processed.append(task)

        if step.results and step.results[0].outcome == Outcome.PASS:
            successes += 1

        decision = gate.check(successes, len(processed))
        if decision.stop:
            break

    final = evaluate(system, TaskSuite(name=suite.name, tasks=[]), grader,
                     seed=seed, system_name=system_name)
    final.results = partial_results

    return SequentialRun(
        run=final,
        decision=decision,
        evaluated=len(processed),
        total_available=len(tasks),
        saved=len(tasks) - len(processed),
    )
