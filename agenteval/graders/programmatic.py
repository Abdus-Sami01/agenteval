from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agenteval.exceptions import ConfigurationError
from agenteval.graders.base import Grader
from agenteval.types import Score, Step, Task, Trajectory


class PredicateGrader(Grader):
    name = "predicate"

    def __init__(self, predicate: Callable[[Any, Task], bool], label: str = ""):
        self._predicate = predicate
        self._label = label or "predicate"

    def grade(self, prediction: Any, task: Task) -> Score:
        try:
            ok = bool(self._predicate(prediction, task))
        except Exception as e:
            return self._score(0.0, passed=False, detail=f"{self._label} raised {type(e).__name__}: {e}")
        return self._score(1.0 if ok else 0.0, passed=ok,
                           detail="" if ok else f"{self._label} returned False")


class CallableGrader(Grader):
    name = "callable"

    def __init__(self, fn: Callable[[Any, Task], Any], threshold: float = 1.0, label: str = ""):
        self._fn = fn
        self._threshold = threshold
        self._label = label or "callable"

    def grade(self, prediction: Any, task: Task) -> Score:
        try:
            raw = self._fn(prediction, task)
        except Exception as e:
            return self._score(0.0, passed=False, detail=f"{self._label} raised {type(e).__name__}: {e}")

        if isinstance(raw, Score):
            return raw

        if isinstance(raw, tuple) and len(raw) == 2:
            value, detail = raw
            value = float(value)
            return self._score(value, passed=value >= self._threshold, detail=str(detail))

        if isinstance(raw, bool):
            return self._score(1.0 if raw else 0.0, passed=raw)

        try:
            value = float(raw)
        except (TypeError, ValueError):
            return self._score(0.0, passed=False, detail=f"{self._label} returned non-numeric {raw!r}")

        return self._score(value, passed=value >= self._threshold)


def _as_trajectory(prediction: Any) -> Trajectory | None:
    if isinstance(prediction, Trajectory):
        return prediction
    if isinstance(prediction, list) and all(isinstance(s, Step) for s in prediction):
        return Trajectory(steps=list(prediction))
    return None


class OutcomeGrader(Grader):
    """Grade an agent's final answer with any ordinary grader.

    Wraps a grader that expects a plain prediction so it can score a
    `Trajectory` instead, by handing it `trajectory.output`. Anything that is
    not a trajectory is passed straight through, so the same grader works on
    agents and on single-shot systems.
    """

    name = "outcome"

    def __init__(self, grader: Grader):
        self._grader = grader

    def grade(self, prediction: Any, task: Task) -> Score:
        traj = _as_trajectory(prediction)
        return self._grader.grade(traj.output if traj else prediction, task)

    def __repr__(self) -> str:
        return f"OutcomeGrader({self._grader!r})"


class ToolSequenceGrader(Grader):
    """Check which tools an agent called, and in what order.

    `mode` picks how strict that is: `exact` demands the full sequence,
    `subsequence` allows extra calls in between as long as the expected ones
    appear in order, and `set` ignores order and repetition entirely. The
    expected sequence comes from `task.expected` unless one is given here.

    Partial credit is the fraction of expected calls matched, so a run that
    gets two of three tools right scores above one that gets none.
    """

    name = "tool_sequence"
    MODES = ("exact", "subsequence", "set")

    def __init__(self, expected: list[str] | None = None, mode: str = "subsequence",
                 forbidden: list[str] | None = None, threshold: float = 1.0):
        if mode not in self.MODES:
            raise ConfigurationError(f"mode must be one of {self.MODES}, got {mode!r}")
        self._expected = expected
        self._mode = mode
        self._forbidden = set(forbidden or ())
        self._threshold = threshold

    def _expected_for(self, task: Task) -> list[str]:
        if self._expected is not None:
            return list(self._expected)
        raw = task.expected
        if isinstance(raw, dict):
            raw = raw.get("tools", raw.get("actions", []))
        return [str(a) for a in raw] if isinstance(raw, (list, tuple)) else []

    def grade(self, prediction: Any, task: Task) -> Score:
        traj = _as_trajectory(prediction)
        if traj is None:
            return self._score(0.0, passed=False, detail="prediction is not a Trajectory")

        actions = traj.actions
        used_forbidden = sorted(self._forbidden.intersection(actions))
        if used_forbidden:
            return self._score(0.0, passed=False,
                               detail=f"called forbidden tools: {', '.join(used_forbidden)}")

        expected = self._expected_for(task)
        if not expected:
            return self._score(0.0, passed=False, detail="no expected tool sequence given")

        if self._mode == "set":
            matched = len(set(expected) & set(actions))
            value = matched / len(set(expected))
            missing = sorted(set(expected) - set(actions))
        elif self._mode == "exact":
            matched = sum(1 for e, a in zip(expected, actions, strict=False) if e == a)
            value = matched / len(expected)
            if len(actions) != len(expected):
                value = min(value, 0.99)
            missing = [e for e in expected if e not in actions]
        else:
            matched = _longest_prefix_in_order(expected, actions)
            value = matched / len(expected)
            missing = expected[matched:]

        detail = ""
        if value < 1.0:
            detail = f"expected {expected}, got {actions}"
            if missing:
                detail += f"; missing {missing}"
        return self._score(value, passed=value >= self._threshold, detail=detail)


def _longest_prefix_in_order(expected: list[str], actions: list[str]) -> int:
    matched = 0
    for action in actions:
        if matched < len(expected) and action == expected[matched]:
            matched += 1
    return matched


class StepBudgetGrader(Grader):
    """Fail agents that reach the answer wastefully, or not cleanly.

    A correct answer after forty tool calls is usually not the answer you
    want shipped, and neither is one that got there through a string of
    failed calls. Score decays linearly once a budget is exceeded rather
    than snapping to zero, so a small overrun ranks above a large one.
    """

    name = "step_budget"

    def __init__(self, max_steps: int = 0, max_cost: float = 0.0, allow_errors: bool = True):
        if max_steps < 0 or max_cost < 0:
            raise ConfigurationError("budgets must be non-negative")
        self._max_steps = max_steps
        self._max_cost = max_cost
        self._allow_errors = allow_errors

    def grade(self, prediction: Any, task: Task) -> Score:
        traj = _as_trajectory(prediction)
        if traj is None:
            return self._score(0.0, passed=False, detail="prediction is not a Trajectory")

        subscores: dict[str, float] = {}
        breaches: list[str] = []

        if self._max_steps:
            subscores["steps"] = _budget_score(len(traj), self._max_steps)
            if len(traj) > self._max_steps:
                breaches.append(f"{len(traj)} steps over budget of {self._max_steps}")

        if self._max_cost:
            subscores["cost"] = _budget_score(traj.total_cost, self._max_cost)
            if traj.total_cost > self._max_cost:
                breaches.append(f"cost {traj.total_cost:.4f} over budget of {self._max_cost:.4f}")

        if not self._allow_errors:
            failed = traj.failed_steps
            subscores["clean"] = 1.0 if not failed else 0.0
            if failed:
                breaches.append(f"{len(failed)} failed step(s): {failed[0].error[:60]}")

        if not subscores:
            return self._score(1.0, passed=True, detail="no budget configured")

        value = sum(subscores.values()) / len(subscores)
        return self._score(value, passed=not breaches, detail="; ".join(breaches), **subscores)


def _budget_score(used: float, budget: float) -> float:
    """1.0 within budget, decaying to 0.0 at twice the budget."""
    if used <= budget:
        return 1.0
    return max(0.0, 1.0 - (used - budget) / budget)
