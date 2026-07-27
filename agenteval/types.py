from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Outcome(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class SystemUnderTest(Protocol):
    def __call__(self, task: Task) -> Any: ...


@dataclass(frozen=True)
class Task:
    id: str
    input: Any
    expected: Any = None
    tags: tuple[str, ...] = ()
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSuite:
    name: str
    tasks: list[Task] = field(default_factory=list)
    description: str = ""

    def filter(self, tags: set[str] | None = None, ids: set[str] | None = None) -> TaskSuite:
        selected = []
        for task in self.tasks:
            if ids and task.id not in ids:
                continue
            if tags and not (tags & set(task.tags)):
                continue
            selected.append(task)
        return TaskSuite(name=self.name, tasks=selected, description=self.description)

    def sample(self, n: int, seed: int = 0) -> TaskSuite:
        import random

        rng = random.Random(seed)
        picked = rng.sample(self.tasks, min(n, len(self.tasks)))
        return TaskSuite(name=f"{self.name}[sample:{n}]", tasks=picked, description=self.description)

    @property
    def all_tags(self) -> set[str]:
        return {t for task in self.tasks for t in task.tags}

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass(frozen=True)
class Step:
    """One action an agent took, plus what came back."""

    action: str
    args: dict[str, Any] = field(default_factory=dict)
    observation: Any = None
    error: str = ""
    cost: float = 0.0
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"action": self.action}
        if self.args:
            d["args"] = self.args
        if self.observation is not None:
            d["observation"] = self.observation
        if self.error:
            d["error"] = self.error
        if self.cost:
            d["cost"] = self.cost
        if self.elapsed_ms:
            d["elapsed_ms"] = round(self.elapsed_ms, 2)
        return d


@dataclass
class Trajectory:
    """What an agent did on its way to an answer.

    Return one of these from the system under test to grade the process as
    well as the result: which tools were called, in what order, how many
    steps it burned, and what it finally produced.
    """

    steps: list[Step] = field(default_factory=list)
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]], output: Any = None, **metadata) -> Trajectory:
        steps = [
            Step(
                action=str(r.get("action", r.get("tool", ""))),
                args=dict(r.get("args", r.get("input", {})) or {}),
                observation=r.get("observation", r.get("output")),
                error=str(r.get("error", "") or ""),
                cost=float(r.get("cost", 0.0) or 0.0),
                elapsed_ms=float(r.get("elapsed_ms", 0.0) or 0.0),
            )
            for r in records
        ]
        return cls(steps=steps, output=output, metadata=dict(metadata))

    @property
    def actions(self) -> list[str]:
        return [s.action for s in self.steps]

    @property
    def failed_steps(self) -> list[Step]:
        return [s for s in self.steps if not s.ok]

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.steps)

    @property
    def total_ms(self) -> float:
        return sum(s.elapsed_ms for s in self.steps)

    def count(self, action: str) -> int:
        return sum(1 for s in self.steps if s.action == action)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.as_dict() for s in self.steps],
            "output": self.output,
            "metadata": self.metadata,
        }

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        chain = " -> ".join(self.actions[:6])
        if len(self.steps) > 6:
            chain += f" -> ... ({len(self.steps)} steps)"
        return f"Trajectory[{chain or 'empty'}] output={self.output!r}"


@dataclass(frozen=True)
class Score:
    value: float
    passed: bool
    grader: str = ""
    detail: str = ""
    subscores: dict[str, float] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    outcome: Outcome
    score: Score | None = None
    prediction: Any = None
    expected: Any = None
    error: str = ""
    elapsed_ms: float = 0.0
    attempts: int = 1
    tags: tuple[str, ...] = ()
    weight: float = 1.0

    @property
    def numeric(self) -> float:
        return self.score.value if self.score else 0.0

    @property
    def is_pass(self) -> bool:
        return self.outcome == Outcome.PASS


@dataclass
class RunMetadata:
    run_id: str = ""
    suite_name: str = ""
    system_name: str = ""
    seed: int = 0
    started_at: float = field(default_factory=time.time)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform_name: str = field(default_factory=platform.platform)
    git_sha: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "suite": self.suite_name,
            "system": self.system_name,
            "seed": self.seed,
            "started_at": self.started_at,
            "python": self.python_version,
            "platform": self.platform_name,
            "git_sha": self.git_sha,
            "notes": self.notes,
            **self.extra,
        }


@dataclass
class EvalRun:
    metadata: RunMetadata
    results: list[TaskResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.outcome == Outcome.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.outcome == Outcome.FAIL)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.outcome in (Outcome.ERROR, Outcome.TIMEOUT))

    @property
    def scored(self) -> list[TaskResult]:
        return [r for r in self.results if r.outcome in (Outcome.PASS, Outcome.FAIL)]

    @property
    def pass_rate(self) -> float:
        graded = self.passed + self.failed
        return self.passed / graded if graded else 0.0

    @property
    def mean_score(self) -> float:
        scored = self.scored
        if not scored:
            return 0.0
        total_weight = sum(r.weight for r in scored) or 1.0
        return sum(r.numeric * r.weight for r in scored) / total_weight

    def by_tag(self) -> dict[str, list[TaskResult]]:
        grouped: dict[str, list[TaskResult]] = {}
        for r in self.results:
            for tag in r.tags:
                grouped.setdefault(tag, []).append(r)
        return grouped

    def scores(self) -> list[float]:
        return [r.numeric for r in self.scored]

    def result_for(self, task_id: str) -> TaskResult | None:
        for r in self.results:
            if r.task_id == task_id:
                return r
        return None

    def __len__(self) -> int:
        return len(self.results)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str = ""


@dataclass
class GateReport:
    gates: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def summary(self) -> str:
        if not self.gates:
            return "No gates configured."
        lines = []
        for g in self.gates:
            mark = "PASS" if g.passed else "FAIL"
            lines.append(f"  [{mark}] {g.name}: observed {g.observed:.4f} vs threshold {g.threshold:.4f}"
                         + (f" - {g.detail}" if g.detail else ""))
        verdict = "ALL GATES PASSED" if self.passed else f"{len(self.failures)} GATE(S) FAILED"
        lines.append("")
        lines.append(verdict)
        return "\n".join(lines)
