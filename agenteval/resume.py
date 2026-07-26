from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

from agenteval.graders.base import Grader
from agenteval.types import (
    EvalRun,
    Outcome,
    RunMetadata,
    Score,
    Task,
    TaskResult,
    TaskSuite,
)


class ResultStore:
    """Append-only record of finished tasks so a crashed run can resume.

    Each completed task is flushed to disk immediately, so an interrupted
    run loses at most the task that was in flight.
    """

    def __init__(self, path: str):
        self.path = path
        self._results: dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        if os.path.exists(path):
            self.load()

    def load(self) -> int:
        self._results.clear()
        if not os.path.exists(self.path):
            return 0

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                result = _result_from_dict(record)
                if result:
                    self._results[result.task_id] = result
        return len(self._results)

    def append(self, result: TaskResult) -> None:
        with self._lock:
            self._results[result.task_id] = result
            directory = os.path.dirname(os.path.abspath(self.path)) or "."
            os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_result_to_dict(result), default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def has(self, task_id: str) -> bool:
        return task_id in self._results

    def get(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    def clear(self) -> None:
        with self._lock:
            self._results.clear()
            if os.path.exists(self.path):
                os.unlink(self.path)

    def compact(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for result in self._results.values():
                    f.write(json.dumps(_result_to_dict(result), default=str) + "\n")
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @property
    def completed_ids(self) -> set[str]:
        return set(self._results)

    @property
    def results(self) -> list[TaskResult]:
        return list(self._results.values())

    def __len__(self) -> int:
        return len(self._results)


def evaluate_resumable(
    system: Callable[[Task], Any],
    suite: TaskSuite,
    grader: Grader,
    store_path: str,
    seed: int = 0,
    system_name: str = "",
    timeout_s: float = 0,
    retries: int = 0,
    fresh: bool = False,
    on_result: Callable[[TaskResult], None] | None = None,
) -> tuple[EvalRun, int]:
    from agenteval.runner import detect_git_sha, evaluate

    store = ResultStore(store_path)
    if fresh:
        store.clear()

    already = store.completed_ids
    pending = [t for t in suite.tasks if t.id not in already]
    reused = len(suite) - len(pending)

    start = time.perf_counter()

    for task in pending:
        step = evaluate(
            system, TaskSuite(name=suite.name, tasks=[task]), grader,
            seed=seed, timeout_s=timeout_s, retries=retries, system_name=system_name,
        )
        for result in step.results:
            store.append(result)
            if on_result:
                on_result(result)

    ordered = [store.get(t.id) for t in suite.tasks if store.has(t.id)]

    metadata = RunMetadata(
        run_id=f"resumable-{abs(hash(store_path)) % 10**8:08d}",
        suite_name=suite.name,
        system_name=system_name or getattr(system, "__name__", "system"),
        seed=seed,
        git_sha=detect_git_sha(),
        notes=f"resumed with {reused} cached result(s)" if reused else "",
    )

    run = EvalRun(
        metadata=metadata,
        results=[r for r in ordered if r is not None],
        total_ms=(time.perf_counter() - start) * 1000,
    )
    return run, reused


def _result_to_dict(r: TaskResult) -> dict[str, Any]:
    d: dict[str, Any] = {
        "task_id": r.task_id,
        "outcome": r.outcome.value,
        "elapsed_ms": r.elapsed_ms,
        "attempts": r.attempts,
        "tags": list(r.tags),
        "weight": r.weight,
        "error": r.error,
        "prediction": str(r.prediction)[:2000] if r.prediction is not None else None,
        "expected": str(r.expected)[:2000] if r.expected is not None else None,
    }
    if r.score:
        d["score"] = {
            "value": r.score.value,
            "passed": r.score.passed,
            "grader": r.score.grader,
            "detail": r.score.detail,
            "subscores": r.score.subscores,
        }
    return d


def _result_from_dict(d: dict[str, Any]) -> TaskResult | None:
    task_id = d.get("task_id")
    if not task_id:
        return None

    score = None
    raw = d.get("score")
    if isinstance(raw, dict):
        score = Score(
            value=float(raw.get("value", 0.0)),
            passed=bool(raw.get("passed", False)),
            grader=raw.get("grader", ""),
            detail=raw.get("detail", ""),
            subscores=raw.get("subscores", {}) or {},
        )

    try:
        outcome = Outcome(d.get("outcome", "error"))
    except ValueError:
        outcome = Outcome.ERROR

    return TaskResult(
        task_id=task_id,
        outcome=outcome,
        score=score,
        prediction=d.get("prediction"),
        expected=d.get("expected"),
        error=d.get("error", "") or "",
        elapsed_ms=float(d.get("elapsed_ms", 0.0)),
        attempts=int(d.get("attempts", 1)),
        tags=tuple(d.get("tags", ()) or ()),
        weight=float(d.get("weight", 1.0)),
    )
