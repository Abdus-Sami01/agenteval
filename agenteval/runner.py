from __future__ import annotations

import logging
import random
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
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

logger = logging.getLogger("agenteval")


def detect_git_sha(path: str = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def evaluate(
    system: Callable[[Task], Any],
    suite: TaskSuite,
    grader: Grader | Callable[[Any, Task], Score],
    max_parallel: int = 1,
    timeout_s: float = 0,
    retries: int = 0,
    seed: int = 0,
    system_name: str = "",
    notes: str = "",
    on_result: Callable[[TaskResult], None] | None = None,
    grader_for: Callable[[Task], Grader] | None = None,
    progress: bool = False,
) -> EvalRun:
    random.seed(seed)

    reporter = None
    if progress:
        from agenteval.progress import ProgressReporter

        reporter = ProgressReporter(len(suite))
        user_callback = on_result

        def on_result(result: TaskResult) -> None:  # type: ignore[misc]
            reporter.update(result)
            if user_callback:
                user_callback(result)

    metadata = RunMetadata(
        run_id=uuid.uuid4().hex[:12],
        suite_name=suite.name,
        system_name=system_name or getattr(system, "__name__", "system"),
        seed=seed,
        git_sha=detect_git_sha(),
        notes=notes,
    )

    start = time.perf_counter()
    results: list[TaskResult] = []

    def run_one(task: Task) -> TaskResult:
        chosen = grader_for(task) if grader_for else grader
        attempt = 0
        last_error = ""
        t0 = time.perf_counter()

        while attempt <= retries:
            attempt += 1
            try:
                if timeout_s > 0:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        prediction = pool.submit(system, task).result(timeout=timeout_s)
                else:
                    prediction = system(task)
            except FuturesTimeout:
                last_error = f"timed out after {timeout_s}s"
                continue
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                continue

            elapsed = (time.perf_counter() - t0) * 1000
            try:
                score = chosen.grade(prediction, task) if isinstance(chosen, Grader) else chosen(prediction, task)
            except Exception as e:
                return TaskResult(
                    task_id=task.id, outcome=Outcome.ERROR,
                    prediction=prediction, expected=task.expected,
                    error=f"grader failed: {type(e).__name__}: {e}",
                    elapsed_ms=elapsed, attempts=attempt,
                    tags=task.tags, weight=task.weight,
                )

            return TaskResult(
                task_id=task.id,
                outcome=Outcome.PASS if score.passed else Outcome.FAIL,
                score=score,
                prediction=prediction,
                expected=task.expected,
                elapsed_ms=elapsed,
                attempts=attempt,
                tags=task.tags,
                weight=task.weight,
            )

        elapsed = (time.perf_counter() - t0) * 1000
        outcome = Outcome.TIMEOUT if "timed out" in last_error else Outcome.ERROR
        return TaskResult(
            task_id=task.id, outcome=outcome, expected=task.expected,
            error=last_error, elapsed_ms=elapsed, attempts=attempt,
            tags=task.tags, weight=task.weight,
        )

    if max_parallel > 1 and len(suite) > 1:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(run_one, t): t for t in suite.tasks}
            collected: dict[str, TaskResult] = {}
            for future in futures:
                task = futures[future]
                try:
                    collected[task.id] = future.result()
                except Exception as e:
                    collected[task.id] = TaskResult(
                        task_id=task.id, outcome=Outcome.ERROR,
                        error=str(e), tags=task.tags, weight=task.weight,
                    )
            results = [collected[t.id] for t in suite.tasks if t.id in collected]
            if on_result:
                for r in results:
                    on_result(r)
    else:
        for task in suite.tasks:
            result = run_one(task)
            results.append(result)
            if on_result:
                on_result(result)

    if reporter is not None:
        reporter.finish()

    run = EvalRun(
        metadata=metadata,
        results=results,
        total_ms=(time.perf_counter() - start) * 1000,
    )
    logger.info(
        "evaluated %s on %s: %d/%d passed (%.1f%%) in %.0fms",
        metadata.system_name, metadata.suite_name,
        run.passed, run.passed + run.failed, run.pass_rate * 100, run.total_ms,
    )
    return run


def iter_evaluate(
    system: Callable[[Task], Any],
    suite: TaskSuite,
    grader: Grader,
    seed: int = 0,
    timeout_s: float = 0,
    retries: int = 0,
    system_name: str = "",
) -> Iterator[TaskResult]:
    """Yield results one task at a time.

    Use this for suites too large to hold in memory, or to stream results
    into a database or dashboard as they arrive. Nothing is accumulated,
    so peak memory stays flat regardless of suite size.
    """
    for task in suite.tasks:
        single = TaskSuite(name=suite.name, tasks=[task])
        run = evaluate(
            system, single, grader,
            seed=seed, timeout_s=timeout_s, retries=retries, system_name=system_name,
        )
        yield from run.results


def evaluate_many(
    systems: dict[str, Callable[[Task], Any]],
    suite: TaskSuite,
    grader: Grader,
    **kwargs,
) -> dict[str, EvalRun]:
    kwargs.pop("system_name", None)
    return {
        name: evaluate(system, suite, grader, system_name=name, **kwargs)
        for name, system in systems.items()
    }


def repeat_evaluate(
    system: Callable[[Task], Any],
    suite: TaskSuite,
    grader: Grader,
    repeats: int = 3,
    base_seed: int = 0,
    **kwargs,
) -> list[EvalRun]:
    kwargs.pop("seed", None)
    return [
        evaluate(system, suite, grader, seed=base_seed + i, **kwargs)
        for i in range(repeats)
    ]
