from __future__ import annotations

import asyncio
import inspect
import logging
import random
import subprocess
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any

from agenteval.exceptions import ConfigurationError
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


@dataclass(frozen=True)
class RetryPolicy:
    """How to retry a system call that failed or timed out.

    Delays grow geometrically and carry random jitter, so a suite that hits a
    rate limit does not resend every task in lockstep. `retry_on` decides which
    exceptions are worth another attempt; the default retries everything, which
    suits flaky network calls but will happily retry a bug too.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.5
    retry_on: Callable[[BaseException], bool] = lambda exc: True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigurationError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ConfigurationError("retry delays must be non-negative")
        if self.multiplier < 1:
            raise ConfigurationError(f"multiplier must be at least 1, got {self.multiplier}")
        if not 0 <= self.jitter <= 1:
            raise ConfigurationError(f"jitter must be between 0 and 1, got {self.jitter}")

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        return attempt < self.max_attempts and self.retry_on(exc)

    def delay_for(self, attempt: int, rng: random.Random | None = None) -> float:
        raw = min(self.base_delay_s * self.multiplier ** (attempt - 1), self.max_delay_s)
        if not self.jitter:
            return raw
        draw = (rng or random).uniform(1 - self.jitter, 1 + self.jitter)
        return max(0.0, min(raw * draw, self.max_delay_s))

    @classmethod
    def none(cls) -> RetryPolicy:
        return cls(max_attempts=1, base_delay_s=0.0, jitter=0.0)


@dataclass
class RateLimiter:
    """Token bucket capping how fast system calls leave the process.

    Shared by every worker, so `max_parallel` controls concurrency and this
    controls throughput. Up to `burst` calls may go out at once; after that
    callers are paced to `rate_per_s`.
    """

    rate_per_s: float
    burst: int = 1
    _tokens: float = field(default=0.0, init=False)
    _updated: float = field(default_factory=time.monotonic, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rate_per_s <= 0:
            raise ConfigurationError(f"rate_per_s must be positive, got {self.rate_per_s}")
        if self.burst < 1:
            raise ConfigurationError(f"burst must be at least 1, got {self.burst}")
        self._tokens = float(self.burst)

    def _claim(self) -> float:
        """Take a token, returning how long the caller must wait to use it."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate_per_s)
            self._updated = now
            if self._tokens >= 1:
                self._tokens -= 1
                return 0.0
            wait = (1 - self._tokens) / self.rate_per_s
            self._tokens -= 1
            self._updated = now + wait
            return wait

    def acquire(self) -> None:
        wait = self._claim()
        if wait > 0:
            time.sleep(wait)

    async def acquire_async(self) -> None:
        wait = self._claim()
        if wait > 0:
            await asyncio.sleep(wait)


def _resolve_policy(retries: int, retry_policy: RetryPolicy | None) -> RetryPolicy:
    if retry_policy is not None:
        return retry_policy
    if retries < 0:
        raise ConfigurationError(f"retries must be non-negative, got {retries}")
    return RetryPolicy(max_attempts=retries + 1, base_delay_s=0.0, jitter=0.0)


def _call_with_timeout(system: Callable[[Task], Any], task: Task, timeout_s: float) -> Any:
    """Run `system(task)`, abandoning the worker thread if it overruns.

    The executor is shut down without waiting, so a system that ignores the
    timeout leaks one thread instead of stalling the whole evaluation.
    """
    if timeout_s <= 0:
        return system(task)
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(system, task)
    try:
        return future.result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _score_result(
    task: Task,
    prediction: Any,
    chosen: Grader | Callable[[Any, Task], Score],
    elapsed_ms: float,
    attempt: int,
) -> TaskResult:
    try:
        score = chosen.grade(prediction, task) if isinstance(chosen, Grader) else chosen(prediction, task)
    except Exception as e:
        return TaskResult(
            task_id=task.id, outcome=Outcome.ERROR,
            prediction=prediction, expected=task.expected,
            error=f"grader failed: {type(e).__name__}: {e}",
            elapsed_ms=elapsed_ms, attempts=attempt,
            tags=task.tags, weight=task.weight,
        )

    return TaskResult(
        task_id=task.id,
        outcome=Outcome.PASS if score.passed else Outcome.FAIL,
        score=score,
        prediction=prediction,
        expected=task.expected,
        elapsed_ms=elapsed_ms,
        attempts=attempt,
        tags=task.tags,
        weight=task.weight,
    )


def _exhausted_result(task: Task, last_error: str, elapsed_ms: float, attempt: int) -> TaskResult:
    return TaskResult(
        task_id=task.id,
        outcome=Outcome.TIMEOUT if "timed out" in last_error else Outcome.ERROR,
        expected=task.expected,
        error=last_error,
        elapsed_ms=elapsed_ms,
        attempts=attempt,
        tags=task.tags,
        weight=task.weight,
    )


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
    retry_policy: RetryPolicy | None = None,
    rate_limiter: RateLimiter | None = None,
) -> EvalRun:
    random.seed(seed)
    if max_parallel < 1:
        raise ConfigurationError(f"max_parallel must be at least 1, got {max_parallel}")
    policy = _resolve_policy(retries, retry_policy)

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
    rng = random.Random(seed)

    def run_one(task: Task) -> TaskResult:
        chosen = grader_for(task) if grader_for else grader
        attempt = 0
        last_error = ""
        t0 = time.perf_counter()

        while attempt < policy.max_attempts:
            attempt += 1
            try:
                if rate_limiter is not None:
                    rate_limiter.acquire()
                prediction = _call_with_timeout(system, task, timeout_s)
            except FuturesTimeout as e:
                last_error = f"timed out after {timeout_s}s"
                if not policy.should_retry(e, attempt):
                    break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if not policy.should_retry(e, attempt):
                    break
            else:
                return _score_result(task, prediction, chosen, (time.perf_counter() - t0) * 1000, attempt)

            delay = policy.delay_for(attempt, rng)
            if delay > 0:
                time.sleep(delay)

        return _exhausted_result(task, last_error, (time.perf_counter() - t0) * 1000, attempt)

    if max_parallel > 1 and len(suite) > 1:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(run_one, t): t for t in suite.tasks}
            collected: dict[str, TaskResult] = {}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    collected[task.id] = future.result()
                except Exception as e:
                    collected[task.id] = TaskResult(
                        task_id=task.id, outcome=Outcome.ERROR,
                        error=str(e), tags=task.tags, weight=task.weight,
                    )
                if on_result:
                    on_result(collected[task.id])
            results = [collected[t.id] for t in suite.tasks if t.id in collected]
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


async def evaluate_async(
    system: Callable[[Task], Awaitable[Any]],
    suite: TaskSuite,
    grader: Grader | Callable[[Any, Task], Score],
    max_parallel: int = 8,
    timeout_s: float = 0,
    retries: int = 0,
    seed: int = 0,
    system_name: str = "",
    notes: str = "",
    on_result: Callable[[TaskResult], None] | None = None,
    grader_for: Callable[[Task], Grader] | None = None,
    retry_policy: RetryPolicy | None = None,
    rate_limiter: RateLimiter | None = None,
) -> EvalRun:
    """Evaluate a coroutine system with asyncio concurrency.

    Preferred over `evaluate(max_parallel=...)` when the system is already
    async: thousands of in-flight API calls cost coroutines instead of threads.
    Graders still run synchronously, on the event loop thread.
    """
    random.seed(seed)
    if max_parallel < 1:
        raise ConfigurationError(f"max_parallel must be at least 1, got {max_parallel}")
    policy = _resolve_policy(retries, retry_policy)

    metadata = RunMetadata(
        run_id=uuid.uuid4().hex[:12],
        suite_name=suite.name,
        system_name=system_name or getattr(system, "__name__", "system"),
        seed=seed,
        git_sha=detect_git_sha(),
        notes=notes,
    )

    start = time.perf_counter()
    rng = random.Random(seed)
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_one(task: Task) -> TaskResult:
        chosen = grader_for(task) if grader_for else grader
        attempt = 0
        last_error = ""
        t0 = time.perf_counter()

        while attempt < policy.max_attempts:
            attempt += 1
            try:
                async with semaphore:
                    if rate_limiter is not None:
                        await rate_limiter.acquire_async()
                    call = system(task)
                    if not inspect.isawaitable(call):
                        raise TypeError(
                            f"evaluate_async needs an awaitable system, got {type(call).__name__}"
                        )
                    prediction = await (
                        asyncio.wait_for(call, timeout_s) if timeout_s > 0 else call
                    )
            except asyncio.TimeoutError as e:
                last_error = f"timed out after {timeout_s}s"
                if not policy.should_retry(e, attempt):
                    break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if not policy.should_retry(e, attempt):
                    break
            else:
                result = _score_result(task, prediction, chosen, (time.perf_counter() - t0) * 1000, attempt)
                if on_result:
                    on_result(result)
                return result

            delay = policy.delay_for(attempt, rng)
            if delay > 0:
                await asyncio.sleep(delay)

        result = _exhausted_result(task, last_error, (time.perf_counter() - t0) * 1000, attempt)
        if on_result:
            on_result(result)
        return result

    results = list(await asyncio.gather(*(run_one(t) for t in suite.tasks)))

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
    retry_policy: RetryPolicy | None = None,
    rate_limiter: RateLimiter | None = None,
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
            retry_policy=retry_policy, rate_limiter=rate_limiter,
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
