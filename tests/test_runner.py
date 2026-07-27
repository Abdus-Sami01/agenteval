from __future__ import annotations

import asyncio
import json
import random
import time

import pytest

from agenteval import (
    ConfigurationError,
    ExactMatchGrader,
    Outcome,
    OutcomeGrader,
    RateLimiter,
    RetryPolicy,
    Score,
    Step,
    StepBudgetGrader,
    TaskSuite,
    Trajectory,
    WeightedGrader,
    evaluate,
    evaluate_async,
    evaluate_many,
    iter_evaluate,
    repeat_evaluate,
    run_to_json,
    suite_from_records,
)
from tests.helpers import adder, always_wrong, raises


class TestEvaluate:
    def test_perfect_system(self, math_suite):
        run = evaluate(adder, math_suite, ExactMatchGrader())
        assert run.pass_rate == 1.0
        assert run.passed == 30 and run.failed == 0 and run.errored == 0

    def test_failing_system(self, math_suite):
        run = evaluate(always_wrong, math_suite, ExactMatchGrader())
        assert run.pass_rate == 0.0
        assert run.failed == 30

    def test_raising_system_is_recorded_as_error(self, small_suite):
        run = evaluate(raises, small_suite, ExactMatchGrader())
        assert run.errored == 3
        assert all(r.outcome == Outcome.ERROR for r in run.results)
        assert "model offline" in run.results[0].error

    def test_errors_excluded_from_pass_rate(self, small_suite):
        run = evaluate(raises, small_suite, ExactMatchGrader())
        assert run.pass_rate == 0.0
        assert run.scored == []

    def test_results_preserve_suite_order(self, math_suite):
        run = evaluate(adder, math_suite, ExactMatchGrader())
        assert [r.task_id for r in run.results] == [t.id for t in math_suite.tasks]

    def test_tags_propagate_to_results(self, math_suite):
        run = evaluate(adder, math_suite, ExactMatchGrader())
        assert all(r.tags for r in run.results)
        assert set(run.by_tag()) == {"even", "odd"}

    def test_metadata_recorded(self, small_suite):
        run = evaluate(adder, small_suite, ExactMatchGrader(), seed=42, system_name="mysystem")
        assert run.metadata.seed == 42
        assert run.metadata.system_name == "mysystem"
        assert run.metadata.suite_name == "small"
        assert run.metadata.run_id
        assert run.metadata.python_version

    def test_system_name_defaults_to_function_name(self, small_suite):
        assert evaluate(adder, small_suite, ExactMatchGrader()).metadata.system_name == "adder"

    def test_on_result_callback(self, small_suite):
        seen = []
        evaluate(adder, small_suite, ExactMatchGrader(), on_result=seen.append)
        assert len(seen) == 3

    def test_empty_suite(self):
        run = evaluate(adder, TaskSuite(name="empty"), ExactMatchGrader())
        assert len(run) == 0 and run.pass_rate == 0.0

    def test_callable_grader_accepted(self, small_suite):
        run = evaluate(adder, small_suite, lambda p, t: Score(value=1.0, passed=True))
        assert run.pass_rate == 1.0

    def test_grader_exception_becomes_error(self, small_suite):
        def bad_grader(prediction, task):
            raise ValueError("grader broke")

        run = evaluate(adder, small_suite, bad_grader)
        assert run.errored == 3
        assert "grader failed" in run.results[0].error

    def test_per_task_grader_selection(self, small_suite):
        def choose(task):
            return ExactMatchGrader() if task.id == "a" else ExactMatchGrader(normalize=False)

        run = evaluate(adder, small_suite, ExactMatchGrader(), grader_for=choose)
        assert run.pass_rate == 1.0


class TestRetriesAndTimeouts:
    def test_retries_until_success(self, small_suite):
        attempts = {"n": 0}

        def flaky(task):
            attempts["n"] += 1
            if attempts["n"] % 2 == 1:
                raise RuntimeError("transient")
            return adder(task)

        run = evaluate(flaky, small_suite, ExactMatchGrader(), retries=2)
        assert run.passed == 3

    def test_retry_count_recorded(self, small_suite):
        calls = {"n": 0}

        def once_failing(task):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first call fails")
            return adder(task)

        run = evaluate(once_failing, small_suite, ExactMatchGrader(), retries=1)
        assert run.results[0].attempts == 2

    def test_exhausted_retries_reported(self, small_suite):
        run = evaluate(raises, small_suite, ExactMatchGrader(), retries=2)
        assert run.errored == 3
        assert run.results[0].attempts == 3

    def test_timeout_marks_task(self, small_suite):
        def slow(task):
            time.sleep(0.3)
            return "x"

        run = evaluate(slow, small_suite, ExactMatchGrader(), timeout_s=0.05)
        assert all(r.outcome == Outcome.TIMEOUT for r in run.results)
        assert "timed out" in run.results[0].error


class TestParallelism:
    def test_parallel_matches_sequential(self, math_suite):
        sequential = evaluate(adder, math_suite, ExactMatchGrader(), max_parallel=1)
        parallel = evaluate(adder, math_suite, ExactMatchGrader(), max_parallel=4)
        assert sequential.pass_rate == parallel.pass_rate
        assert [r.task_id for r in sequential.results] == [r.task_id for r in parallel.results]

    def test_parallel_preserves_order(self, math_suite):
        run = evaluate(adder, math_suite, ExactMatchGrader(), max_parallel=8)
        assert [r.task_id for r in run.results] == [t.id for t in math_suite.tasks]

    def test_parallel_speeds_up_blocking_work(self):
        suite = suite_from_records("io", [
            {"id": f"t{i}", "input": "x", "expected": "x"} for i in range(8)
        ])

        def blocking(task):
            time.sleep(0.05)
            return "x"

        start = time.perf_counter()
        evaluate(blocking, suite, ExactMatchGrader(), max_parallel=8)
        parallel_s = time.perf_counter() - start

        assert parallel_s < 0.30, f"expected overlap, took {parallel_s:.2f}s"


class TestEvaluateMany:
    def test_runs_each_system(self, math_suite):
        runs = evaluate_many({"good": adder, "bad": always_wrong}, math_suite, ExactMatchGrader())
        assert set(runs) == {"good", "bad"}
        assert runs["good"].pass_rate == 1.0
        assert runs["bad"].pass_rate == 0.0

    def test_system_names_are_labelled(self, small_suite):
        runs = evaluate_many({"alpha": adder}, small_suite, ExactMatchGrader())
        assert runs["alpha"].metadata.system_name == "alpha"


class TestRepeatEvaluate:
    def test_produces_requested_number_of_runs(self, small_suite):
        runs = repeat_evaluate(adder, small_suite, ExactMatchGrader(), repeats=4)
        assert len(runs) == 4

    def test_seeds_differ_across_repeats(self, small_suite):
        runs = repeat_evaluate(adder, small_suite, ExactMatchGrader(), repeats=3, base_seed=10)
        assert [r.metadata.seed for r in runs] == [10, 11, 12]


class TestIterEvaluate:
    def test_yields_one_result_per_task(self, math_suite):
        results = list(iter_evaluate(adder, math_suite, ExactMatchGrader()))
        assert len(results) == 30
        assert all(r.is_pass for r in results)

    def test_is_lazy(self, math_suite):
        stream = iter_evaluate(adder, math_suite, ExactMatchGrader())
        first = next(stream)
        assert first.task_id == "q1"

    def test_matches_batch_evaluation(self, math_suite):
        streamed = [r.task_id for r in iter_evaluate(adder, math_suite, ExactMatchGrader())]
        batch = [r.task_id for r in evaluate(adder, math_suite, ExactMatchGrader()).results]
        assert streamed == batch


class TestEvalRunAggregates:
    def test_counts_and_rates(self, math_suite):
        mixed = suite_from_records("mixed", [
            {"id": "pass", "input": "1+1", "expected": "2"},
            {"id": "fail", "input": "1+1", "expected": "99"},
        ])
        run = evaluate(adder, mixed, ExactMatchGrader())
        assert run.passed == 1 and run.failed == 1
        assert run.pass_rate == 0.5

    def test_weighted_mean_score(self):
        suite = suite_from_records("w", [
            {"id": "heavy", "input": "1+1", "expected": "2", "weight": 9.0},
            {"id": "light", "input": "1+1", "expected": "99", "weight": 1.0},
        ])
        run = evaluate(adder, suite, ExactMatchGrader())
        assert run.mean_score == pytest.approx(0.9)

    def test_result_lookup(self, small_suite):
        run = evaluate(adder, small_suite, ExactMatchGrader())
        assert run.result_for("a") is not None
        assert run.result_for("missing") is None

    def test_scores_list(self, small_suite):
        assert evaluate(adder, small_suite, ExactMatchGrader()).scores() == [1.0, 1.0, 1.0]


class TestRetryPolicy:
    def test_delay_grows_geometrically(self):
        policy = RetryPolicy(base_delay_s=1.0, multiplier=2.0, jitter=0.0)
        assert [policy.delay_for(i) for i in (1, 2, 3)] == [1.0, 2.0, 4.0]

    def test_delay_is_capped(self):
        policy = RetryPolicy(base_delay_s=1.0, multiplier=10.0, max_delay_s=5.0, jitter=0.0)
        assert policy.delay_for(9) == 5.0

    def test_jitter_stays_in_band_and_under_cap(self):
        policy = RetryPolicy(base_delay_s=2.0, multiplier=1.0, max_delay_s=2.5, jitter=0.5)
        rng = random.Random(0)
        draws = [policy.delay_for(1, rng) for _ in range(200)]
        assert all(1.0 <= d <= 2.5 for d in draws)
        assert len(set(draws)) > 1

    def test_none_policy_does_not_retry(self):
        assert RetryPolicy.none().should_retry(RuntimeError(), attempt=1) is False

    def test_rejects_bad_configuration(self):
        with pytest.raises(ConfigurationError):
            RetryPolicy(max_attempts=0)
        with pytest.raises(ConfigurationError):
            RetryPolicy(jitter=2.0)
        with pytest.raises(ConfigurationError):
            RetryPolicy(multiplier=0.5)
        with pytest.raises(ConfigurationError):
            RetryPolicy(base_delay_s=-1)

    def test_retry_on_filters_exception_types(self, small_suite):
        calls = {"n": 0}

        def broken(task):
            calls["n"] += 1
            raise ValueError("not worth retrying")

        policy = RetryPolicy(
            max_attempts=4, base_delay_s=0.0, jitter=0.0,
            retry_on=lambda exc: isinstance(exc, TimeoutError),
        )
        run = evaluate(broken, small_suite, ExactMatchGrader(), retry_policy=policy)
        assert calls["n"] == 3
        assert run.results[0].attempts == 1

    def test_policy_backoff_is_actually_slept(self, small_suite):
        policy = RetryPolicy(max_attempts=3, base_delay_s=0.05, multiplier=1.0, jitter=0.0)
        start = time.perf_counter()
        run = evaluate(raises, small_suite, ExactMatchGrader(), retry_policy=policy)
        assert run.errored == 3
        assert time.perf_counter() - start >= 0.05 * 2 * 3

    def test_policy_overrides_retries_argument(self, small_suite):
        run = evaluate(
            raises, small_suite, ExactMatchGrader(),
            retries=0, retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, jitter=0.0),
        )
        assert run.results[0].attempts == 2

    def test_negative_retries_rejected(self, small_suite):
        with pytest.raises(ConfigurationError):
            evaluate(adder, small_suite, ExactMatchGrader(), retries=-1)


class TestRateLimiter:
    def test_burst_then_paced(self):
        limiter = RateLimiter(rate_per_s=50.0, burst=2)
        start = time.perf_counter()
        for _ in range(6):
            limiter.acquire()
        assert time.perf_counter() - start >= 4 / 50.0

    def test_shared_across_workers(self):
        suite = suite_from_records("rl", [
            {"id": f"t{i}", "input": "x", "expected": "x"} for i in range(8)
        ])
        limiter = RateLimiter(rate_per_s=100.0, burst=1)
        start = time.perf_counter()
        evaluate(lambda task: "x", suite, ExactMatchGrader(), max_parallel=8, rate_limiter=limiter)
        assert time.perf_counter() - start >= 7 / 100.0

    def test_rejects_bad_configuration(self):
        with pytest.raises(ConfigurationError):
            RateLimiter(rate_per_s=0)
        with pytest.raises(ConfigurationError):
            RateLimiter(rate_per_s=1.0, burst=0)


class TestParallelStreaming:
    def test_callbacks_arrive_before_the_run_ends(self):
        suite = suite_from_records("stream", [
            {"id": f"t{i}", "input": "x", "expected": "x"} for i in range(8)
        ])
        seen: list[float] = []

        def slow(task):
            time.sleep(0.05)
            return "x"

        start = time.perf_counter()
        evaluate(slow, suite, ExactMatchGrader(), max_parallel=2,
                 on_result=lambda r: seen.append(time.perf_counter() - start))
        assert len(seen) == 8
        assert seen[0] < seen[-1]

    def test_max_parallel_must_be_positive(self, small_suite):
        with pytest.raises(ConfigurationError):
            evaluate(adder, small_suite, ExactMatchGrader(), max_parallel=0)


class TestEvaluateAsync:
    def test_scores_a_coroutine_system(self, math_suite):
        async def system(task):
            return adder(task)

        run = asyncio.run(evaluate_async(system, math_suite, ExactMatchGrader()))
        assert run.pass_rate == 1.0
        assert [r.task_id for r in run.results] == [t.id for t in math_suite.tasks]

    def test_concurrency_overlaps_awaits(self):
        suite = suite_from_records("io", [
            {"id": f"t{i}", "input": "x", "expected": "x"} for i in range(16)
        ])

        async def system(task):
            await asyncio.sleep(0.05)
            return "x"

        start = time.perf_counter()
        asyncio.run(evaluate_async(system, suite, ExactMatchGrader(), max_parallel=16))
        assert time.perf_counter() - start < 0.30

    def test_semaphore_bounds_in_flight_calls(self):
        suite = suite_from_records("io", [
            {"id": f"t{i}", "input": "x", "expected": "x"} for i in range(12)
        ])
        state = {"now": 0, "peak": 0}

        async def system(task):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.01)
            state["now"] -= 1
            return "x"

        asyncio.run(evaluate_async(system, suite, ExactMatchGrader(), max_parallel=3))
        assert state["peak"] <= 3

    def test_timeout_marks_task(self, small_suite):
        async def slow(task):
            await asyncio.sleep(0.3)
            return "x"

        run = asyncio.run(evaluate_async(slow, small_suite, ExactMatchGrader(), timeout_s=0.02))
        assert all(r.outcome == Outcome.TIMEOUT for r in run.results)

    def test_retries_transient_failures(self, small_suite):
        calls = {"n": 0}

        async def flaky_system(task):
            calls["n"] += 1
            if calls["n"] % 2 == 1:
                raise RuntimeError("transient")
            return adder(task)

        policy = RetryPolicy(max_attempts=2, base_delay_s=0.0, jitter=0.0)
        run = asyncio.run(
            evaluate_async(flaky_system, small_suite, ExactMatchGrader(), retry_policy=policy)
        )
        assert run.passed == 3

    def test_errors_are_recorded(self, small_suite):
        async def broken(task):
            raise RuntimeError("model offline")

        run = asyncio.run(evaluate_async(broken, small_suite, ExactMatchGrader()))
        assert run.errored == 3
        assert "model offline" in run.results[0].error

    def test_sync_system_is_rejected(self, small_suite):
        run = asyncio.run(evaluate_async(adder, small_suite, ExactMatchGrader()))
        assert run.errored == 3
        assert "awaitable" in run.results[0].error

    def test_on_result_called_per_task(self, small_suite):
        async def system(task):
            return adder(task)

        seen = []
        asyncio.run(evaluate_async(system, small_suite, ExactMatchGrader(), on_result=seen.append))
        assert len(seen) == 3

    def test_rate_limiter_paces_async_calls(self):
        suite = suite_from_records("rl", [
            {"id": f"t{i}", "input": "x", "expected": "x"} for i in range(6)
        ])

        async def system(task):
            return "x"

        limiter = RateLimiter(rate_per_s=100.0, burst=1)
        start = time.perf_counter()
        asyncio.run(evaluate_async(system, suite, ExactMatchGrader(), rate_limiter=limiter))
        assert time.perf_counter() - start >= 5 / 100.0


class TestTrajectorySystems:
    def agent(self, task):
        a, b = str(task.input).split("+")
        return Trajectory(
            steps=[Step(action="parse"), Step(action="add", cost=0.01)],
            output=str(int(a) + int(b)),
        )

    def test_evaluates_an_agent_end_to_end(self, math_suite):
        run = evaluate(self.agent, math_suite, OutcomeGrader(ExactMatchGrader()))
        assert run.pass_rate == 1.0

    def test_trajectory_survives_into_the_result(self, small_suite):
        run = evaluate(self.agent, small_suite, OutcomeGrader(ExactMatchGrader()))
        assert isinstance(run.results[0].prediction, Trajectory)
        assert run.results[0].prediction.actions == ["parse", "add"]

    def test_json_report_keeps_the_steps(self, small_suite):
        run = evaluate(self.agent, small_suite, OutcomeGrader(ExactMatchGrader()))
        payload = json.loads(run_to_json(run))
        prediction = payload["results"][0]["prediction"]
        assert [s["action"] for s in prediction["steps"]] == ["parse", "add"]
        assert prediction["output"] == "2"

    def test_process_failure_fails_the_task(self, small_suite):
        grader = WeightedGrader({
            "answer": OutcomeGrader(ExactMatchGrader()),
            "budget": StepBudgetGrader(max_steps=1),
        }, threshold=0.99)
        run = evaluate(self.agent, small_suite, grader)
        assert run.pass_rate == 0.0
        assert run.results[0].score.subscores["answer"] == 1.0

    def test_async_agent_is_supported(self, small_suite):
        async def async_agent(task):
            return self.agent(task)

        run = asyncio.run(evaluate_async(async_agent, small_suite, OutcomeGrader(ExactMatchGrader())))
        assert run.pass_rate == 1.0
