"""Self-check for an agenteval install.

Verifies the public API is intact and that the statistical routines still
match hand-computed closed forms. Run it after installing, or in CI:

    python scripts/verify.py

Exits non-zero on the first failure.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agenteval
from agenteval import *  # noqa: F403
from agenteval.graders.base import GraderRegistry

CHECKS: list[tuple[str, object]] = []
FAILURES: list[str] = []


def check(label: str):
    def wrap(fn):
        CHECKS.append((label, fn))
        return fn
    return wrap


@check("public API integrity")
def _api() -> str:
    missing = [n for n in agenteval.__all__ if not hasattr(agenteval, n)]
    assert not missing, f"exported but absent: {missing}"
    dupes = [n for n in set(agenteval.__all__) if agenteval.__all__.count(n) > 1]
    assert not dupes, f"duplicate exports: {dupes}"
    ordered = sorted(set(agenteval.__all__), key=lambda s: (s.lstrip("_").lower(), s))
    assert agenteval.__all__ == ordered, "__all__ is not sorted"
    return f"{len(agenteval.__all__)} exports, version {agenteval.__version__}"


@check("exception hierarchy")
def _exceptions() -> str:
    for exc in (SuiteError, SuiteFormatError, GraderError, UnknownGraderError,
                StatisticsError, PairedLengthError, ConfigurationError, BudgetExceededError):
        assert issubclass(exc, AgentEvalError), f"{exc.__name__} is not an AgentEvalError"
    assert BudgetExceeded is BudgetExceededError
    try:
        GraderRegistry.create("nope")
        raise AssertionError("unknown grader should raise")
    except UnknownGraderError:
        pass
    return "all errors derive from AgentEvalError"


@check("normal quantiles")
def _quantiles() -> str:
    from agenteval.stats import _z_for
    for level, expected in [(0.95, 1.959964), (0.99, 2.575829), (0.90, 1.644854)]:
        got = _z_for(level)
        assert abs(got - expected) < 1e-4, f"z({level}) = {got}, expected {expected}"
    return "z(0.95)=1.959964, z(0.99)=2.575829, z(0.90)=1.644854"


@check("Wilson interval matches closed form")
def _wilson() -> str:
    def closed_form(s: int, n: int, z: float = 1.959964) -> tuple[float, float]:
        p = s / n
        d = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / d
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return max(0.0, centre - spread), min(1.0, centre + spread)

    for s, n in [(8, 10), (0, 10), (10, 10), (50, 100), (1, 3)]:
        ci = wilson_interval(s, n)
        low, high = closed_form(s, n)
        assert abs(ci.low - low) < 1e-6 and abs(ci.high - high) < 1e-6, f"{s}/{n}: {ci}"
    for s in range(21):
        ci = wilson_interval(s, 20)
        assert 0.0 <= ci.low <= ci.high <= 1.0
    return "8/10 = [0.4902, 0.9433]; bounded in [0,1] across all counts"


@check("McNemar exact test")
def _mcnemar() -> str:
    r = mcnemar_test([True] * 10 + [False] * 10, [True] * 20)
    assert abs(r.p_value - 0.001953) < 1e-5, r.p_value
    assert r.significant
    balanced = mcnemar_test([True] * 5 + [False] * 5, [False] * 5 + [True] * 5)
    assert balanced.p_value == 1.0 and not balanced.significant
    return "10 discordant one-way = 2/2^10 = 0.001953"


@check("permutation test")
def _permutation() -> str:
    same = permutation_test([0.5] * 30, [0.5] * 30, iterations=2000, seed=1)
    assert same.p_value > 0.9, same.p_value
    apart = permutation_test([0.0] * 30, [1.0] * 30, iterations=2000, seed=1)
    assert apart.p_value < 0.01 and apart.significant
    return "identical samples not significant, disjoint samples significant"


@check("bootstrap and BCa")
def _bootstrap() -> str:
    values = [1.0] * 50 + [0.0] * 50
    a = bootstrap_interval(values, iterations=3000, seed=3)
    b = bootstrap_interval(values, iterations=3000, seed=3)
    assert (a.low, a.high) == (b.low, b.high), "not reproducible under a fixed seed"
    assert a.low < 0.5 < a.high
    analytic = wilson_interval(50, 100)
    assert abs(a.low - analytic.low) < 0.05 and abs(a.high - analytic.high) < 0.05
    bca = bca_interval(values, iterations=3000, seed=3)
    assert bca.low < 0.5 < bca.high
    return "reproducible, and agrees with the analytic Wilson interval"


@check("multiple-comparison corrections")
def _corrections() -> str:
    ps = [0.01, 0.02, 0.03, 0.04]
    holm = [round(t.adjusted_p, 4) for t in sorted(holm_bonferroni(ps).tests, key=lambda x: x.raw_p)]
    assert holm == [0.04, 0.06, 0.06, 0.06], holm
    bh = [round(t.adjusted_p, 4) for t in sorted(benjamini_hochberg(ps).tests, key=lambda x: x.raw_p)]
    assert bh == [0.04, 0.04, 0.04, 0.04], bh
    assert [round(t.adjusted_p, 4) for t in bonferroni([0.01, 0.02]).tests] == [0.02, 0.04]
    for method in ("holm", "bh", "bonferroni"):
        adjusted = [t.adjusted_p for t in sorted(adjust(ps, method).tests, key=lambda x: x.raw_p)]
        assert all(adjusted[i] <= adjusted[i + 1] + 1e-12 for i in range(len(adjusted) - 1))
        assert all(p <= 1.0 for p in adjusted)
    return "Holm [.04,.06,.06,.06], BH [.04]*4, all monotone and capped at 1"


@check("Cohen's kappa")
def _kappa() -> str:
    a = [True] * 20 + [True] * 5 + [False] * 10 + [False] * 15
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    r = cohens_kappa(a, b)
    assert abs(r.raw_agreement - 0.70) < 1e-9
    assert abs(r.expected_agreement - 0.50) < 1e-9
    assert abs(r.kappa - 0.40) < 1e-9
    assert cohens_kappa([True, False] * 10, [True, False] * 10).kappa == 1.0
    assert cohens_kappa([True] * 10 + [False] * 10, [False] * 10 + [True] * 10).kappa < 0
    return "observed .70, chance .50, kappa .40"


@check("calibration")
def _calibration() -> str:
    perfect = calibration([1.0] * 10 + [0.0] * 10, [True] * 10 + [False] * 10)
    assert perfect.ece == 0.0 and perfect.brier == 0.0
    over = calibration([0.9] * 20, [True] * 10 + [False] * 10)
    assert abs(over.ece - 0.4) < 1e-9, over.ece
    assert abs(over.brier - 0.41) < 1e-9, over.brier
    assert over.overconfident
    return "ECE 0.40 and Brier 0.41 for a 90%-confident 50%-accurate system"


@check("power analysis")
def _power() -> str:
    n = required_sample_size(0.5, 0.1)
    assert 380 <= n <= 420, n
    assert required_sample_size(0.5, 0.05) > 3 * n
    mde = minimum_detectable_effect(400, 0.5)
    assert 0.08 < mde < 0.12, mde
    return f"n={n} to detect 50%->60%; MDE at n=400 is {mde:.3f}"


@check("graders")
def _graders() -> str:
    def task(inp, exp):
        return Task(id="t", input=inp, expected=exp)

    assert ExactMatchGrader().grade("The Answer.", task("q", "the answer")).passed
    assert ContainsGrader().grade("alpha beta", task("q", ["alpha"])).passed
    assert RegexGrader(pattern=r"\d{3}").grade("abc 123", task("q", None)).passed
    assert F1TokenGrader(threshold=0.5).grade("the cat sat", task("q", "the cat sat on mat")).passed
    assert NumericGrader(tolerance=0.01).grade("3.145", task("q", 3.14)).passed
    assert not NumericGrader(tolerance=0.01).grade("3.20", task("q", 3.14)).passed
    assert RangeGrader(low=0, high=10).grade("7", task("q", None)).passed
    assert SetGrader().grade('["a","b"]', task("q", ["a", "b"])).passed
    assert StructuralGrader().grade('{"a":1}', task("q", {"a": 1})).passed
    assert EditDistanceGrader(threshold=0.5).grade("kitten", task("q", "sitting")).passed
    assert PredicateGrader(lambda p, t: len(str(p)) > 2).grade("abc", task("q", None)).passed
    assert CallableGrader(lambda p, t: 1.0).grade("x", task("q", None)).passed
    assert RubricGrader({"ok": lambda p, t: True}).grade("x", task("q", None)).passed
    assert LLMJudgeGrader(judge_fn=lambda p: "Score: 1").grade("x", task("q", "y")).passed
    assert WeightedGrader({"e": ExactMatchGrader()}).grade("a", task("q", "a")).passed
    assert len(GraderRegistry.available()) == 18
    return f"{len(GraderRegistry.available())} graders registered and scoring"


@check("end-to-end evaluation")
def _end_to_end() -> str:
    suite = suite_from_records("math", [
        {"id": f"q{i}", "input": f"{i}+{i}", "expected": str(i * 2), "tags": ["even" if i % 2 == 0 else "odd"]}
        for i in range(1, 31)
    ])
    assert validate_suite(suite) == []

    def good(task):
        a, b = task.input.split("+")
        return str(int(a) + int(b))

    runs = evaluate_many({"good": good, "bad": lambda t: "0"}, suite, ExactMatchGrader(), seed=1)
    assert runs["good"].pass_rate == 1.0 and runs["bad"].pass_rate == 0.0

    improvement = compare(runs["bad"], runs["good"], iterations=2000, seed=1)
    assert improvement.is_improvement and len(improvement.fixed) == 30
    assert not gate(runs["bad"], min_pass_rate=0.9).passed
    assert not regression_gate(compare(runs["good"], runs["bad"], iterations=2000, seed=1)).passed
    assert compare_all(runs, method="holm", iterations=1000, seed=1).correction is not None
    return "30-task suite: compare, gates, and correction all behave"


@check("scale features")
def _scale() -> str:
    suite = suite_from_records("s", [
        {"id": f"q{i}", "input": f"{i}+{i}", "expected": str(i * 2)} for i in range(1, 21)
    ])

    def system(task):
        a, b = task.input.split("+")
        return str(int(a) + int(b))

    streamed = list(iter_evaluate(system, suite, ExactMatchGrader()))
    assert len(streamed) == 20 and all(r.is_pass for r in streamed)

    calls = [0]

    def counted(task):
        calls[0] += 1
        return system(task)

    cache = PredictionCache()
    evaluate(cache.wrap(counted, "s"), suite, ExactMatchGrader())
    evaluate(cache.wrap(counted, "s"), suite, ExactMatchGrader())
    assert calls[0] == 20, f"cache did not prevent re-execution: {calls[0]}"

    with tempfile.TemporaryDirectory() as tmp:
        store = str(Path(tmp) / "run.jsonl")
        run, reused = evaluate_resumable(system, suite, ExactMatchGrader(), store)
        assert len(run) == 20 and reused == 0
        _, reused_again = evaluate_resumable(system, suite, ExactMatchGrader(), store)
        assert reused_again == 20, "resume did not reuse stored results"

    decision = evaluate_sequential(system, suite, ExactMatchGrader(), threshold=0.5, min_samples=10)
    assert decision.decision.stop
    return "streaming, caching, resume, and early stopping all work"


@check("agent trajectories")
def _trajectories() -> str:
    suite = suite_from_records("agents", [
        {"id": "t1", "input": "weather in Lahore", "expected": "31C"},
    ])

    def agent(task):
        return Trajectory(
            steps=[
                Step(action="search", args={"q": task.input}, observation="wttr.in", cost=0.001),
                Step(action="fetch", args={"url": "wttr.in"}, observation="31C", cost=0.002),
            ],
            output="31C",
        )

    grader = WeightedGrader({
        "answer": OutcomeGrader(ExactMatchGrader()),
        "tools": ToolSequenceGrader(["search", "fetch"], mode="exact"),
        "budget": StepBudgetGrader(max_steps=4, max_cost=0.01, allow_errors=False),
    }, threshold=0.99)

    run = evaluate(agent, suite, grader)
    assert run.pass_rate == 1.0, run.results[0].score

    wasteful = Trajectory(steps=[Step(action="search") for _ in range(6)], output="31C")
    over = StepBudgetGrader(max_steps=4).grade(wasteful, suite.tasks[0])
    assert not over.passed and 0.0 < over.value < 1.0, over
    return "trajectory graders score outcome, tool use, and step budget together"


@check("stability, contamination, cost")
def _analysis() -> str:
    suite = suite_from_records("s", [
        {"id": f"q{i}", "input": f"{i}+{i}", "expected": str(i * 2)} for i in range(1, 21)
    ])

    def system(task):
        a, b = task.input.split("+")
        return str(int(a) + int(b))

    stability = analyze_stability(repeat_evaluate(system, suite, ExactMatchGrader(), repeats=3))
    assert stability.reliable and stability.spread == 0.0 and not stability.flaky

    leaked = suite_from_records("c", [{"id": "x", "input": "alpha beta gamma delta epsilon zeta eta theta"}])
    report = detect_contamination(leaked, ["alpha beta gamma delta epsilon zeta eta theta"],
                                  n_gram=8, threshold=0.5)
    assert not report.clean and "x" in report.contaminated_ids

    tracker = CostTracker(budget=1.0)
    evaluate(tracker.wrap(system, fixed_cost=0.001), suite, ExactMatchGrader())
    assert abs(tracker.report.total - 0.02) < 1e-9, tracker.report.total
    return "deterministic run is stable, leak detected, cost tracked exactly"


@check("reporting")
def _reporting() -> str:
    import json

    suite = suite_from_records("s", [
        {"id": f"q{i}", "input": f"{i}+{i}", "expected": str(i * 2), "tags": ["t"]} for i in range(1, 11)
    ])
    runs = evaluate_many({"a": lambda t: "0", "b": lambda t: "0"}, suite, ExactMatchGrader())
    run = runs["a"]

    assert "pass rate" in run_to_text(run)
    assert "| metric |" in run_to_markdown(run)
    json.loads(run_to_json(run))
    assert "95% CI" in tag_breakdown(run)
    assert "Tier 1" in leaderboard_tiers(runs)
    html = run_to_html(run)
    assert '<section class="ae">' in html and "prefers-color-scheme" in html
    assert "http://" not in html and "https://" not in html, "HTML must not reference external assets"
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "r.html")
        write_html(html, out)
        assert Path(out).read_text(encoding="utf-8").startswith("<!doctype html>")
    return "text, markdown, JSON, and self-contained HTML all render"


def main() -> int:
    print(f"agenteval {agenteval.__version__} - self check")
    print(f"python {sys.version.split()[0]} on {sys.platform}\n")

    width = max(len(label) for label, _ in CHECKS) + 2
    for label, fn in CHECKS:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(label)
            print(f"  FAIL  {label.ljust(width)} {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {label.ljust(width)} {detail}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {len(CHECKS)} checks FAILED: {', '.join(FAILURES)}")
        return 1
    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
