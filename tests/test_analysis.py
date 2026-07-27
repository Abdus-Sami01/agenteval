"""Comparison, gates, corrections, agreement, calibration, stability, contamination, cost."""

from __future__ import annotations

import pytest

from agenteval import (
    BudgetExceededError,
    ConfigurationError,
    CorpusIndex,
    CostTracker,
    ExactMatchGrader,
    PairedLengthError,
    Task,
    adjust,
    analyze_stability,
    benjamini_hochberg,
    bonferroni,
    brier_score,
    brier_skill_score,
    calibration,
    clean_suite,
    cohens_kappa,
    compare,
    compare_all,
    compare_by_tag,
    cost_efficiency,
    detect_contamination,
    estimate_tokens,
    evaluate,
    evaluate_many,
    find_duplicates,
    gate,
    holm_bonferroni,
    intraclass_correlation,
    krippendorff_alpha,
    log_loss,
    percent_agreement,
    regression_gate,
    reliability_diagram_text,
    repeat_evaluate,
    required_repeats,
    suite_from_records,
    tag_regression_gate,
    token_cost,
    validate_judge,
)
from tests.helpers import adder, always_wrong, flaky


class TestCompare:
    def test_clear_improvement(self, failing_run, perfect_run):
        result = compare(failing_run, perfect_run, iterations=2000, seed=1)
        assert result.is_improvement
        assert result.verdict() == "IMPROVEMENT"
        assert len(result.fixed) == 30 and not result.broken

    def test_clear_regression(self, perfect_run, failing_run):
        result = compare(perfect_run, failing_run, iterations=2000, seed=1)
        assert result.is_regression
        assert result.verdict() == "REGRESSION"
        assert len(result.broken) == 30

    def test_identical_runs_are_inconclusive(self, perfect_run):
        result = compare(perfect_run, perfect_run, iterations=2000, seed=1)
        assert result.inconclusive
        assert result.verdict() == "INCONCLUSIVE"

    def test_small_difference_is_inconclusive(self):
        suite = suite_from_records("tiny", [
            {"id": f"q{i}", "input": f"{i}+{i}", "expected": str(i * 2)} for i in range(1, 6)
        ])
        base = evaluate(adder, suite, ExactMatchGrader(), system_name="base")

        def one_wrong(task):
            return "0" if task.id == "q1" else adder(task)

        cand = evaluate(one_wrong, suite, ExactMatchGrader(), system_name="cand")
        result = compare(base, cand, iterations=3000, seed=1)
        assert result.inconclusive, "5 tasks is not enough evidence for a verdict"
        assert result.broken == ["q1"]

    def test_no_shared_tasks(self, perfect_run):
        other_suite = suite_from_records("other", [{"id": "zz", "input": "1+1", "expected": "2"}])
        other = evaluate(adder, other_suite, ExactMatchGrader())
        result = compare(perfect_run, other, iterations=500, seed=1)
        assert result.paired_ids == []

    def test_summary_mentions_verdict(self, failing_run, perfect_run):
        summary = compare(failing_run, perfect_run, iterations=1000, seed=1).summary()
        assert "IMPROVEMENT" in summary


class TestGates:
    def test_min_pass_rate(self, perfect_run, failing_run):
        assert gate(perfect_run, min_pass_rate=0.9).passed
        assert not gate(failing_run, min_pass_rate=0.9).passed

    def test_max_error_rate(self, perfect_run):
        assert gate(perfect_run, max_error_rate=0.0).passed

    def test_per_tag_gate(self, perfect_run):
        assert gate(perfect_run, min_tag_pass_rate={"even": 0.9}).passed

    def test_ci_lower_bound_gate_is_stricter_than_point_estimate(self, perfect_run):
        assert gate(perfect_run, min_pass_rate=0.95).passed
        assert not gate(perfect_run, require_ci_above=0.95).passed

    def test_failures_are_listed(self, failing_run):
        report = gate(failing_run, min_pass_rate=0.9)
        assert len(report.failures) == 1
        assert "FAIL" in report.summary()

    def test_no_gates_configured(self, perfect_run):
        assert "No gates" in gate(perfect_run).summary()

    def test_regression_gate_blocks_broken_tasks(self, perfect_run, failing_run):
        result = compare(perfect_run, failing_run, iterations=1000, seed=1)
        assert not regression_gate(result, max_broken=0).passed

    def test_regression_gate_allows_tolerance(self, perfect_run, failing_run):
        result = compare(perfect_run, failing_run, iterations=1000, seed=1)
        assert not regression_gate(result, max_broken=100).passed  # score drop still fails


class TestMultipleComparison:
    def test_holm_matches_textbook(self):
        report = holm_bonferroni([0.01, 0.02, 0.03, 0.04], alpha=0.05)
        adjusted = [round(t.adjusted_p, 4) for t in sorted(report.tests, key=lambda x: x.raw_p)]
        assert adjusted == [0.04, 0.06, 0.06, 0.06]
        assert len(report.significant) == 1

    def test_bh_matches_textbook(self):
        report = benjamini_hochberg([0.01, 0.02, 0.03, 0.04], alpha=0.05)
        adjusted = [round(t.adjusted_p, 4) for t in sorted(report.tests, key=lambda x: x.raw_p)]
        assert adjusted == [0.04, 0.04, 0.04, 0.04]
        assert len(report.significant) == 4

    def test_bonferroni_multiplies(self):
        assert [round(t.adjusted_p, 4) for t in bonferroni([0.01, 0.02]).tests] == [0.02, 0.04]

    def test_bh_is_less_conservative_than_holm(self):
        ps = [0.01, 0.02, 0.03, 0.04]
        assert len(benjamini_hochberg(ps).significant) >= len(holm_bonferroni(ps).significant)

    @pytest.mark.parametrize("method", ["holm", "bh", "fdr", "bonferroni"])
    def test_adjusted_values_are_monotone_and_capped(self, method):
        import random

        ps = sorted(random.Random(1).random() for _ in range(20))
        report = adjust(ps, method)
        adjusted = [t.adjusted_p for t in sorted(report.tests, key=lambda x: x.raw_p)]
        assert all(adjusted[i] <= adjusted[i + 1] + 1e-12 for i in range(len(adjusted) - 1))
        assert all(p <= 1.0 for p in adjusted)

    def test_empty_input(self):
        assert holm_bonferroni([]).tests == []

    def test_unknown_method_raises(self):
        with pytest.raises(ConfigurationError):
            adjust([0.1], method="nonsense")

    def test_compare_all_pairwise(self, math_suite):
        runs = evaluate_many({"a": adder, "b": always_wrong}, math_suite, ExactMatchGrader())
        matrix = compare_all(runs, iterations=1000, seed=1)
        assert len(matrix.comparisons) == 1
        assert matrix.correction is not None

    def test_compare_all_with_baseline(self, math_suite):
        runs = evaluate_many(
            {"base": adder, "x": always_wrong, "y": always_wrong}, math_suite, ExactMatchGrader()
        )
        matrix = compare_all(runs, baseline="base", iterations=500, seed=1)
        assert len(matrix.comparisons) == 2
        assert all(pair[0] == "base" for pair in matrix.comparisons)

    def test_compare_all_unknown_baseline(self, math_suite):
        runs = evaluate_many({"a": adder}, math_suite, ExactMatchGrader())
        with pytest.raises(ConfigurationError):
            compare_all(runs, baseline="missing")


class TestAgreement:
    def test_kappa_matches_hand_computation(self):
        a = [True] * 20 + [True] * 5 + [False] * 10 + [False] * 15
        b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
        report = cohens_kappa(a, b)
        assert report.raw_agreement == pytest.approx(0.70)
        assert report.expected_agreement == pytest.approx(0.50)
        assert report.kappa == pytest.approx(0.40)
        assert report.interpretation == "fair"

    def test_perfect_agreement(self):
        assert cohens_kappa([True, False] * 10, [True, False] * 10).kappa == 1.0

    def test_total_disagreement_is_below_chance(self):
        report = cohens_kappa([True] * 10 + [False] * 10, [False] * 10 + [True] * 10)
        assert report.kappa < 0
        assert report.interpretation == "worse than chance"

    def test_empty_input(self):
        assert cohens_kappa([], []).n == 0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(PairedLengthError):
            cohens_kappa([True], [True, False])

    def test_trustworthy_threshold(self):
        strong = cohens_kappa([True] * 10 + [False] * 10, [True] * 10 + [False] * 10)
        assert strong.trustworthy

    def test_percent_agreement(self):
        assert percent_agreement(["a", "b"], ["a", "b"]) == 1.0
        assert percent_agreement(["a", "b"], ["a", "c"]) == 0.5
        assert percent_agreement([], []) == 0.0

    def test_krippendorff_alpha(self):
        assert krippendorff_alpha([[1, 1, 1], [1, 1, 1]]) == pytest.approx(1.0)

    def test_lenient_judge_is_caught_despite_high_raw_agreement(self):
        tasks = [Task(id=f"t{i}", input="q", expected="a") for i in range(30)]
        labels = [i % 3 != 0 for i in range(30)]
        labeled = [(tasks[i], "pred", labels[i]) for i in range(30)]

        validation = validate_judge(lambda p, t: True, labeled)
        assert validation.report.raw_agreement > 0.5, "raw agreement looks fine"
        assert validation.report.kappa == pytest.approx(0.0, abs=1e-9), "but kappa exposes it"
        assert not validation.report.trustworthy
        assert validation.judge_is_lenient

    def test_good_judge_scores_well(self):
        tasks = [Task(id=f"t{i}", input="q", expected="a") for i in range(30)]
        labels = [i % 3 != 0 for i in range(30)]
        labeled = [(tasks[i], "pred", labels[i]) for i in range(30)]

        validation = validate_judge(lambda p, t: int(t.id[1:]) % 3 != 0, labeled)
        assert validation.report.kappa == 1.0
        assert not validation.disagreements

    def test_judge_exception_counts_as_reject(self):
        tasks = [Task(id="t", input="q", expected="a")]
        validation = validate_judge(lambda p, t: 1 / 0, [(tasks[0], "p", True)])
        assert validation.disagreements


class TestCalibration:
    def test_perfect_calibration(self):
        report = calibration([1.0] * 10 + [0.0] * 10, [True] * 10 + [False] * 10)
        assert report.ece == 0.0
        assert report.brier == 0.0
        assert report.well_calibrated

    def test_overconfidence_detected(self):
        report = calibration([0.9] * 20, [True] * 10 + [False] * 10)
        assert report.ece == pytest.approx(0.4)
        assert report.brier == pytest.approx(0.41)
        assert report.overconfident
        assert not report.underconfident

    def test_underconfidence_detected(self):
        report = calibration([0.1] * 20, [True] * 18 + [False] * 2)
        assert report.underconfident

    def test_confidences_are_clipped(self):
        report = calibration([1.5, -0.5], [True, False])
        assert report.mean_confidence == pytest.approx(0.5)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(PairedLengthError):
            calibration([0.5], [True, False])

    def test_empty_input(self):
        assert calibration([], []).n == 0

    def test_brier_score_bounds(self):
        assert brier_score([1.0, 0.0], [True, False]) == 0.0
        assert brier_score([0.0, 1.0], [True, False]) == 1.0

    def test_brier_skill_score(self):
        assert brier_skill_score([1.0] * 10, [True] * 10) == 0.0  # no variance in reference
        assert brier_skill_score([0.9] * 10 + [0.1] * 10, [True] * 10 + [False] * 10) > 0.9

    def test_log_loss(self):
        assert log_loss([1.0], [True]) == pytest.approx(0.0, abs=1e-9)
        assert log_loss([0.0], [True]) > 30  # epsilon-clamped, very large

    def test_reliability_diagram_renders(self):
        report = calibration([0.9] * 20, [True] * 10 + [False] * 10)
        assert "Reliability diagram" in reliability_diagram_text(report)


class TestStability:
    def test_deterministic_system_is_stable(self, math_suite):
        runs = repeat_evaluate(adder, math_suite, ExactMatchGrader(), repeats=3)
        report = analyze_stability(runs)
        assert report.spread == 0.0
        assert not report.flaky
        assert report.reliable
        assert len(report.always_pass) == 30

    def test_flaky_tasks_detected(self, math_suite):
        import random

        random.seed(1)

        def unstable(task):
            index = int(str(task.input).split("+")[0])
            if index <= 10:
                return adder(task)
            if index <= 20:
                return "0"
            return adder(task) if random.random() < 0.5 else "0"

        report = analyze_stability(repeat_evaluate(unstable, math_suite, ExactMatchGrader(), repeats=6))
        assert len(report.always_pass) == 10
        assert report.flaky, "coin-flip tasks should be detected"
        assert not report.reliable

    def test_variance_aware_interval_is_never_narrower(self, math_suite):
        import random

        random.seed(2)
        runs = repeat_evaluate(flaky(0.6, "s"), math_suite, ExactMatchGrader(), repeats=4)
        report = analyze_stability(runs)
        assert report.combined_interval().width >= report.naive_interval().width

    def test_entropy_peaks_at_even_split(self):
        from agenteval import TaskStability

        assert TaskStability("t", runs=4, passes=2).entropy == pytest.approx(1.0)
        assert TaskStability("t", runs=4, passes=0).entropy == 0.0
        assert not TaskStability("t", runs=4, passes=4).is_flaky

    def test_icc_needs_multiple_runs(self, math_suite):
        assert intraclass_correlation([]) == 0.0
        single = repeat_evaluate(adder, math_suite, ExactMatchGrader(), repeats=1)
        assert intraclass_correlation(single) == 0.0

    def test_required_repeats_scales_with_variance(self):
        assert required_repeats(0.0) == 1
        assert required_repeats(0.10) > required_repeats(0.01)

    def test_empty_runs(self):
        assert analyze_stability([]).runs == 0


class TestContamination:
    def test_exact_leak_detected(self):
        text = "the quick brown fox jumps over the lazy dog near the river bank today"
        suite = suite_from_records("s", [{"id": "leaked", "input": text}])
        report = detect_contamination(suite, [text], n_gram=8, threshold=0.5)
        assert not report.clean
        assert "leaked" in report.contaminated_ids
        assert report.hits[0].overlap > 0.5

    def test_clean_task_untouched(self):
        suite = suite_from_records("s", [{"id": "clean", "input": "completely unrelated content here"}])
        report = detect_contamination(suite, ["nothing at all like the task input"], n_gram=8)
        assert report.clean

    def test_clean_suite_removes_hits(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        suite = suite_from_records("s", [
            {"id": "leaked", "input": text},
            {"id": "safe", "input": "totally different words appear in this one entirely"},
        ])
        report = detect_contamination(suite, [text], n_gram=8, threshold=0.5)
        cleaned = clean_suite(suite, report)
        assert len(cleaned) == len(suite) - len(report.hits)
        assert all(t.id not in report.contaminated_ids for t in cleaned.tasks)

    def test_corpus_index_reused(self):
        index = CorpusIndex(n_gram=5)
        index.add("one two three four five six seven")
        assert index.documents == 1 and index.size > 0
        suite = suite_from_records("s", [{"id": "x", "input": "one two three four five six seven"}])
        assert not detect_contamination(suite, index, n_gram=5, threshold=0.5).clean

    def test_duplicate_detection(self):
        suite = suite_from_records("s", [
            {"id": "a", "input": "what is the capital city of france"},
            {"id": "b", "input": "what is the capital city of france"},
            {"id": "c", "input": "an entirely unrelated chemistry question"},
        ])
        duplicates = find_duplicates(suite, n_gram=3, threshold=0.8)
        assert duplicates and duplicates[0][:2] == ("a", "b")
        assert duplicates[0][2] == pytest.approx(1.0)

    def test_no_duplicates(self):
        suite = suite_from_records("s", [
            {"id": "a", "input": "first distinct question about biology"},
            {"id": "b", "input": "second unrelated prompt regarding astronomy"},
        ])
        assert find_duplicates(suite, n_gram=3, threshold=0.8) == []


class TestCost:
    def test_charges_accumulate(self, small_suite):
        tracker = CostTracker(budget=1.0)
        evaluate(tracker.wrap(adder, fixed_cost=0.01), small_suite, ExactMatchGrader())
        assert tracker.report.total == pytest.approx(0.03)
        assert tracker.report.tasks == 3

    def test_budget_blocks_before_execution(self, math_suite):
        executed = []

        def tracked(task):
            executed.append(task.id)
            return adder(task)

        tracker = CostTracker(budget=0.025)
        run = evaluate(tracker.wrap(tracked, fixed_cost=0.01), math_suite, ExactMatchGrader())
        assert len(executed) == 2, "third task must not run"
        assert run.errored > 0
        assert tracker.report.stopped_early

    def test_soft_budget_does_not_stop(self, small_suite):
        tracker = CostTracker(budget=0.001, hard_stop=False)
        run = evaluate(tracker.wrap(adder, fixed_cost=0.01), small_suite, ExactMatchGrader())
        assert run.errored == 0

    def test_cost_fn_receives_prediction(self, small_suite):
        tracker = CostTracker()
        evaluate(
            tracker.wrap(adder, cost_fn=lambda t, p: len(str(p)) * 0.001),
            small_suite, ExactMatchGrader(),
        )
        assert tracker.report.total > 0

    def test_cost_per_pass(self, small_suite):
        tracker = CostTracker()
        run = evaluate(tracker.wrap(adder, fixed_cost=0.01), small_suite, ExactMatchGrader())
        assert tracker.report.cost_per_pass(run) == pytest.approx(0.01)

    def test_cost_per_pass_with_no_passes(self, small_suite):
        tracker = CostTracker()
        run = evaluate(tracker.wrap(always_wrong, fixed_cost=0.01), small_suite, ExactMatchGrader())
        assert tracker.report.cost_per_pass(run) == float("inf")

    def test_projection(self, small_suite):
        tracker = CostTracker()
        evaluate(tracker.wrap(adder, fixed_cost=0.01), small_suite, ExactMatchGrader())
        assert tracker.report.project(1000) == pytest.approx(10.0)

    def test_remaining_and_exhausted(self):
        tracker = CostTracker(budget=1.0)
        tracker.charge("t", 0.75)
        assert tracker.remaining == pytest.approx(0.25)
        assert not tracker.exhausted
        tracker.charge("t2", 0.30)
        assert tracker.exhausted

    def test_unbounded_budget(self):
        assert CostTracker().remaining == float("inf")
        assert not CostTracker().would_exceed(1e9)

    def test_token_estimation(self):
        assert estimate_tokens("x" * 400) == 100
        assert estimate_tokens("") == 1
        assert token_cost("x" * 4000, "y" * 400) > 0

    def test_budget_exceeded_is_library_error(self):
        tracker = CostTracker(budget=0.001)
        wrapped = tracker.wrap(adder, fixed_cost=1.0)
        with pytest.raises(BudgetExceededError):
            wrapped(Task(id="t", input="1+1", expected="2"))

    def test_cost_efficiency_table(self, math_suite):
        runs = evaluate_many({"a": adder, "b": always_wrong}, math_suite, ExactMatchGrader())
        trackers = {}
        for name in runs:
            tracker = CostTracker()
            for i in range(30):
                tracker.charge(f"q{i}", 0.001)
            trackers[name] = tracker.report
        table = cost_efficiency(runs, trackers)
        assert "per pass" in table
        assert "cheapest" in table

    def test_cost_efficiency_no_data(self):
        assert "No cost data" in cost_efficiency({}, {})


class TestStabilityInternals:
    def test_icc_is_high_when_difficulty_is_consistent(self, math_suite):
        def half_hard(task):
            index = int(str(task.input).split("+")[0])
            return adder(task) if index <= 15 else "0"

        runs = repeat_evaluate(half_hard, math_suite, ExactMatchGrader(), repeats=4)
        assert intraclass_correlation(runs) > 0.9

    def test_icc_is_low_when_outcomes_are_noise(self, math_suite):
        import random

        rng = random.Random(7)
        runs = [
            evaluate(lambda t: adder(t) if rng.random() < 0.5 else "0", math_suite, ExactMatchGrader())
            for _ in range(6)
        ]
        assert intraclass_correlation(runs) < 0.3

    def test_icc_ignores_tasks_seen_only_once(self, math_suite):
        runs = repeat_evaluate(adder, math_suite, ExactMatchGrader(), repeats=2)
        runs[0].results = runs[0].results[:5]
        assert intraclass_correlation(runs) == 0.0

    def test_icc_is_zero_without_any_variance(self, small_suite):
        runs = repeat_evaluate(adder, small_suite, ExactMatchGrader(), repeats=3)
        assert intraclass_correlation(runs) == 0.0

    def test_summary_reports_flaky_tasks(self, math_suite):
        import random

        rng = random.Random(3)
        runs = [
            evaluate(lambda t: adder(t) if rng.random() < 0.5 else "0", math_suite, ExactMatchGrader())
            for _ in range(5)
        ]
        text = analyze_stability(runs).summary()
        assert "flakiest tasks" in text
        assert "WARNING" in text

    def test_summary_of_a_stable_system_is_reassuring(self, math_suite):
        runs = repeat_evaluate(adder, math_suite, ExactMatchGrader(), repeats=3)
        text = analyze_stability(runs).summary()
        assert "stable enough" in text
        assert "flakiest tasks" not in text

    def test_empty_report_intervals_are_degenerate(self):
        report = analyze_stability([])
        assert report.combined_interval().low == 0.0
        assert report.naive_interval().high == 1.0
        assert report.spread == 0.0 and report.flake_rate == 0.0


class TestCompareByTag:
    def build(self):
        records = (
            [{"id": f"s{i}", "input": "x", "expected": "ok", "tags": ["safety"]} for i in range(10)]
            + [{"id": f"e{i}", "input": "x", "expected": "ok", "tags": ["easy"]} for i in range(10)]
        )
        suite = suite_from_records("m", records)
        base = evaluate(lambda t: "ok", suite, ExactMatchGrader(), system_name="base")
        cand = evaluate(
            lambda t: "no" if t.id.startswith("s") else "ok",
            suite, ExactMatchGrader(), system_name="cand",
        )
        return base, cand

    def test_finds_the_regressed_slice(self):
        base, cand = self.build()
        by_tag = compare_by_tag(base, cand)
        assert by_tag["safety"].verdict() == "REGRESSION"
        assert by_tag["easy"].verdict() == "INCONCLUSIVE"

    def test_tag_is_recorded_on_the_comparison(self):
        base, cand = self.build()
        comparison = compare_by_tag(base, cand)["safety"]
        assert comparison.tag == "safety"
        assert "[tag: safety]" in comparison.summary()

    def test_only_shared_tags_compared(self):
        base, cand = self.build()
        cand.results[0].tags = ("safety", "new_tag")
        assert set(compare_by_tag(base, cand)) == {"safety", "easy"}

    def test_explicit_tag_selection(self):
        base, cand = self.build()
        assert set(compare_by_tag(base, cand, tags={"safety"})) == {"safety"}

    def test_untagged_runs_produce_nothing(self, small_suite):
        base = evaluate(adder, small_suite, ExactMatchGrader())
        assert compare_by_tag(base, base) == {}


class TestTagRegressionGate:
    def build(self, broken=5, safety=10):
        records = (
            [{"id": f"s{i}", "input": "x", "expected": "ok", "tags": ["safety"]} for i in range(safety)]
            + [{"id": f"e{i}", "input": "x", "expected": "ok", "tags": ["easy"]} for i in range(20)]
        )
        suite = suite_from_records("m", records)
        broken_ids = {f"s{i}" for i in range(broken)}
        base = evaluate(lambda t: "ok", suite, ExactMatchGrader(), system_name="base")
        cand = evaluate(
            lambda t: "no" if t.id in broken_ids else "ok",
            suite, ExactMatchGrader(), system_name="cand",
        )
        return base, cand

    def test_blocks_a_tag_regression_the_aggregate_hides(self):
        base, cand = self.build()
        report = tag_regression_gate(base, cand, max_drop=0.05)
        assert not report.passed
        assert [g.name for g in report.failures] == ["tag:safety"]

    def test_healthy_tags_pass(self):
        base, cand = self.build(broken=0)
        assert tag_regression_gate(base, cand).passed

    def test_small_tags_are_reported_but_not_enforced(self):
        base, cand = self.build(broken=2, safety=3)
        report = tag_regression_gate(base, cand, min_tasks=5)
        safety = next(g for g in report.gates if g.name == "tag:safety")
        assert safety.passed
        assert "not enforced" in safety.detail

    def test_per_tag_thresholds(self):
        base, cand = self.build(broken=5)
        strict = tag_regression_gate(base, cand, max_drop={"safety": 0.0})
        lenient = tag_regression_gate(base, cand, max_drop={"safety": 0.9})
        assert not strict.passed and lenient.passed

    def test_threshold_dict_falls_back_for_unlisted_tags(self):
        base, cand = self.build(broken=5)
        report = tag_regression_gate(base, cand, max_drop={"easy": 0.5})
        assert not report.passed

    def test_broken_task_ids_appear_in_the_detail(self):
        base, cand = self.build()
        safety = next(g for g in tag_regression_gate(base, cand).gates if g.name == "tag:safety")
        assert "broken: s0" in safety.detail
