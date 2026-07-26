"""Every human-readable summary, including on empty and degenerate input.

These render arithmetic (percentages, ratios, widths) into strings, so a
division by zero or an index error here would surface directly to users.
"""

from __future__ import annotations

import pytest

from agenteval import (
    CorpusIndex,
    CostReport,
    CostTracker,
    ExactMatchGrader,
    GateReport,
    SequentialGate,
    Task,
    analyze_stability,
    benjamini_hochberg,
    calibration,
    clean_suite,
    cohens_kappa,
    compare,
    compare_all,
    detect_contamination,
    evaluate,
    evaluate_many,
    failure_digest,
    gate,
    holm_bonferroni,
    leaderboard,
    leaderboard_tiers,
    reliability_diagram_text,
    repeat_evaluate,
    run_to_html,
    run_to_text,
    suite_from_records,
    tag_breakdown,
    validate_judge,
)
from tests.helpers import adder, always_wrong, flaky


class TestComparisonSummary:
    def test_full_summary_renders(self, perfect_run, failing_run):
        text = compare(failing_run, perfect_run, iterations=500, seed=1).summary()
        assert "pass rate" in text and "VERDICT" in text

    def test_inconclusive_explains_itself(self, perfect_run):
        text = compare(perfect_run, perfect_run, iterations=500, seed=1).summary()
        assert "INCONCLUSIVE" in text
        assert "spans zero" in text

    def test_summary_with_no_shared_tasks(self, perfect_run):
        other = evaluate(adder, suite_from_records("o", [{"id": "z", "input": "1+1", "expected": "2"}]),
                         ExactMatchGrader())
        assert compare(perfect_run, other, iterations=200, seed=1).summary()

    def test_lists_fixed_and_broken(self, perfect_run, failing_run):
        text = compare(perfect_run, failing_run, iterations=500, seed=1).summary()
        assert "broken" in text


class TestMultipleComparisonSummary:
    def test_holm_summary(self):
        text = holm_bonferroni([0.01, 0.02, 0.03, 0.04]).summary()
        assert "Holm-Bonferroni" in text
        assert "significant" in text

    def test_notes_when_correction_changes_the_answer(self):
        text = holm_bonferroni([0.01, 0.02, 0.03, 0.04]).summary()
        assert "uncorrected" in text, "must say how many survived correction"

    def test_bh_summary(self):
        assert "Benjamini-Hochberg" in benjamini_hochberg([0.01, 0.5]).summary()

    def test_empty_summary(self):
        assert "No comparisons" in holm_bonferroni([]).summary()

    def test_matrix_summary(self, math_suite):
        runs = evaluate_many({"a": adder, "b": always_wrong}, math_suite, ExactMatchGrader())
        matrix = compare_all(runs, iterations=500, seed=1)
        text = matrix.summary()
        assert "Pairwise comparisons" in text

    def test_matrix_corrected_verdict(self, math_suite):
        runs = evaluate_many({"a": adder, "b": always_wrong}, math_suite, ExactMatchGrader())
        matrix = compare_all(runs, iterations=500, seed=1)
        assert matrix.corrected_verdict("a", "b") in {"IMPROVEMENT", "REGRESSION", "INCONCLUSIVE"}
        assert matrix.corrected_verdict("nope", "nah") == "UNKNOWN"

    def test_empty_matrix_summary(self):
        from agenteval import SystemMatrix

        assert "No pairwise" in SystemMatrix().summary()


class TestGateSummary:
    def test_passing_gates(self, perfect_run):
        assert "ALL GATES PASSED" in gate(perfect_run, min_pass_rate=0.5).summary()

    def test_failing_gates_are_counted(self, failing_run):
        text = gate(failing_run, min_pass_rate=0.9, min_mean_score=0.9).summary()
        assert "2 GATE(S) FAILED" in text

    def test_empty_gate_report(self):
        assert "No gates" in GateReport().summary()

    def test_latency_gate(self, perfect_run):
        assert gate(perfect_run, max_mean_latency_ms=1e9).passed


class TestAgreementSummary:
    def test_trustworthy_judge(self):
        report = cohens_kappa([True] * 10 + [False] * 10, [True] * 10 + [False] * 10)
        text = report.summary()
        assert "Cohen's kappa" in text
        assert "well enough to rely on" in text

    def test_untrustworthy_judge_warns(self):
        report = cohens_kappa([True] * 10 + [False] * 10, [True] * 20)
        assert "WARNING" in report.summary()

    def test_empty_agreement_summary(self):
        assert cohens_kappa([], []).summary()

    def test_judge_validation_summary_flags_leniency(self):
        tasks = [Task(id=f"t{i}", input="q", expected="a") for i in range(20)]
        labeled = [(tasks[i], "p", i % 2 == 0) for i in range(20)]
        text = validate_judge(lambda p, t: True, labeled).summary()
        assert "BIAS" in text and "lenient" in text

    def test_judge_validation_summary_flags_strictness(self):
        tasks = [Task(id=f"t{i}", input="q", expected="a") for i in range(20)]
        labeled = [(tasks[i], "p", True) for i in range(20)]
        text = validate_judge(lambda p, t: False, labeled).summary()
        assert "strict" in text

    def test_balanced_judge_reports_no_bias(self):
        tasks = [Task(id=f"t{i}", input="q", expected="a") for i in range(20)]
        labeled = [(tasks[i], "p", i % 2 == 0) for i in range(20)]
        text = validate_judge(lambda p, t: int(t.id[1:]) % 2 == 0, labeled).summary()
        assert "similar rates" in text

    def test_disagreements_are_listed(self):
        tasks = [Task(id=f"t{i}", input="q", expected="a") for i in range(10)]
        labeled = [(tasks[i], "p", True) for i in range(10)]
        text = validate_judge(lambda p, t: False, labeled).summary()
        assert "disagreements" in text


class TestCalibrationSummary:
    def test_overconfident_summary(self):
        text = calibration([0.9] * 20, [True] * 10 + [False] * 10).summary()
        assert "OVERCONFIDENT" in text
        assert "ECE" in text

    def test_underconfident_summary(self):
        text = calibration([0.1] * 20, [True] * 18 + [False] * 2).summary()
        assert "UNDERCONFIDENT" in text

    def test_well_calibrated_summary(self):
        text = calibration([1.0] * 10 + [0.0] * 10, [True] * 10 + [False] * 10).summary()
        assert "Well calibrated" in text

    def test_aggregate_can_look_calibrated_while_bins_are_wrong(self):
        # Half claim 90% confidence, half claim 10%, both are 50% accurate.
        # Mean confidence equals accuracy, so neither over- nor under-confident,
        # but ECE is large because every individual bin is badly off.
        confidences = [0.9] * 10 + [0.1] * 10
        correct = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
        report = calibration(confidences, correct)
        assert report.mean_confidence == pytest.approx(report.accuracy)
        assert not report.overconfident and not report.underconfident
        assert report.ece > 0.05
        assert "Roughly calibrated" in report.summary()

    def test_empty_summary(self):
        assert calibration([], []).summary()

    def test_reliability_diagram(self):
        report = calibration([0.2] * 10 + [0.8] * 10, [False] * 10 + [True] * 10)
        diagram = reliability_diagram_text(report)
        assert "Reliability diagram" in diagram
        assert "critical" not in diagram


class TestStabilitySummary:
    def test_stable_system_summary(self, math_suite):
        report = analyze_stability(repeat_evaluate(adder, math_suite, ExactMatchGrader(), repeats=3))
        text = report.summary()
        assert "run-to-run stdev" in text
        assert "stable enough" in text

    def test_unstable_system_warns(self, math_suite):
        import random

        random.seed(5)
        runs = repeat_evaluate(flaky(0.5, "x"), math_suite, ExactMatchGrader(), repeats=5)
        report = analyze_stability(runs)
        if not report.reliable:
            assert "WARNING" in report.summary()
        assert "pass rates" in report.summary()

    def test_summary_reports_both_intervals(self, math_suite):
        report = analyze_stability(repeat_evaluate(adder, math_suite, ExactMatchGrader(), repeats=3))
        text = report.summary()
        assert "single-run CI" in text and "variance-aware CI" in text

    def test_empty_stability_summary(self):
        assert analyze_stability([]).summary()

    def test_flaky_tasks_listed(self, math_suite):
        import random

        random.seed(6)
        runs = repeat_evaluate(flaky(0.5, "y"), math_suite, ExactMatchGrader(), repeats=5)
        report = analyze_stability(runs)
        if report.flaky:
            assert "flakiest tasks" in report.summary()


class TestContaminationSummary:
    def test_clean_summary(self):
        suite = suite_from_records("s", [{"id": "a", "input": "unrelated text entirely different"}])
        text = detect_contamination(suite, ["nothing similar here at all"], n_gram=8).summary()
        assert "No overlap" in text

    def test_contaminated_summary_warns(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        suite = suite_from_records("s", [{"id": "leak", "input": text}])
        report = detect_contamination(suite, [text], n_gram=8, threshold=0.5)
        summary = report.summary()
        assert "memorization" in summary
        assert "leak" in summary

    def test_corpus_index_properties(self):
        index = CorpusIndex(n_gram=4)
        index.add_all(["one two three four five", "six seven eight nine ten"])
        assert index.documents == 2 and index.size > 0
        assert index.matched_sample("one two three four five")
        assert index.matched_sample("zzz yyy xxx www") == ""

    def test_clean_suite_preserves_description(self):
        suite = suite_from_records("s", [{"id": "a", "input": "x"}], description="desc")
        report = detect_contamination(suite, [], n_gram=8)
        assert clean_suite(suite, report).description == "desc"


class TestCostSummary:
    def test_summary_with_run(self, small_suite):
        tracker = CostTracker(budget=1.0)
        run = evaluate(tracker.wrap(adder, fixed_cost=0.01), small_suite, ExactMatchGrader())
        text = tracker.report.summary(run)
        assert "total cost" in text and "cost per passing" in text
        assert "projected" in text

    def test_summary_without_run(self):
        report = CostReport(total=1.0, unit="usd")
        assert "total cost" in report.summary()

    def test_empty_cost_summary(self):
        assert CostReport().summary()

    def test_stopped_early_is_surfaced(self, math_suite):
        tracker = CostTracker(budget=0.015)
        evaluate(tracker.wrap(adder, fixed_cost=0.01), math_suite, ExactMatchGrader())
        assert "STOPPED EARLY" in tracker.report.summary()

    def test_token_counts_reported(self):
        tracker = CostTracker()
        tracker.charge("t", 0.01, tokens_in=100, tokens_out=50)
        assert "tokens in / out" in tracker.report.summary()


class TestSequentialSummary:
    def test_decision_str(self):
        gate_ = SequentialGate(threshold=0.5, min_samples=5)
        early = gate_.check(1, 2)
        assert "CONTINUE" in str(early)
        decisive = gate_.check(20, 20)
        assert "STOP" in str(decisive)

    def test_budget_exhaustion_is_reported(self):
        gate_ = SequentialGate(threshold=0.5, min_samples=2, max_samples=4)
        decision = gate_.check(2, 4)
        assert decision.stop
        assert "budget" in decision.reason or "above" in decision.reason or "below" in decision.reason


class TestReportRendering:
    def test_text_report_on_empty_run(self):
        run = evaluate(adder, suite_from_records("e", [{"id": "a", "input": "1+1", "expected": "2"}]),
                       ExactMatchGrader())
        assert run_to_text(run)

    def test_tag_breakdown_without_tags(self, small_suite):
        run = evaluate(adder, small_suite, ExactMatchGrader())
        assert "No tags" in tag_breakdown(run)

    def test_tag_breakdown_warns_on_thin_tags(self, math_suite):
        run = evaluate(adder, math_suite, ExactMatchGrader())
        assert "fewer than 20" in tag_breakdown(run)

    def test_leaderboard_empty(self):
        assert "No runs" in leaderboard({})
        assert "No runs" in leaderboard_tiers({})

    def test_leaderboard_single_system(self, perfect_run):
        assert "perfect" in leaderboard({"perfect": perfect_run})

    def test_tiers_all_tied(self, math_suite):
        runs = evaluate_many({"a": adder, "b": adder}, math_suite, ExactMatchGrader())
        assert "within noise" in leaderboard_tiers(runs)

    def test_failure_digest_groups_by_cause(self, math_suite):
        def broken(task):
            raise RuntimeError("same cause")

        run = evaluate(broken, math_suite, ExactMatchGrader())
        digest = failure_digest(run)
        assert "30x" in digest or "30" in digest

    def test_html_without_failures(self, perfect_run):
        html = run_to_html(perfect_run)
        assert "Failures" not in html

    def test_html_truncates_long_failure_lists(self, math_suite):
        run = evaluate(always_wrong, math_suite, ExactMatchGrader())
        html = run_to_html(run, show_failures=5)
        assert "Showing 5 of 30" in html

    def test_html_with_calibration_and_stability(self, math_suite):
        run = evaluate(adder, math_suite, ExactMatchGrader())
        cal = calibration([0.9] * 20, [True] * 10 + [False] * 10)
        stab = analyze_stability(repeat_evaluate(adder, math_suite, ExactMatchGrader(), repeats=2))
        html = run_to_html(run, calibration_report=cal, stability_report=stab)
        assert "Calibration" in html and "Run-to-run stability" in html
        assert "Overconfident" in html
