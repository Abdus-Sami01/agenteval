from __future__ import annotations

import pytest

from agenteval import (
    CallableGrader,
    ConfigurationError,
    ContainsGrader,
    EditDistanceGrader,
    ExactMatchGrader,
    F1TokenGrader,
    Grader,
    GraderRegistry,
    JSONSchemaGrader,
    LLMJudgeGrader,
    NumericGrader,
    OutcomeGrader,
    PredicateGrader,
    RangeGrader,
    RegexGrader,
    RubricGrader,
    Score,
    SetGrader,
    Step,
    StepBudgetGrader,
    StructuralGrader,
    Task,
    ToolSequenceGrader,
    Trajectory,
    UnknownGraderError,
    WeightedGrader,
    normalize_text,
)
from agenteval.graders.numeric import extract_number
from agenteval.graders.rubric import parse_judge_score
from agenteval.graders.structured import _shallow_schema_check, coerce_json


def make_task(expected, inp="question"):
    return Task(id="t", input=inp, expected=expected)


class TestNormalizeText:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_text("The Answer!") == "answer"

    def test_strips_articles(self):
        assert normalize_text("a cat and the dog") == "cat and dog"

    def test_collapses_whitespace(self):
        assert normalize_text("  too   many spaces ") == "too many spaces"

    def test_unicode_normalization(self):
        assert normalize_text("ﬁle") == "file"

    def test_flags_can_be_disabled(self):
        assert normalize_text("The Answer!", lowercase=False, strip_punctuation=False) == "The Answer!"


class TestExactMatch:
    def test_normalized_match(self):
        assert ExactMatchGrader().grade("The Answer.", make_task("the answer")).passed

    def test_mismatch_reports_both_sides(self):
        score = ExactMatchGrader().grade("wrong", make_task("right"))
        assert not score.passed
        assert "right" in score.detail and "wrong" in score.detail

    def test_case_sensitivity(self):
        assert not ExactMatchGrader(case_sensitive=True, normalize=False).grade("ABC", make_task("abc")).passed
        assert ExactMatchGrader(case_sensitive=False, normalize=False).grade("ABC", make_task("abc")).passed

    def test_without_normalization_punctuation_matters(self):
        assert not ExactMatchGrader(normalize=False).grade("answer.", make_task("answer")).passed


class TestContains:
    def test_all_present(self):
        score = ContainsGrader().grade("alpha beta gamma", make_task(["alpha", "gamma"]))
        assert score.passed and score.value == 1.0

    def test_partial_credit(self):
        score = ContainsGrader().grade("alpha only", make_task(["alpha", "missing"]))
        assert not score.passed
        assert score.value == 0.5
        assert "missing" in score.detail

    def test_any_mode(self):
        assert ContainsGrader(require_all=False).grade("alpha", make_task(["alpha", "zzz"])).passed

    def test_scalar_expected(self):
        assert ContainsGrader().grade("hello world", make_task("world")).passed


class TestRegex:
    def test_match(self):
        assert RegexGrader(pattern=r"\d{3}-\d{4}").grade("call 555-1234", make_task(None)).passed

    def test_no_match(self):
        assert not RegexGrader(pattern=r"\d+").grade("no digits", make_task(None)).passed

    def test_pattern_from_expected(self):
        assert RegexGrader(use_expected=True).grade("abc123", make_task(r"[a-z]+\d+")).passed

    def test_missing_pattern(self):
        assert "no pattern" in RegexGrader().grade("x", make_task(None)).detail

    def test_invalid_pattern_reports_error(self):
        score = RegexGrader(pattern="[unclosed").grade("x", make_task(None))
        assert not score.passed and "invalid regex" in score.detail


class TestEditDistance:
    def test_identical(self):
        assert EditDistanceGrader().grade("hello", make_task("hello")).value == 1.0

    def test_similarity_reported(self):
        score = EditDistanceGrader(threshold=0.5).grade("kitten", make_task("sitting"))
        assert score.value == pytest.approx(4 / 7, abs=1e-6)
        assert score.passed

    def test_threshold_enforced(self):
        assert not EditDistanceGrader(threshold=0.9).grade("kitten", make_task("sitting")).passed

    def test_both_empty(self):
        assert EditDistanceGrader().grade("", make_task("")).value == 1.0


class TestF1Token:
    def test_perfect_overlap(self):
        score = F1TokenGrader().grade("the cat sat", make_task("the cat sat"))
        assert score.value == 1.0 and score.passed

    def test_partial_overlap_precision_and_recall(self):
        score = F1TokenGrader(threshold=0.5).grade("cat sat", make_task("cat sat on mat"))
        assert score.subscores["precision"] == 1.0
        assert score.subscores["recall"] == 0.5
        assert score.value == pytest.approx(2 / 3, abs=1e-6)

    def test_no_overlap(self):
        score = F1TokenGrader().grade("alpha", make_task("beta"))
        assert score.value == 0.0 and "no token overlap" in score.detail

    def test_one_side_empty(self):
        assert not F1TokenGrader().grade("", make_task("something")).passed

    def test_both_empty(self):
        assert F1TokenGrader().grade("", make_task("")).value == 1.0


class TestNumeric:
    def test_extract_from_prose(self):
        assert extract_number("the result is 3.14") == pytest.approx(3.14)
        assert extract_number("1,234") == pytest.approx(1234)
        assert extract_number("1e3") == pytest.approx(1000)
        assert extract_number("no digits") is None

    def test_booleans_are_not_numbers(self):
        assert extract_number(True) is None

    def test_within_tolerance(self):
        assert NumericGrader(tolerance=0.01).grade("3.145", make_task(3.14)).passed

    def test_outside_tolerance(self):
        score = NumericGrader(tolerance=0.01).grade("3.20", make_task(3.14))
        assert not score.passed and "delta" in score.detail

    def test_relative_tolerance(self):
        assert NumericGrader(tolerance=0.05, relative=True).grade("102", make_task(100)).passed
        assert not NumericGrader(tolerance=0.01, relative=True).grade("102", make_task(100)).passed

    def test_missing_number(self):
        assert "no number found" in NumericGrader().grade("nothing", make_task(5)).detail

    def test_non_numeric_expected(self):
        assert not NumericGrader().grade("5", make_task("abc")).passed


class TestRange:
    def test_inside_and_outside(self):
        assert RangeGrader(low=0, high=10).grade("7", make_task(None)).passed
        assert not RangeGrader(low=0, high=10).grade("70", make_task(None)).passed

    def test_bounds_from_expected(self):
        assert RangeGrader().grade("5", make_task([0, 10])).passed

    def test_exclusive_bounds(self):
        assert not RangeGrader(low=0, high=10, inclusive=False).grade("10", make_task(None)).passed
        assert RangeGrader(low=0, high=10, inclusive=True).grade("10", make_task(None)).passed

    def test_open_ended(self):
        assert RangeGrader(low=0).grade("999999", make_task(None)).passed


class TestSet:
    def test_exact_set(self):
        assert SetGrader().grade('["a","b"]', make_task(["a", "b"])).passed

    def test_order_insensitive_by_default(self):
        assert SetGrader().grade('["b","a"]', make_task(["a", "b"])).passed

    def test_order_sensitive_mode(self):
        assert not SetGrader(order_matters=True).grade(["b", "a"], make_task(["a", "b"])).passed

    def test_precision_and_recall(self):
        score = SetGrader(threshold=0.5).grade("a, b, z", make_task(["a", "b", "c"]))
        assert score.subscores["precision"] == pytest.approx(2 / 3)
        assert score.subscores["recall"] == pytest.approx(2 / 3)

    def test_comma_separated_fallback(self):
        assert SetGrader().grade("a, b", make_task(["a", "b"])).passed

    def test_both_empty(self):
        assert SetGrader().grade("[]", make_task([])).value == 1.0


class TestJSONHandling:
    def test_coerce_plain_json(self):
        assert coerce_json('{"a": 1}') == {"a": 1}

    def test_coerce_from_surrounding_prose(self):
        assert coerce_json('here you go: {"a": 1} done') == {"a": 1}

    def test_coerce_nested(self):
        assert coerce_json('{"a": {"b": [1,2]}}') == {"a": {"b": [1, 2]}}

    def test_coerce_array(self):
        assert coerce_json("prefix [1,2,3]") == [1, 2, 3]

    def test_coerce_invalid(self):
        assert coerce_json("not json at all") is None


class TestJSONSchema:
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    }

    def test_valid(self):
        assert JSONSchemaGrader(schema=self.schema).grade('{"n": 5}', make_task(None)).passed

    def test_wrong_type_rejected(self):
        score = JSONSchemaGrader(schema=self.schema).grade('{"n": "five"}', make_task(None))
        assert not score.passed and "integer" in score.detail

    def test_missing_required_rejected(self):
        assert not JSONSchemaGrader(schema=self.schema).grade("{}", make_task(None)).passed

    def test_unparseable_output(self):
        score = JSONSchemaGrader(schema=self.schema).grade("garbage", make_task(None))
        assert not score.passed and "not valid JSON" in score.detail

    def test_no_schema_configured(self):
        assert "no schema" in JSONSchemaGrader().grade("{}", make_task(None)).detail

    def test_schema_from_task_metadata(self):
        task = Task(id="t", input="q", expected=None, metadata={"schema": self.schema})
        assert JSONSchemaGrader().grade('{"n": 1}', task).passed


class TestStructural:
    def test_exact_structure(self):
        assert StructuralGrader().grade('{"a":1,"b":{"c":2}}', make_task({"a": 1, "b": {"c": 2}})).value == 1.0

    def test_partial_credit_on_nested_mismatch(self):
        score = StructuralGrader().grade('{"a":1,"b":{"c":99}}', make_task({"a": 1, "b": {"c": 2}}))
        assert 0 < score.value < 1
        assert "b.c" in score.detail

    def test_missing_key_reported(self):
        score = StructuralGrader().grade('{"a":1}', make_task({"a": 1, "b": 2}))
        assert "b" in score.detail

    def test_partial_credit_can_be_disabled(self):
        score = StructuralGrader(partial_credit=False).grade('{"a":1,"b":9}', make_task({"a": 1, "b": 2}))
        assert score.value == 0.0

    def test_unparseable(self):
        assert not StructuralGrader().grade("nope", make_task({"a": 1})).passed


class TestProgrammatic:
    def test_predicate_true_and_false(self):
        assert PredicateGrader(lambda p, t: len(str(p)) > 3).grade("hello", make_task(None)).passed
        assert not PredicateGrader(lambda p, t: False).grade("x", make_task(None)).passed

    def test_predicate_exception_is_captured(self):
        score = PredicateGrader(lambda p, t: 1 / 0, label="div").grade("x", make_task(None))
        assert not score.passed and "ZeroDivisionError" in score.detail

    def test_callable_returning_float(self):
        assert CallableGrader(lambda p, t: 1.0).grade("x", make_task(None)).passed

    def test_callable_returning_tuple(self):
        score = CallableGrader(lambda p, t: (0.75, "partial")).grade("x", make_task(None))
        assert score.value == 0.75 and score.detail == "partial"

    def test_callable_returning_bool(self):
        assert CallableGrader(lambda p, t: True).grade("x", make_task(None)).passed

    def test_callable_returning_score_passes_through(self):
        original = Score(value=0.5, passed=True, grader="custom")
        assert CallableGrader(lambda p, t: original).grade("x", make_task(None)) is original

    def test_callable_returning_garbage(self):
        score = CallableGrader(lambda p, t: object()).grade("x", make_task(None))
        assert not score.passed and "non-numeric" in score.detail


class TestRubric:
    def test_weighted_criteria(self):
        grader = RubricGrader(
            {
                "has_digit": lambda p, t: any(c.isdigit() for c in str(p)),
                "is_long": lambda p, t: len(str(p)) > 50,
                "polite": lambda p, t: "please" in str(p).lower(),
            },
            weights={"has_digit": 2.0},
        )
        score = grader.grade("value is 42 please", make_task(None))
        assert score.value == pytest.approx(0.75)
        assert score.subscores["is_long"] == 0.0
        assert "is_long" in score.detail

    def test_no_criteria(self):
        assert "no criteria" in RubricGrader({}).grade("x", make_task(None)).detail

    def test_criterion_exception_counts_as_failure(self):
        score = RubricGrader({"boom": lambda p, t: 1 / 0}).grade("x", make_task(None))
        assert score.value == 0.0 and "boom" in score.detail


class TestWeighted:
    def test_combines_subgraders(self):
        grader = WeightedGrader(
            {"exact": ExactMatchGrader(), "f1": F1TokenGrader(threshold=0.0)},
            weights={"f1": 3.0},
            threshold=0.5,
        )
        score = grader.grade("the cat sat", make_task("the cat sat on mat"))
        assert score.subscores["exact"] == 0.0
        assert score.subscores["f1"] > 0
        assert 0 < score.value < 1

    def test_no_graders(self):
        assert "no graders" in WeightedGrader({}).grade("x", make_task(None)).detail


class TestJudgeParsing:
    @pytest.mark.parametrize("raw,expected", [
        ("Score: 1", 1.0),
        ("score = 0", 0.0),
        ("Rating: 4", 4.0),
        ("3/5", 0.6),
        ("Yes, correct", 1.0),
        ("No, incorrect", 0.0),
        ("  7  ", 7.0),
    ])
    def test_recognized_formats(self, raw, expected):
        assert parse_judge_score(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["mumble", "hello there", "maybe"])
    def test_unparseable_returns_none(self, raw):
        assert parse_judge_score(raw) is None

    def test_ambiguous_verdict_returns_none(self):
        assert parse_judge_score("yes and no") is None

    def test_scale_divides(self):
        assert parse_judge_score("Score: 5", scale=5.0) == pytest.approx(1.0)


class TestLLMJudge:
    def test_passing_verdict(self):
        assert LLMJudgeGrader(judge_fn=lambda p: "Score: 1").grade("x", make_task("y")).passed

    def test_failing_verdict(self):
        assert not LLMJudgeGrader(judge_fn=lambda p: "Score: 0").grade("x", make_task("y")).passed

    def test_unparseable_fails_loudly(self):
        score = LLMJudgeGrader(judge_fn=lambda p: "mumble").grade("x", make_task("y"))
        assert not score.passed
        assert "could not parse" in score.detail

    def test_judge_exception_surfaces(self):
        def boom(prompt):
            raise RuntimeError("api down")

        score = LLMJudgeGrader(judge_fn=boom).grade("x", make_task("y"))
        assert "judge call failed" in score.detail

    def test_prompt_receives_substitutions(self):
        seen = {}

        def judge(prompt):
            seen["prompt"] = prompt
            return "Score: 1"

        LLMJudgeGrader(judge_fn=judge).grade("MYPREDICTION", make_task("MYEXPECTED", inp="MYINPUT"))
        assert "MYPREDICTION" in seen["prompt"]
        assert "MYEXPECTED" in seen["prompt"]
        assert "MYINPUT" in seen["prompt"]


class TestRegistry:
    def test_all_graders_registered(self):
        assert len(GraderRegistry.available()) == 18

    def test_create_by_name(self):
        assert isinstance(GraderRegistry.create("exact"), ExactMatchGrader)

    def test_unknown_grader_lists_alternatives(self):
        with pytest.raises(UnknownGraderError) as excinfo:
            GraderRegistry.create("nonexistent")
        assert "exact" in str(excinfo.value)

    def test_has(self):
        assert GraderRegistry.has("f1")
        assert not GraderRegistry.has("nope")

    def test_custom_grader_can_register(self):
        class Custom(Grader):
            name = "custom"

            def grade(self, prediction, task):
                return self._score(1.0)

        GraderRegistry.register("custom_test", Custom)
        try:
            assert GraderRegistry.create("custom_test").grade("x", make_task(None)).passed
        finally:
            GraderRegistry._registry.pop("custom_test", None)


class TestScoreClamping:
    def test_scores_are_clamped_to_unit_interval(self):
        assert CallableGrader(lambda p, t: 5.0).grade("x", make_task(None)).value == 1.0
        assert CallableGrader(lambda p, t: -3.0).grade("x", make_task(None)).value == 0.0

    def test_base_grader_is_abstract(self):
        with pytest.raises(NotImplementedError):
            Grader().grade("x", make_task(None))


def trajectory(actions, output="ok", **kw):
    return Trajectory(steps=[Step(action=a, **kw) for a in actions], output=output)


class TestTrajectoryType:
    def test_actions_and_length(self):
        traj = trajectory(["search", "fetch", "answer"])
        assert traj.actions == ["search", "fetch", "answer"]
        assert len(traj) == 3

    def test_counts_repeated_actions(self):
        assert trajectory(["search", "search", "fetch"]).count("search") == 2

    def test_totals_roll_up(self):
        traj = Trajectory(steps=[
            Step(action="a", cost=0.5, elapsed_ms=10.0),
            Step(action="b", cost=0.25, elapsed_ms=5.0),
        ])
        assert traj.total_cost == 0.75
        assert traj.total_ms == 15.0

    def test_failed_steps_isolated(self):
        traj = Trajectory(steps=[Step(action="a"), Step(action="b", error="429 rate limited")])
        assert [s.action for s in traj.failed_steps] == ["b"]
        assert traj.steps[0].ok and not traj.steps[1].ok

    def test_from_records_accepts_common_key_names(self):
        traj = Trajectory.from_records(
            [{"tool": "search", "input": {"q": "x"}, "output": "hit"},
             {"action": "fetch", "args": {"url": "u"}, "observation": "body", "cost": 0.02}],
            output="done", model="gpt-4o",
        )
        assert traj.actions == ["search", "fetch"]
        assert traj.steps[0].args == {"q": "x"} and traj.steps[0].observation == "hit"
        assert traj.total_cost == 0.02
        assert traj.output == "done" and traj.metadata == {"model": "gpt-4o"}

    def test_as_dict_omits_empty_fields(self):
        d = Trajectory(steps=[Step(action="a")], output="x").as_dict()
        assert d["steps"] == [{"action": "a"}]
        assert d["output"] == "x"

    def test_str_is_readable_and_truncates(self):
        assert "search -> fetch" in str(trajectory(["search", "fetch"]))
        assert "8 steps" in str(trajectory([f"s{i}" for i in range(8)]))

    def test_empty_trajectory_renders(self):
        assert "empty" in str(Trajectory())


class TestOutcomeGrader:
    def test_grades_the_final_output(self):
        grader = OutcomeGrader(ExactMatchGrader())
        assert grader.grade(trajectory(["a"], output="42"), Task(id="t", input="q", expected="42")).passed

    def test_ignores_the_steps(self):
        grader = OutcomeGrader(ExactMatchGrader())
        assert not grader.grade(trajectory(["a"], output="7"), Task(id="t", input="q", expected="42")).passed

    def test_passes_plain_predictions_through(self):
        grader = OutcomeGrader(ExactMatchGrader())
        assert grader.grade("42", Task(id="t", input="q", expected="42")).passed

    def test_accepts_a_bare_step_list(self):
        grader = OutcomeGrader(ExactMatchGrader())
        steps = [Step(action="a")]
        assert not grader.grade(steps, Task(id="t", input="q", expected="42")).passed

    def test_repr_names_the_inner_grader(self):
        assert "ExactMatchGrader" in repr(OutcomeGrader(ExactMatchGrader()))


class TestToolSequenceGrader:
    task = Task(id="t", input="q", expected=["search", "fetch"])

    def test_exact_match(self):
        grader = ToolSequenceGrader(["search", "fetch"], mode="exact")
        assert grader.grade(trajectory(["search", "fetch"]), self.task).passed

    def test_exact_rejects_extra_calls(self):
        grader = ToolSequenceGrader(["search", "fetch"], mode="exact")
        score = grader.grade(trajectory(["search", "fetch", "fetch"]), self.task)
        assert not score.passed and score.value < 1.0

    def test_subsequence_tolerates_detours(self):
        grader = ToolSequenceGrader(["search", "fetch"], mode="subsequence")
        assert grader.grade(trajectory(["search", "think", "fetch"]), self.task).passed

    def test_subsequence_requires_order(self):
        grader = ToolSequenceGrader(["search", "fetch"], mode="subsequence")
        score = grader.grade(trajectory(["fetch", "search"]), self.task)
        assert not score.passed and score.value == 0.5

    def test_set_mode_ignores_order(self):
        grader = ToolSequenceGrader(["search", "fetch"], mode="set")
        assert grader.grade(trajectory(["fetch", "search"]), self.task).passed

    def test_partial_credit_beats_nothing(self):
        grader = ToolSequenceGrader(["search", "fetch", "answer"], mode="subsequence")
        partial = grader.grade(trajectory(["search", "fetch"]), self.task)
        nothing = grader.grade(trajectory(["nap"]), self.task)
        assert partial.value == pytest.approx(2 / 3)
        assert nothing.value == 0.0

    def test_forbidden_tool_zeroes_the_score(self):
        grader = ToolSequenceGrader(["search"], forbidden=["delete_database"])
        score = grader.grade(trajectory(["search", "delete_database"]), self.task)
        assert score.value == 0.0 and "forbidden" in score.detail

    def test_expected_sequence_read_from_task(self):
        assert ToolSequenceGrader().grade(trajectory(["search", "fetch"]), self.task).passed

    def test_expected_sequence_read_from_task_dict(self):
        task = Task(id="t", input="q", expected={"tools": ["search"], "answer": "42"})
        assert ToolSequenceGrader().grade(trajectory(["search"]), task).passed

    def test_missing_expectation_is_reported(self):
        score = ToolSequenceGrader().grade(trajectory(["search"]), Task(id="t", input="q"))
        assert not score.passed and "no expected tool sequence" in score.detail

    def test_non_trajectory_prediction_is_reported(self):
        score = ToolSequenceGrader(["search"]).grade("just a string", self.task)
        assert not score.passed and "not a Trajectory" in score.detail

    def test_detail_shows_both_sequences(self):
        score = ToolSequenceGrader(["search", "fetch"]).grade(trajectory(["search"]), self.task)
        assert "expected" in score.detail and "missing" in score.detail

    def test_unknown_mode_rejected(self):
        with pytest.raises(ConfigurationError):
            ToolSequenceGrader(["a"], mode="fuzzy")


class TestStepBudgetGrader:
    task = Task(id="t", input="q", expected=None)

    def test_within_budget_passes(self):
        assert StepBudgetGrader(max_steps=5).grade(trajectory(["a", "b"]), self.task).passed

    def test_over_budget_fails(self):
        score = StepBudgetGrader(max_steps=2).grade(trajectory(["a", "b", "c"]), self.task)
        assert not score.passed and "over budget" in score.detail

    def test_score_decays_with_overrun(self):
        grader = StepBudgetGrader(max_steps=4)
        small = grader.grade(trajectory(["a"] * 5), self.task)
        large = grader.grade(trajectory(["a"] * 7), self.task)
        assert small.value > large.value > 0.0

    def test_score_floors_at_twice_the_budget(self):
        assert StepBudgetGrader(max_steps=4).grade(trajectory(["a"] * 40), self.task).value == 0.0

    def test_cost_budget_enforced(self):
        traj = Trajectory(steps=[Step(action="a", cost=0.5), Step(action="b", cost=0.4)])
        score = StepBudgetGrader(max_cost=0.5).grade(traj, self.task)
        assert not score.passed and "cost" in score.detail

    def test_failed_steps_can_be_disallowed(self):
        traj = Trajectory(steps=[Step(action="a", error="timeout")])
        assert StepBudgetGrader(allow_errors=False).grade(traj, self.task).passed is False
        assert StepBudgetGrader(allow_errors=True).grade(traj, self.task).passed is True

    def test_subscores_reported_per_budget(self):
        traj = Trajectory(steps=[Step(action="a", cost=0.1)])
        score = StepBudgetGrader(max_steps=2, max_cost=1.0, allow_errors=False).grade(traj, self.task)
        assert set(score.subscores) == {"steps", "cost", "clean"}

    def test_no_budget_configured_is_a_pass(self):
        assert StepBudgetGrader().grade(trajectory(["a"] * 99), self.task).passed

    def test_non_trajectory_prediction_is_reported(self):
        score = StepBudgetGrader(max_steps=1).grade("string", self.task)
        assert not score.passed and "not a Trajectory" in score.detail

    def test_negative_budget_rejected(self):
        with pytest.raises(ConfigurationError):
            StepBudgetGrader(max_steps=-1)


class TestTrajectoryComposition:
    def test_weighted_grader_blends_outcome_and_process(self):
        task = Task(id="t", input="q", expected="42")
        grader = WeightedGrader({
            "answer": OutcomeGrader(ExactMatchGrader()),
            "tools": ToolSequenceGrader(["search", "fetch"], mode="exact"),
            "budget": StepBudgetGrader(max_steps=3),
        }, threshold=0.99)

        good = trajectory(["search", "fetch"], output="42")
        assert grader.grade(good, task).passed

        right_answer_wrong_process = trajectory(["guess"], output="42")
        score = grader.grade(right_answer_wrong_process, task)
        assert not score.passed
        assert score.subscores["answer"] == 1.0 and score.subscores["tools"] == 0.0


class TestShallowSchemaFallback:
    """The no-jsonschema path, exercised directly since jsonschema is a test dep."""

    def test_type_checks(self):
        assert _shallow_schema_check({"a": 1}, {"type": "object"})
        assert not _shallow_schema_check([1], {"type": "object"})
        assert _shallow_schema_check("x", {"type": "string"})
        assert not _shallow_schema_check(1, {"type": "string"})

    def test_booleans_are_not_integers(self):
        assert not _shallow_schema_check(True, {"type": "integer"})
        assert not _shallow_schema_check(False, {"type": "number"})
        assert _shallow_schema_check(True, {"type": "boolean"})

    def test_required_keys_enforced(self):
        assert _shallow_schema_check({"a": 1}, {"required": ["a"]})
        assert not _shallow_schema_check({"b": 1}, {"required": ["a"]})

    def test_nested_properties_are_checked(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
        assert _shallow_schema_check({"n": 3}, schema)
        assert not _shallow_schema_check({"n": "three"}, schema)

    def test_absent_optional_property_is_fine(self):
        assert _shallow_schema_check({}, {"type": "object", "properties": {"n": {"type": "integer"}}})

    def test_unknown_type_is_ignored(self):
        assert _shallow_schema_check("anything", {"type": "null"})


class TestStructuralLists:
    def task(self, expected):
        return Task(id="t", input="q", expected=expected)

    def test_ordered_lists_must_line_up(self):
        grader = StructuralGrader(ignore_order=False)
        assert grader.grade([1, 2, 3], self.task([1, 2, 3])).passed
        assert not grader.grade([3, 2, 1], self.task([1, 2, 3])).passed

    def test_unordered_lists_ignore_position(self):
        assert StructuralGrader(ignore_order=True).grade([3, 1, 2], self.task([1, 2, 3])).passed

    def test_partial_credit_on_ordered_lists(self):
        score = StructuralGrader(ignore_order=False).grade([1, 2, 9], self.task([1, 2, 3]))
        assert score.value == pytest.approx(2 / 3)

    def test_partial_credit_on_unordered_lists(self):
        score = StructuralGrader(ignore_order=True).grade([3, 1], self.task([1, 2, 3]))
        assert score.value == pytest.approx(2 / 3)
        assert "2/3 items matched" in score.detail

    def test_short_prediction_reports_missing_items(self):
        score = StructuralGrader(ignore_order=False).grade([1], self.task([1, 2, 3]))
        assert "missing" in score.detail and score.value == pytest.approx(1 / 3)

    def test_extra_items_do_not_earn_credit(self):
        score = StructuralGrader(ignore_order=True).grade([1, 2, 3, 4, 5], self.task([1, 2, 3]))
        assert score.passed

    def test_type_mismatch_is_named(self):
        score = StructuralGrader().grade('{"a": 1}', self.task([1, 2]))
        assert not score.passed and "expected array" in score.detail

    def test_nested_lists_of_objects(self):
        gold = {"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
        grader = StructuralGrader(ignore_order=False)
        assert grader.grade(gold, self.task(gold)).passed
        wrong = {"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "z"}]}
        score = grader.grade(wrong, self.task(gold))
        assert not score.passed and "items[1].name" in score.detail

    def test_empty_expected_list_is_a_pass(self):
        assert StructuralGrader(ignore_order=True).grade([], self.task([])).passed

    def test_partial_credit_can_be_switched_off(self):
        score = StructuralGrader(ignore_order=False, partial_credit=False).grade([1, 2, 9], self.task([1, 2, 3]))
        assert score.value == 0.0

    def test_empty_expected_list_inside_an_object(self):
        gold = {"items": [], "n": 1}
        assert StructuralGrader(ignore_order=True).grade(gold, self.task(gold)).passed
