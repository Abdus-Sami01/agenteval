"""Suite loading, reporting, caching, resume, sequential stopping, and the CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agenteval import (
    ExactMatchGrader,
    OutcomeGrader,
    PredictionCache,
    ResultStore,
    Step,
    SuiteError,
    SuiteFormatError,
    Task,
    TaskSuite,
    Trajectory,
    compare,
    evaluate,
    evaluate_many,
    evaluate_resumable,
    evaluate_sequential,
    failure_digest,
    leaderboard,
    leaderboard_tiers,
    leaderboard_to_html,
    load_csv,
    load_json,
    load_jsonl,
    load_run,
    load_suite,
    run_from_dict,
    run_from_json,
    run_to_dict,
    run_to_html,
    run_to_json,
    run_to_markdown,
    run_to_text,
    save_suite,
    suite_from_records,
    tag_breakdown,
    validate_suite,
    write_html,
)
from tests.helpers import adder, always_wrong, flaky, raises

ROOT = Path(__file__).resolve().parent.parent


class TestSuiteConstruction:
    def test_from_records(self):
        suite = suite_from_records("s", [{"id": "a", "input": "x", "expected": "y"}])
        assert len(suite) == 1
        assert suite.tasks[0].id == "a"

    def test_generated_ids(self):
        suite = suite_from_records("s", [{"input": "x"}, {"input": "y"}])
        assert [t.id for t in suite.tasks] == ["task_0", "task_1"]

    def test_unknown_keys_go_to_metadata(self):
        suite = suite_from_records("s", [{"id": "a", "input": "x", "difficulty": "hard"}])
        assert suite.tasks[0].metadata["difficulty"] == "hard"

    def test_missing_input_key_raises(self):
        with pytest.raises(SuiteError, match="missing required key"):
            suite_from_records("s", [{"id": "a"}])

    def test_non_mapping_record_raises(self):
        with pytest.raises(SuiteError, match="not a mapping"):
            suite_from_records("s", ["oops"])

    def test_filter_by_tag(self, math_suite):
        assert len(math_suite.filter(tags={"even"})) == 15

    def test_filter_by_id(self, math_suite):
        assert len(math_suite.filter(ids={"q1", "q2"})) == 2

    def test_sample_is_deterministic(self, math_suite):
        a = math_suite.sample(5, seed=1)
        b = math_suite.sample(5, seed=1)
        assert [t.id for t in a.tasks] == [t.id for t in b.tasks]

    def test_sample_larger_than_suite(self, math_suite):
        assert len(math_suite.sample(999, seed=1)) == 30

    def test_all_tags(self, math_suite):
        assert math_suite.all_tags == {"even", "odd"}


class TestSuiteValidation:
    def test_valid_suite(self, math_suite):
        assert validate_suite(math_suite) == []

    def test_empty_suite_flagged(self):
        assert "no tasks" in validate_suite(TaskSuite(name="e"))[0]

    def test_duplicate_ids_flagged(self):
        suite = TaskSuite(name="d", tasks=[
            Task(id="same", input="a"), Task(id="same", input="b"),
        ])
        assert any("duplicate" in p for p in validate_suite(suite))

    def test_non_positive_weight_flagged(self):
        suite = TaskSuite(name="w", tasks=[Task(id="a", input="x", weight=0.0)])
        assert any("weight" in p for p in validate_suite(suite))


class TestSuiteLoading:
    def test_jsonl(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text('{"id":"a","input":"x","expected":"y"}\n\n{"id":"b","input":"p","expected":"q"}\n',
                        encoding="utf-8")
        suite = load_jsonl(str(path))
        assert len(suite) == 2, "blank lines should be skipped"
        assert suite.name == "s"

    def test_jsonl_reports_bad_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id":"a","input":"x"}\n{not json}\n', encoding="utf-8")
        with pytest.raises(SuiteFormatError) as excinfo:
            load_jsonl(str(path))
        assert excinfo.value.line == 2
        assert "bad.jsonl:2" in str(excinfo.value)

    def test_json_object_form(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({
            "name": "named", "description": "d",
            "tasks": [{"id": "a", "input": "x"}],
        }), encoding="utf-8")
        suite = load_json(str(path))
        assert suite.name == "named" and suite.description == "d"

    def test_json_array_form(self, tmp_path):
        path = tmp_path / "arr.json"
        path.write_text(json.dumps([{"id": "a", "input": "x"}]), encoding="utf-8")
        assert len(load_json(str(path))) == 1

    def test_csv_with_quoting_and_tags(self, tmp_path):
        path = tmp_path / "s.csv"
        path.write_text('id,input,expected,tags\nc1,"multi\nline","a,b",easy;hard\n',
                        encoding="utf-8", newline="")
        suite = load_csv(str(path))
        assert suite.tasks[0].input == "multi\nline"
        assert suite.tasks[0].tags == ("easy", "hard")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "s.txt"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(SuiteFormatError, match="unsupported format"):
            load_suite(str(path))

    def test_unicode_round_trip(self, tmp_path):
        path = tmp_path / "u.jsonl"
        path.write_text(json.dumps({"id": "u", "input": "café ☕", "expected": "日本語"},
                                   ensure_ascii=False) + "\n", encoding="utf-8")
        assert load_jsonl(str(path)).tasks[0].expected == "日本語"

    def test_crlf_line_endings(self, tmp_path):
        path = tmp_path / "crlf.jsonl"
        path.write_bytes(b'{"id":"a","input":"x"}\r\n{"id":"b","input":"y"}\r\n')
        assert len(load_jsonl(str(path))) == 2

    def test_save_and_reload(self, tmp_path, math_suite):
        path = tmp_path / "out.json"
        save_suite(math_suite, str(path))
        assert len(load_suite(str(path))) == 30

    def test_load_suite_dispatch(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text('{"id":"a","input":"x"}\n', encoding="utf-8")
        assert len(load_suite(str(path))) == 1


class TestReporting:
    def test_text_report(self, perfect_run):
        text = run_to_text(perfect_run)
        assert "pass rate" in text and "95% CI" in text

    def test_text_report_lists_failures(self, failing_run):
        assert "failures" in run_to_text(failing_run)

    def test_markdown_report(self, perfect_run):
        assert "| metric |" in run_to_markdown(perfect_run)

    def test_json_report_is_valid(self, perfect_run):
        payload = json.loads(run_to_json(perfect_run))
        assert payload["summary"]["pass_rate"] == 1.0
        assert len(payload["results"]) == 30

    def test_dict_report_includes_ci(self, perfect_run):
        assert "pass_rate_ci" in run_to_dict(perfect_run)["summary"]

    def test_predictions_can_be_excluded(self, perfect_run):
        payload = run_to_dict(perfect_run, include_predictions=False)
        assert "prediction" not in payload["results"][0]

    def test_tag_breakdown_has_intervals(self, perfect_run):
        assert "95% CI" in tag_breakdown(perfect_run)

    def test_leaderboard_flags_overlap(self, math_suite):
        runs = evaluate_many({"a": adder, "b": always_wrong}, math_suite, ExactMatchGrader())
        board = leaderboard(runs)
        assert "non-overlapping" in board or "overlapping" in board

    def test_leaderboard_tiers_group_ties(self, math_suite):
        runs = evaluate_many(
            {"x": flaky(0.9, "1"), "y": flaky(0.9, "2"), "z": always_wrong},
            math_suite, ExactMatchGrader(), seed=1,
        )
        tiers = leaderboard_tiers(runs)
        assert "Tier 1" in tiers and "Tier 2" in tiers

    def test_failure_digest_groups_causes(self, failing_run):
        digest = failure_digest(failing_run)
        assert "failures" in digest

    def test_failure_digest_when_clean(self, perfect_run):
        assert "No failures" in failure_digest(perfect_run)

    def test_html_is_self_contained(self, failing_run):
        html = run_to_html(failing_run)
        assert '<section class="ae">' in html
        assert "prefers-color-scheme" in html
        assert "http://" not in html and "https://" not in html

    def test_html_escapes_content(self):
        suite = suite_from_records("x", [{"id": "<script>", "input": "1+1", "expected": "2"}])
        html = run_to_html(evaluate(always_wrong, suite, ExactMatchGrader()))
        assert "<script>" not in html.replace('<section class="ae">', "")
        assert "&lt;script&gt;" in html

    def test_html_warns_on_small_sample(self, small_suite):
        html = run_to_html(evaluate(adder, small_suite, ExactMatchGrader()))
        assert "Fewer than 30" in html

    def test_leaderboard_html(self, math_suite):
        runs = evaluate_many({"a": adder, "b": always_wrong}, math_suite, ExactMatchGrader())
        assert "<table>" in leaderboard_to_html(runs)

    def test_write_html_document(self, tmp_path, perfect_run):
        path = tmp_path / "r.html"
        write_html(run_to_html(perfect_run), str(path))
        assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


class TestPredictionCache:
    def test_prevents_re_execution(self, math_suite):
        calls = {"n": 0}

        def counted(task):
            calls["n"] += 1
            return adder(task)

        cache = PredictionCache()
        evaluate(cache.wrap(counted, "sys"), math_suite, ExactMatchGrader())
        evaluate(cache.wrap(counted, "sys"), math_suite, ExactMatchGrader())
        assert calls["n"] == 30

    def test_persistence_round_trip(self, tmp_path, small_suite):
        path = str(tmp_path / "c.json")
        cache = PredictionCache(path=path)
        evaluate(cache.wrap(adder, "sys"), small_suite, ExactMatchGrader())
        cache.save()
        assert PredictionCache(path=path).size == 3

    def test_version_bump_invalidates(self, tmp_path, small_suite):
        path = str(tmp_path / "c.json")
        cache = PredictionCache(path=path, version="v1")
        evaluate(cache.wrap(adder, "sys"), small_suite, ExactMatchGrader())
        cache.save()
        assert PredictionCache(path=path, version="v2").size == 0

    def test_corrupt_file_is_ignored(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{broken", encoding="utf-8")
        assert PredictionCache(path=str(path)).size == 0

    def test_different_systems_do_not_collide(self, small_suite):
        cache = PredictionCache()
        task = small_suite.tasks[0]
        cache.put("a", task, "from-a")
        assert cache.get("b", task) is None

    def test_stats(self, small_suite):
        cache = PredictionCache()
        evaluate(cache.wrap(adder, "s"), small_suite, ExactMatchGrader())
        evaluate(cache.wrap(adder, "s"), small_suite, ExactMatchGrader())
        assert cache.stats["hits"] == 3


class TestResume:
    def test_reuses_stored_results(self, tmp_path, math_suite):
        store = str(tmp_path / "run.jsonl")
        run, reused = evaluate_resumable(adder, math_suite, ExactMatchGrader(), store)
        assert len(run) == 30 and reused == 0

        run2, reused2 = evaluate_resumable(adder, math_suite, ExactMatchGrader(), store)
        assert reused2 == 30
        assert run2.pass_rate == run.pass_rate

    def test_survives_a_crash(self, tmp_path, math_suite):
        store = str(tmp_path / "run.jsonl")
        calls = {"n": 0}

        def crashes_partway(task):
            calls["n"] += 1
            if calls["n"] == 8:
                raise KeyboardInterrupt("simulated crash")
            return adder(task)

        with pytest.raises(KeyboardInterrupt):
            evaluate_resumable(crashes_partway, math_suite, ExactMatchGrader(), store)

        assert len(ResultStore(store)) == 7, "completed work must survive"

        run, reused = evaluate_resumable(adder, math_suite, ExactMatchGrader(), store)
        assert reused == 7
        assert len(run) == 30

    def test_fresh_discards_store(self, tmp_path, small_suite):
        store = str(tmp_path / "run.jsonl")
        evaluate_resumable(adder, small_suite, ExactMatchGrader(), store)
        _, reused = evaluate_resumable(adder, small_suite, ExactMatchGrader(), store, fresh=True)
        assert reused == 0

    def test_store_roundtrips_scores(self, tmp_path, small_suite):
        store = str(tmp_path / "run.jsonl")
        evaluate_resumable(adder, small_suite, ExactMatchGrader(), store)
        restored = ResultStore(store)
        result = restored.get("a")
        assert result is not None and result.is_pass and result.score is not None

    def test_store_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text('{"task_id":"a","outcome":"pass"}\nnot json\n', encoding="utf-8")
        assert len(ResultStore(str(path))) == 1

    def test_compact(self, tmp_path, small_suite):
        store_path = str(tmp_path / "run.jsonl")
        evaluate_resumable(adder, small_suite, ExactMatchGrader(), store_path)
        store = ResultStore(store_path)
        store.compact()
        assert len(ResultStore(store_path)) == 3


class TestSequential:
    def test_stops_early_when_clearly_above(self, math_suite):
        result = evaluate_sequential(adder, math_suite, ExactMatchGrader(),
                                     threshold=0.5, min_samples=10, seed=1)
        assert result.decision.stop
        assert result.evaluated < result.total_available
        assert result.saved > 0

    def test_stops_early_when_clearly_below(self, math_suite):
        result = evaluate_sequential(always_wrong, math_suite, ExactMatchGrader(),
                                     threshold=0.9, min_samples=10, seed=1)
        assert result.decision.stop
        assert result.saved > 0

    def test_respects_minimum_samples(self, math_suite):
        result = evaluate_sequential(adder, math_suite, ExactMatchGrader(),
                                     threshold=0.5, min_samples=25, seed=1)
        assert result.evaluated >= 25

    def test_summary_mentions_decision(self, math_suite):
        result = evaluate_sequential(adder, math_suite, ExactMatchGrader(),
                                     threshold=0.5, min_samples=10, seed=1)
        assert "decision" in result.summary()


class TestCLI:
    def test_graders_command(self, capsys):
        from agenteval.cli import main

        assert main(["graders"]) == 0
        assert "exact" in capsys.readouterr().out

    def test_power_command(self, capsys):
        from agenteval.cli import main

        assert main(["power", "--baseline", "0.7", "--delta", "0.05"]) == 0
        assert "required tasks" in capsys.readouterr().out

    def test_validate_command(self, tmp_path, capsys):
        from agenteval.cli import main

        path = tmp_path / "s.jsonl"
        path.write_text('{"id":"a","input":"x","expected":"y"}\n', encoding="utf-8")
        assert main(["validate", str(path)]) == 0
        assert "valid" in capsys.readouterr().out

    def test_validate_rejects_duplicates(self, tmp_path, capsys):
        from agenteval.cli import main

        path = tmp_path / "dup.jsonl"
        path.write_text('{"id":"a","input":"x"}\n{"id":"a","input":"y"}\n', encoding="utf-8")
        assert main(["validate", str(path)]) == 1
        assert "duplicate" in capsys.readouterr().out

    def test_describe_command(self, tmp_path, capsys):
        from agenteval.cli import main

        path = tmp_path / "s.jsonl"
        path.write_text('{"id":"a","input":"x","expected":"y","tags":["t"]}\n', encoding="utf-8")
        assert main(["describe", str(path)]) == 0
        out = capsys.readouterr().out
        assert "Tasks:" in out and "t" in out

    def test_missing_file_returns_error_code(self, capsys):
        from agenteval.cli import main

        assert main(["validate", "does_not_exist.jsonl"]) == 2

    def test_run_command_end_to_end(self, tmp_path, capsys):
        from agenteval.cli import main

        suite = tmp_path / "s.jsonl"
        suite.write_text(
            "\n".join(json.dumps({"id": f"q{i}", "input": f"{i}+{i}", "expected": str(i * 2)})
                      for i in range(1, 11)),
            encoding="utf-8",
        )
        out = tmp_path / "run.json"
        code = main(["run", str(suite), "--system", "tests.helpers:adder", "--out", str(out)])
        assert code == 0
        assert json.loads(out.read_text(encoding="utf-8"))["summary"]["pass_rate"] == 1.0

    def test_run_gate_fails_the_build(self, tmp_path):
        from agenteval.cli import main

        suite = tmp_path / "s.jsonl"
        suite.write_text('{"id":"a","input":"1+1","expected":"2"}\n', encoding="utf-8")
        code = main(["run", str(suite), "--system", "tests.helpers:always_wrong",
                     "--min-pass-rate", "0.9"])
        assert code == 1, "a failing gate must exit non-zero for CI"

    def test_run_writes_html(self, tmp_path):
        from agenteval.cli import main

        suite = tmp_path / "s.jsonl"
        suite.write_text('{"id":"a","input":"1+1","expected":"2"}\n', encoding="utf-8")
        html = tmp_path / "r.html"
        main(["run", str(suite), "--system", "tests.helpers:adder", "--html", str(html)])
        assert html.exists()

    def test_compare_command_detects_regression(self, tmp_path):
        from agenteval.cli import main

        suite = tmp_path / "s.jsonl"
        suite.write_text(
            "\n".join(json.dumps({"id": f"q{i}", "input": f"{i}+{i}", "expected": str(i * 2)})
                      for i in range(1, 31)),
            encoding="utf-8",
        )
        good = tmp_path / "good.json"
        bad = tmp_path / "bad.json"
        main(["run", str(suite), "--system", "tests.helpers:adder", "--out", str(good)])
        main(["run", str(suite), "--system", "tests.helpers:always_wrong", "--out", str(bad)])

        code = main(["compare", str(good), str(bad), "--iterations", "1000",
                     "--fail-on-regression"])
        assert code == 1, "a real regression must fail the build"

    def test_compare_command_blocks_a_tag_regression(self, tmp_path):
        from agenteval.cli import main

        suite = tmp_path / "s.jsonl"
        rows = [{"id": f"s{i}", "input": f"{i}+{i}", "expected": str(i * 2), "tags": ["safety"]}
                for i in range(1, 11)]
        rows += [{"id": f"e{i}", "input": f"{i}+{i}", "expected": str(i * 2), "tags": ["easy"]}
                 for i in range(11, 31)]
        suite.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        good = tmp_path / "good.json"
        partial = tmp_path / "partial.json"
        main(["run", str(suite), "--system", "tests.helpers:adder", "--out", str(good)])
        main(["run", str(suite), "--system", "tests.helpers:safety_regressor", "--out", str(partial)])

        assert main(["compare", str(good), str(partial), "--iterations", "500"]) == 0
        code = main(["compare", str(good), str(partial), "--iterations", "500",
                     "--max-tag-drop", "0.05"])
        assert code == 1, "a tag-level regression must fail the build"

    def test_compare_command_reports_untagged_runs_cleanly(self, tmp_path, capsys):
        from agenteval.cli import main

        suite = tmp_path / "s.jsonl"
        suite.write_text('{"id":"a","input":"1+1","expected":"2"}\n', encoding="utf-8")
        out = tmp_path / "r.json"
        main(["run", str(suite), "--system", "tests.helpers:adder", "--out", str(out)])
        assert main(["compare", str(out), str(out), "--iterations", "200",
                     "--max-tag-drop", "0.05"]) == 0

    def test_compare_command_needs_overlapping_ids(self, tmp_path):
        from agenteval.cli import main

        left = tmp_path / "l.jsonl"
        right = tmp_path / "r.jsonl"
        left.write_text('{"id":"a","input":"1+1","expected":"2"}\n', encoding="utf-8")
        right.write_text('{"id":"z","input":"1+1","expected":"2"}\n', encoding="utf-8")
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        main(["run", str(left), "--system", "tests.helpers:adder", "--out", str(a)])
        main(["run", str(right), "--system", "tests.helpers:adder", "--out", str(b)])
        assert main(["compare", str(a), str(b)]) == 2

    def test_report_command_renders_a_saved_run(self, tmp_path, capsys):
        from agenteval.cli import main

        suite = tmp_path / "s.jsonl"
        suite.write_text('{"id":"a","input":"1+1","expected":"2","tags":["t"]}\n', encoding="utf-8")
        out = tmp_path / "r.json"
        main(["run", str(suite), "--system", "tests.helpers:adder", "--out", str(out)])
        capsys.readouterr()
        assert main(["report", str(out)]) == 0
        text = capsys.readouterr().out
        assert "pass rate" in text and "t" in text

    def test_load_object_requires_module_colon_attr(self):
        from agenteval.cli import load_object

        with pytest.raises(ValueError, match="module:attribute"):
            load_object("nocolon")

    def test_load_object_missing_attribute(self):
        from agenteval.cli import load_object

        with pytest.raises(AttributeError):
            load_object("tests.helpers:does_not_exist")


class TestRunRoundTrip:
    def build(self, tags=("x",), weight=1.0):
        suite = suite_from_records("m", [
            {"id": "a", "input": "1+1", "expected": "2", "tags": list(tags), "weight": weight},
            {"id": "b", "input": "2+2", "expected": "5"},
        ])
        return evaluate(adder, suite, ExactMatchGrader(), system_name="sys", notes="nightly")

    def test_outcomes_and_scores_survive(self):
        original = self.build()
        restored = run_from_json(run_to_json(original))
        assert [r.outcome for r in restored.results] == [r.outcome for r in original.results]
        assert restored.pass_rate == original.pass_rate
        assert restored.results[0].score.value == 1.0

    def test_metadata_survives(self):
        original = self.build()
        restored = run_from_dict(run_to_dict(original))
        assert restored.metadata.system_name == "sys"
        assert restored.metadata.suite_name == "m"
        assert restored.metadata.notes == "nightly"
        assert restored.metadata.run_id == original.metadata.run_id

    def test_tags_and_weights_survive(self):
        restored = run_from_dict(run_to_dict(self.build(tags=("safety", "hard"), weight=3.0)))
        assert restored.results[0].tags == ("safety", "hard")
        assert restored.results[0].weight == 3.0
        assert restored.mean_score == pytest.approx(0.75)

    def test_extra_metadata_is_preserved(self):
        original = self.build()
        original.metadata.extra["model"] = "gpt-4o"
        assert run_from_dict(run_to_dict(original)).metadata.extra["model"] == "gpt-4o"

    def test_errors_survive(self, small_suite):
        original = evaluate(raises, small_suite, ExactMatchGrader())
        restored = run_from_dict(run_to_dict(original))
        assert restored.errored == 3
        assert "model offline" in restored.results[0].error

    def test_comparison_of_restored_runs_matches(self, math_suite):
        base = evaluate(adder, math_suite, ExactMatchGrader(), system_name="base")
        cand = evaluate(always_wrong, math_suite, ExactMatchGrader(), system_name="cand")
        direct = compare(base, cand, iterations=500)
        restored = compare(run_from_dict(run_to_dict(base)), run_from_dict(run_to_dict(cand)),
                           iterations=500)
        assert restored.delta == direct.delta
        assert restored.broken == direct.broken

    def test_load_run_reads_a_file(self, tmp_path):
        path = tmp_path / "run.json"
        path.write_text(run_to_json(self.build()), encoding="utf-8")
        assert load_run(str(path)).metadata.system_name == "sys"

    def test_unknown_outcome_degrades_to_error(self):
        payload = run_to_dict(self.build())
        payload["results"][0]["outcome"] = "exploded"
        assert run_from_dict(payload).results[0].outcome.value == "error"

    def test_result_without_task_id_is_rejected(self):
        payload = run_to_dict(self.build())
        del payload["results"][0]["task_id"]
        with pytest.raises(SuiteFormatError):
            run_from_dict(payload)

    def test_empty_payload_gives_an_empty_run(self):
        assert len(run_from_dict({})) == 0


class TestResumeHardening:
    def test_parallel_resume_persists_every_result(self, tmp_path, math_suite):
        store = str(tmp_path / "run.jsonl")
        run, _ = evaluate_resumable(adder, math_suite, ExactMatchGrader(), store, max_parallel=8)
        assert len(run) == 30
        assert len(ResultStore(store)) == 30

    def test_parallel_resume_keeps_suite_order(self, tmp_path, math_suite):
        store = str(tmp_path / "run.jsonl")
        run, _ = evaluate_resumable(adder, math_suite, ExactMatchGrader(), store, max_parallel=8)
        assert [r.task_id for r in run.results] == [t.id for t in math_suite.tasks]

    def test_parallel_resume_reuses_prior_work(self, tmp_path, math_suite):
        store = str(tmp_path / "run.jsonl")
        half = TaskSuite(name="math", tasks=math_suite.tasks[:10])
        evaluate_resumable(adder, half, ExactMatchGrader(), store, max_parallel=4)
        _, reused = evaluate_resumable(adder, math_suite, ExactMatchGrader(), store, max_parallel=4)
        assert reused == 10

    def test_run_id_is_stable_across_processes(self, tmp_path, small_suite):
        store = str(tmp_path / "run.jsonl")
        first, _ = evaluate_resumable(adder, small_suite, ExactMatchGrader(), store)
        second, _ = evaluate_resumable(adder, small_suite, ExactMatchGrader(), store)
        assert first.metadata.run_id == second.metadata.run_id

        expected = subprocess.run(
            [sys.executable, "-c",
             f"from agenteval.resume import _stable_id; print(_stable_id({store!r}))"],
            capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "1"},
            cwd=str(ROOT),
        ).stdout.strip()
        assert first.metadata.run_id.endswith(expected)

    def test_truncated_final_line_is_skipped(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text(
            '{"task_id":"a","outcome":"pass"}\n{"task_id":"b","outcome":"pa',
            encoding="utf-8",
        )
        store = ResultStore(str(path))
        assert store.completed_ids == {"a"}

    def test_a_corrupt_store_does_not_lose_later_lines(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text(
            '{"task_id":"a","outcome":"pass"}\n[]\n{"task_id":"c","outcome":"fail"}\n',
            encoding="utf-8",
        )
        assert ResultStore(str(path)).completed_ids == {"a", "c"}

    def test_resuming_after_corruption_reruns_the_lost_task(self, tmp_path, small_suite):
        store_path = str(tmp_path / "run.jsonl")
        evaluate_resumable(adder, small_suite, ExactMatchGrader(), store_path)
        lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
        (tmp_path / "run.jsonl").write_text("\n".join(lines[:-1] + ["garbage"]) + "\n",
                                            encoding="utf-8")
        run, reused = evaluate_resumable(adder, small_suite, ExactMatchGrader(), store_path)
        assert reused == 2 and len(run) == 3

    def test_later_record_wins_for_a_repeated_task(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text(
            '{"task_id":"a","outcome":"fail"}\n{"task_id":"a","outcome":"pass"}\n',
            encoding="utf-8",
        )
        store = ResultStore(str(path))
        assert len(store) == 1 and store.get("a").is_pass

    def test_compact_removes_superseded_records(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text(
            '{"task_id":"a","outcome":"fail"}\n{"task_id":"a","outcome":"pass"}\n',
            encoding="utf-8",
        )
        store = ResultStore(str(path))
        store.compact()
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1
        assert ResultStore(str(path)).get("a").is_pass

    def test_store_creates_missing_directories(self, tmp_path, small_suite):
        store = str(tmp_path / "nested" / "deeper" / "run.jsonl")
        run, _ = evaluate_resumable(adder, small_suite, ExactMatchGrader(), store)
        assert len(run) == 3 and os.path.exists(store)

    def test_trajectory_predictions_keep_their_steps(self, tmp_path, small_suite):
        store = str(tmp_path / "run.jsonl")

        def agent(task):
            return Trajectory(steps=[Step(action="add")], output=adder(task))

        evaluate_resumable(agent, small_suite, OutcomeGrader(ExactMatchGrader()), store)
        stored = json.loads((tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert stored["prediction"]["steps"] == [{"action": "add"}]


class TestRecordCoercion:
    def test_a_string_tag_is_one_tag_not_six_characters(self):
        suite = suite_from_records("s", [{"id": "a", "input": "x", "tags": "safety"}])
        assert suite.tasks[0].tags == ("safety",)

    def test_delimited_string_tags_are_split(self):
        suite = suite_from_records("s", [
            {"id": "a", "input": "x", "tags": "safety, hard"},
            {"id": "b", "input": "x", "tags": "safety;hard"},
        ])
        assert suite.tasks[0].tags == ("safety", "hard")
        assert suite.tasks[1].tags == ("safety", "hard")

    def test_list_tags_are_normalized(self):
        suite = suite_from_records("s", [{"id": "a", "input": "x", "tags": [" safety ", "", "hard"]}])
        assert suite.tasks[0].tags == ("safety", "hard")

    def test_missing_and_empty_tags(self):
        suite = suite_from_records("s", [
            {"id": "a", "input": "x"},
            {"id": "b", "input": "x", "tags": ""},
            {"id": "c", "input": "x", "tags": None},
        ])
        assert all(t.tags == () for t in suite.tasks)

    def test_non_iterable_tags_are_rejected(self):
        with pytest.raises(SuiteError):
            suite_from_records("s", [{"id": "a", "input": "x", "tags": {"safety": True}}])

    def test_blank_weight_defaults_to_one(self):
        suite = suite_from_records("s", [{"id": "a", "input": "x", "weight": ""}])
        assert suite.tasks[0].weight == 1.0

    def test_numeric_string_weight_is_parsed(self):
        assert suite_from_records("s", [{"id": "a", "input": "x", "weight": "2.5"}]).tasks[0].weight == 2.5

    def test_non_numeric_weight_names_the_task(self):
        with pytest.raises(SuiteError) as excinfo:
            suite_from_records("s", [{"id": "a", "input": "x", "weight": "heavy"}])
        assert "'a'" in str(excinfo.value)

    def test_csv_with_blank_weight_column(self, tmp_path):
        path = tmp_path / "s.csv"
        path.write_text("id,input,expected,weight\na,1+1,2,\nb,2+2,4,3\n", encoding="utf-8")
        suite = load_csv(str(path))
        assert [t.weight for t in suite.tasks] == [1.0, 3.0]

    def test_yaml_suite_with_string_tags(self, tmp_path):
        path = tmp_path / "s.yaml"
        path.write_text(
            "name: y\ndescription: d\ntasks:\n  - id: a\n    input: 1+1\n    expected: '2'\n    tags: math\n",
            encoding="utf-8",
        )
        suite = load_suite(str(path))
        assert suite.name == "y" and suite.description == "d"
        assert suite.tasks[0].tags == ("math",)

    def test_yaml_list_form(self, tmp_path):
        path = tmp_path / "plain.yaml"
        path.write_text("- id: a\n  input: 1+1\n  expected: '2'\n", encoding="utf-8")
        suite = load_suite(str(path))
        assert len(suite) == 1 and suite.name == "plain"

    def test_empty_yaml_is_an_empty_suite(self, tmp_path):
        path = tmp_path / "empty.yml"
        path.write_text("", encoding="utf-8")
        assert len(load_suite(str(path))) == 0

    def test_save_and_reload_preserves_tags_and_weights(self, tmp_path):
        suite = suite_from_records("s", [
            {"id": "a", "input": "x", "expected": "y", "tags": "safety", "weight": 2.0, "note": "keep"},
        ])
        path = str(tmp_path / "out.json")
        save_suite(suite, path)
        restored = load_suite(path)
        assert restored.tasks[0].tags == ("safety",)
        assert restored.tasks[0].weight == 2.0
        assert restored.tasks[0].metadata["note"] == "keep"
