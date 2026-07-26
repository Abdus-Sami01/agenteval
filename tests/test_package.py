"""Public surface, packaging, exceptions, logging, and progress reporting."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pytest

import agenteval
from agenteval import (
    AgentEvalError,
    BudgetExceeded,
    BudgetExceededError,
    ConfigurationError,
    ExactMatchGrader,
    GraderError,
    PairedLengthError,
    ProgressReporter,
    StatisticsError,
    SuiteError,
    SuiteFormatError,
    UnknownGraderError,
    configure_logging,
    evaluate,
)
from tests.helpers import adder

ROOT = Path(__file__).resolve().parent.parent


class TestPublicAPI:
    def test_every_export_resolves(self):
        missing = [name for name in agenteval.__all__ if not hasattr(agenteval, name)]
        assert missing == []

    def test_no_duplicate_exports(self):
        duplicates = [n for n in set(agenteval.__all__) if agenteval.__all__.count(n) > 1]
        assert duplicates == []

    def test_exports_are_sorted(self):
        expected = sorted(set(agenteval.__all__), key=lambda s: (s.lstrip("_").lower(), s))
        assert agenteval.__all__ == expected

    def test_version_is_semver(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", agenteval.__version__)

    def test_version_info_matches_version(self):
        assert ".".join(str(p) for p in agenteval.__version_info__) == agenteval.__version__

    def test_version_matches_pyproject(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        declared = re.search(r'^version = "([^"]+)"', text, re.M).group(1)
        assert declared == agenteval.__version__

    def test_changelog_documents_current_version(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert f"[{agenteval.__version__}]" in changelog

    def test_py_typed_marker_present(self):
        assert (ROOT / "agenteval" / "py.typed").exists()

    def test_module_has_docstring(self):
        assert agenteval.__doc__ and "evaluation harness" in agenteval.__doc__.lower()

    def test_core_names_are_exported(self):
        for name in ("evaluate", "compare", "gate", "ExactMatchGrader", "wilson_interval"):
            assert name in agenteval.__all__


class TestExceptions:
    @pytest.mark.parametrize("exc", [
        SuiteError, SuiteFormatError, GraderError, UnknownGraderError,
        StatisticsError, PairedLengthError, ConfigurationError, BudgetExceededError,
    ])
    def test_all_derive_from_base(self, exc):
        assert issubclass(exc, AgentEvalError)

    def test_base_is_an_exception(self):
        assert issubclass(AgentEvalError, Exception)

    def test_budget_alias_is_the_same_class(self):
        assert BudgetExceeded is BudgetExceededError

    def test_suite_format_error_carries_location(self):
        err = SuiteFormatError("f.jsonl", "bad", line=7)
        assert err.path == "f.jsonl" and err.line == 7
        assert "f.jsonl:7" in str(err)

    def test_suite_format_error_without_line(self):
        err = SuiteFormatError("f.jsonl", "bad")
        assert err.line is None
        assert not re.search(r"f\.jsonl:\d+", str(err)), "no line number when none was given"
        assert str(err) == "f.jsonl: bad"

    def test_unknown_grader_lists_alternatives(self):
        err = UnknownGraderError("nope", {"exact", "f1"})
        assert "exact" in str(err) and "f1" in str(err)

    def test_paired_length_error_reports_both_sizes(self):
        err = PairedLengthError(3, 5, "my context")
        assert "3 and 5" in str(err) and "my context" in str(err)

    def test_catching_base_catches_specifics(self):
        with pytest.raises(AgentEvalError):
            raise SuiteFormatError("x", "y")


class TestLogging:
    def test_null_handler_attached_on_import(self):
        logger = logging.getLogger("agenteval")
        assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)

    def test_configure_logging_emits_records(self, small_suite):
        stream = io.StringIO()
        logger = logging.getLogger("agenteval")
        original = list(logger.handlers)
        try:
            configure_logging(logging.INFO, stream=stream)
            evaluate(adder, small_suite, ExactMatchGrader())
            assert "evaluated" in stream.getvalue()
        finally:
            logger.handlers.clear()
            logger.handlers.extend(original)
            logger.propagate = True

    def test_library_is_quiet_by_default(self, small_suite, capsys):
        evaluate(adder, small_suite, ExactMatchGrader())
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""


class TestProgressReporter:
    def test_silent_when_not_a_tty(self, small_suite):
        stream = io.StringIO()
        reporter = ProgressReporter(total=3, stream=stream, enabled=False)
        run = evaluate(adder, small_suite, ExactMatchGrader(), on_result=reporter.callback)
        reporter.finish()
        assert stream.getvalue() == ""
        assert len(run) == 3

    def test_renders_when_enabled(self, small_suite):
        stream = io.StringIO()
        reporter = ProgressReporter(total=3, stream=stream, enabled=True, min_interval_s=0)
        run = evaluate(adder, small_suite, ExactMatchGrader())
        for result in run.results:
            reporter.update(result)
        reporter.finish()
        output = stream.getvalue()
        assert "3/3" in output and "pass" in output

    def test_counts_outcomes(self, small_suite):
        stream = io.StringIO()
        reporter = ProgressReporter(total=3, stream=stream, enabled=True, min_interval_s=0)
        run = evaluate(adder, small_suite, ExactMatchGrader())
        for result in run.results:
            reporter.update(result)
        reporter.finish()
        assert "100.0%" in stream.getvalue()

    def test_context_manager_finishes(self, small_suite):
        stream = io.StringIO()
        with ProgressReporter(total=1, stream=stream, enabled=True, min_interval_s=0) as reporter:
            run = evaluate(adder, small_suite, ExactMatchGrader())
            reporter.update(run.results[0])
        assert stream.getvalue()

    def test_evaluate_progress_flag(self, small_suite):
        run = evaluate(adder, small_suite, ExactMatchGrader(), progress=True)
        assert len(run) == 3

    def test_progress_preserves_user_callback(self, small_suite):
        seen = []
        evaluate(adder, small_suite, ExactMatchGrader(), progress=True, on_result=seen.append)
        assert len(seen) == 3

    def test_broken_stream_disables_itself(self, small_suite):
        class Broken(io.StringIO):
            def write(self, s):
                raise OSError("pipe closed")

        reporter = ProgressReporter(total=1, stream=Broken(), enabled=True, min_interval_s=0)
        run = evaluate(adder, small_suite, ExactMatchGrader())
        reporter.update(run.results[0])
        reporter.finish()


class TestReproducibility:
    def test_same_seed_gives_same_results(self, math_suite):
        a = evaluate(adder, math_suite, ExactMatchGrader(), seed=99)
        b = evaluate(adder, math_suite, ExactMatchGrader(), seed=99)
        assert [r.numeric for r in a.results] == [r.numeric for r in b.results]

    def test_metadata_records_environment(self, small_suite):
        meta = evaluate(adder, small_suite, ExactMatchGrader(), seed=7).metadata
        payload = meta.as_dict()
        assert payload["seed"] == 7
        assert payload["python"] and payload["platform"]

    def test_notes_are_carried(self, small_suite):
        run = evaluate(adder, small_suite, ExactMatchGrader(), notes="ablation A")
        assert run.metadata.notes == "ablation A"
