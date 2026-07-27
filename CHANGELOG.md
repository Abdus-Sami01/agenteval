# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Agent trajectories, plus throughput and failure handling for evaluations that
call real APIs.

### Added
- `Trajectory` and `Step`: a system under test can now return the steps it took
  rather than only a final answer, so tool use, step count, spend, and failed
  calls are all gradable. `Trajectory.from_records()` accepts the common step
  dict shapes (`tool`/`action`, `input`/`args`, `output`/`observation`).
- Three graders for that: `OutcomeGrader` (adapts any existing grader to a
  trajectory's final output), `ToolSequenceGrader` (`exact`, `subsequence`, or
  `set` matching, partial credit, forbidden-tool list), and `StepBudgetGrader`
  (step, cost, and failed-call budgets, decaying to zero at twice the limit).
  Compose them with `WeightedGrader` to score outcome and process together.
- Trajectories serialize into the JSON report with their steps intact, so a
  change in tool use shows up in a report diff.
- `evaluate_async()` for coroutine systems: concurrency is bounded by an
  asyncio semaphore, so thousands of in-flight calls cost coroutines instead
  of OS threads.
- `RetryPolicy` with geometric backoff, jitter, a delay ceiling, and a
  `retry_on` predicate so attempts are not spent on errors that cannot recover.
  Accepted by `evaluate()`, `evaluate_async()`, and `iter_evaluate()`.
- `RateLimiter`, a token bucket shared across workers, capping calls per second
  independently of `max_parallel`.
- `compare_by_tag()` and `tag_regression_gate()`: an aggregate that improves can
  hide a slice that got worse, so the same paired comparison now runs inside
  each tag. Tags below `min_tasks` paired tasks are reported but never
  enforced, since a one-task swing in a tiny slice is noise.
- `run_from_dict()`, `run_from_json()`, and `load_run()`: saved runs load back
  into a real `EvalRun`, so a later job can compare, gate, and report on runs it
  did not produce. Outcomes, scores, subscores, tags, and weights round-trip
  exactly.
- `agenteval run --rate-limit` and `--retry-backoff`;
  `agenteval compare --max-tag-drop` and `--min-tag-tasks`.

### Fixed
- `timeout_s` no longer waits on the worker it just abandoned: the executor is
  shut down without joining, so a system that ignores its timeout strands one
  thread instead of stalling the evaluation.
- Parallel runs deliver `on_result` callbacks as tasks complete rather than in
  one batch at the end, so `progress=True` tracks a live run.
- `StructuralGrader(ignore_order=True)` scored an expected empty list as a total
  miss instead of a match, which dragged down any object containing one.
- The no-jsonschema fallback in `JSONSchemaGrader` accepted `True` where an
  integer or number was required, and never looked inside `properties`.
- `agenteval compare` and `agenteval report` reimplemented the comparison and
  summary logic against raw JSON, so they had drifted from the library: no
  McNemar test, no effect size, no per-tag view. Both now load the run and call
  the same functions the Python API does.

### Changed
- `evaluate()` rejects `max_parallel < 1` and negative `retries` with
  `ConfigurationError` instead of silently running sequentially.

## [0.2.0] - 2026-07-26

Packaging and hardening pass. First release intended for public use.

### Added
- pytest suite: 451 tests at 90% coverage, run on Linux, macOS, and Windows
  across Python 3.10-3.13. Statistical routines are verified against
  hand-computed closed forms rather than recorded output.
- `MANIFEST.in` so the sdist carries tests, fixtures, examples, and the
  changelog; `pip install .[dev] && pytest` works from an unpacked sdist.
- `dev` extra with pytest, pytest-cov, and ruff.
- Typed exception hierarchy rooted at `AgentEvalError` (`SuiteError`,
  `SuiteFormatError`, `GraderError`, `UnknownGraderError`, `StatisticsError`,
  `PairedLengthError`, `ConfigurationError`, `BudgetExceededError`).
- `iter_evaluate()` streams results one task at a time so peak memory stays
  flat on suites too large to hold in RAM.
- `ProgressReporter` and `evaluate(progress=True)` for live progress on long
  runs. Writes to stderr and disables itself when not attached to a TTY.
- `configure_logging()`; the library logs through the `agenteval` logger and
  attaches a `NullHandler` on import.
- `py.typed` marker so downstream users get type information (PEP 561).
- `__version__` and `__version_info__`.
- Packaging metadata: classifiers, keywords, project URLs, extras
  (`yaml`, `schema`, `all`), and a `LICENSE` file.
- README and this changelog.

### Changed
- `BudgetExceeded` is now an alias of `BudgetExceededError`; there is one
  canonical budget exception instead of two unrelated classes.
- `__all__` is sorted and verified complete.
- Library code raises typed errors instead of bare `ValueError` / `ImportError`.

## [0.1.0] - 2026-07-26

Initial implementation.

### Added
- 15 graders: exact, contains, regex, edit distance, token F1, numeric,
  range, set, JSON Schema, structural, predicate, callable, rubric,
  LLM judge, and weighted composition.
- Statistics: Wilson intervals, bootstrap and BCa intervals, paired bootstrap,
  paired permutation test, McNemar's exact test, Cohen's d, Cliff's delta,
  power analysis and minimum detectable effect. Pure Python, no SciPy.
- `compare()` with `IMPROVEMENT` / `REGRESSION` / `INCONCLUSIVE` verdicts, plus
  quality gates and regression gates for CI.
- `compare_all()` with Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg
  corrections for multi-system comparisons.
- `validate_judge()` and Cohen's kappa for checking whether an LLM judge
  agrees with reference labels well enough to be trusted.
- Calibration: ECE, MCE, Brier score, Brier skill score, log loss, and a
  text reliability diagram.
- `analyze_stability()` for run-to-run variance, flaky-task detection, ICC,
  and a variance-aware confidence interval.
- `detect_contamination()` and `find_duplicates()` via hashed n-gram overlap.
- `CostTracker` with pre-call budget enforcement and cost-per-passing-task.
- `evaluate_sequential()` early stopping with an alpha-spending bound.
- `PredictionCache` and `evaluate_resumable()` with fsynced, crash-safe results.
- Reporting: text, JSON, Markdown, and self-contained HTML; leaderboards with
  statistical tiers and per-tag confidence intervals.
- CLI: `run`, `compare`, `validate`, `describe`, `power`, `report`, `graders`.
- Suite loading from JSONL, JSON, CSV, and YAML.
