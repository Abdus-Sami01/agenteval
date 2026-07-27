# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Throughput and failure handling for evaluations that call real APIs.

### Added
- `evaluate_async()` for coroutine systems: concurrency is bounded by an
  asyncio semaphore, so thousands of in-flight calls cost coroutines instead
  of OS threads.
- `RetryPolicy` with geometric backoff, jitter, a delay ceiling, and a
  `retry_on` predicate so attempts are not spent on errors that cannot recover.
  Accepted by `evaluate()`, `evaluate_async()`, and `iter_evaluate()`.
- `RateLimiter`, a token bucket shared across workers, capping calls per second
  independently of `max_parallel`.
- `agenteval run --rate-limit` and `--retry-backoff`.

### Fixed
- `timeout_s` no longer waits on the worker it just abandoned: the executor is
  shut down without joining, so a system that ignores its timeout strands one
  thread instead of stalling the evaluation.
- Parallel runs deliver `on_result` callbacks as tasks complete rather than in
  one batch at the end, so `progress=True` tracks a live run.

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
