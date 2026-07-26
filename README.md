# agenteval

Evaluation harness for LLM systems and agents, with statistics that refuse to overclaim.

Most eval scripts print a single number. That number is usually a coin flip dressed up as a
measurement: no confidence interval, no correction for comparing five systems at once, no check
on whether the model is just nondeterministic, and no way to tell whether the "judge" grading
everything is any good. `agenteval` fixes that, and stays out of your way while doing it.

- **No required dependencies.** Pure Python, including the statistics.
- **Honest by default.** Comparisons return `INCONCLUSIVE` when the evidence is thin.
- **Fits your system.** Anything callable that takes a `Task` and returns a prediction.

```bash
pip install agenteval
```

Optional extras: `pip install "agenteval[yaml]"` for YAML suites, `"agenteval[schema]"` for
full JSON-Schema grading, `"agenteval[all]"` for both.

---

## Quick start

```python
from agenteval import ExactMatchGrader, evaluate, run_to_text, suite_from_records

suite = suite_from_records("math", [
    {"id": "q1", "input": "2+2", "expected": "4", "tags": ["easy"]},
    {"id": "q2", "input": "13*7", "expected": "91", "tags": ["hard"]},
])

def my_system(task):
    return str(eval(task.input))          # your model / agent / pipeline goes here

run = evaluate(my_system, suite, ExactMatchGrader())
print(run_to_text(run))
```

```
  tasks      2
  passed     2
  pass rate  100.0%  (95% CI 34.2% - 100.0%)
```

That interval is the point. Two passing tasks is not evidence of a 100% system.

---

## Why the statistics matter

### Comparisons can come back undecided

```python
from agenteval import compare

result = compare(baseline_run, candidate_run)
print(result.summary())
```

```
  score delta         : -0.1333 [-0.2667, 0.0000]
  paired-permutation: stat=-0.1333, p=0.5015 (not significant)

  VERDICT: INCONCLUSIVE
  (confidence interval spans zero - not enough evidence to call it either way)
```

Two tasks broke, and the tool still refuses to call it a regression, because at this sample size
it cannot tell that apart from noise. Regression *gates* still fire on the concrete breakage, so
you can block a merge without pretending the aggregate difference is proven.

### Comparing many systems inflates false positives

Six pairwise tests at α=0.05 will hand you a "winner" by luck. `compare_all` corrects for it:

```python
from agenteval import compare_all

matrix = compare_all(runs, method="holm", baseline="gpt-4o")
print(matrix.summary())
```

Holm, Bonferroni, and Benjamini-Hochberg are all available. On p = [0.01, 0.02, 0.03, 0.04],
naive testing passes all four; Holm passes one.

### Rankings that admit ties

```python
from agenteval import leaderboard_tiers
print(leaderboard_tiers(runs))
```

```
  Tier 1: strong, good
      strong            98.0%  [93.0%, 99.4%]
      good              95.0%  [88.8%, 97.8%]
  Tier 2: weak, poor
```

`strong` beat `good` on raw score, but their intervals overlap, so no ranking is claimed.

### Nondeterministic systems need more than one run

```python
from agenteval import analyze_stability, repeat_evaluate

runs = repeat_evaluate(my_system, suite, grader, repeats=5)
print(analyze_stability(runs).summary())
```

```
  pass rates          42.5%, 52.5%, 45.0%, 47.5%, 55.0%
  spread (max-min)    12.5%
  single-run CI       [32.9%, 62.5%]   (assumes determinism)
  variance-aware CI   [32.0%, 63.9%]   (includes run-to-run noise)
```

It also separates always-pass, always-fail, and genuinely flaky tasks, ranked by entropy.

### Your LLM judge might be worthless

A judge that passes everything can still score 65% raw agreement. Cohen's kappa catches it:

```python
from agenteval import validate_judge

print(validate_judge(my_judge, labeled_examples).summary())
```

```
  raw agreement       65.0%
  Cohen's kappa       0.0000  (slight)
  WARNING: agreement is weak - scores from this judge should not be
  treated as ground truth without further validation.
  BIAS: judge passes +35.0% more often than the reference (lenient)
```

---

## Graders

15 built in, all sharing one interface:

| grader | use |
| --- | --- |
| `ExactMatchGrader` | normalized string equality |
| `ContainsGrader` | required substrings, with partial credit |
| `RegexGrader` | pattern match |
| `EditDistanceGrader` | fuzzy similarity threshold |
| `F1TokenGrader` | token overlap with precision/recall |
| `NumericGrader` | absolute or relative tolerance, extracts numbers from prose |
| `RangeGrader` | value within bounds |
| `SetGrader` | set F1, order-sensitive optional |
| `JSONSchemaGrader` | validate structure against JSON Schema |
| `StructuralGrader` | recursive object comparison with partial credit |
| `PredicateGrader` | any boolean function |
| `CallableGrader` | any scoring function |
| `RubricGrader` | weighted named criteria |
| `LLMJudgeGrader` | model-graded, with robust score parsing |
| `WeightedGrader` | combine several graders |

Custom graders subclass `Grader` and implement `grade(prediction, task) -> Score`.

---

## Running at scale

```python
# Stream results without holding the suite in memory
for result in iter_evaluate(system, huge_suite, grader):
    save_to_db(result)

# Survive a crash: each result is fsynced as it completes
run, reused = evaluate_resumable(system, suite, grader, "run.jsonl")

# Skip work you already paid for
cache = PredictionCache(path="preds.json")
run = evaluate(cache.wrap(system, "gpt-4o"), suite, grader)

# Stop as soon as the answer is clear
result = evaluate_sequential(system, suite, grader, threshold=0.7)
# -> "evaluated 26 of 200 tasks, skipped 174 (87%)"

# Cap spend, enforced before each call
tracker = CostTracker(budget=5.00)
run = evaluate(tracker.wrap(system, cost_fn=my_cost), suite, grader)
```

Contamination detection is included, because a benchmark score is meaningless if the test data
was in the training set:

```python
report = detect_contamination(suite, training_corpus, n_gram=8)
clean = clean_suite(suite, report)
```

---

## Command line

```bash
agenteval run suite.jsonl --system mypkg.models:gpt4 --out gpt4.json --html report.html
agenteval compare baseline.json candidate.json --fail-on-regression
agenteval power --baseline 0.7 --delta 0.05     # how many tasks do I actually need?
agenteval validate suite.jsonl
agenteval graders
```

`run --min-pass-rate 0.9` and `compare --fail-on-regression` exit non-zero, so both drop straight
into CI.

`--system` takes `module:function` and imports it, which executes that module. Point it only at
code you control. Nothing is auto-discovered.

---

## Suites

Load from JSONL, JSON, CSV, or YAML, or build them in Python:

```python
suite = load_suite("tasks.jsonl")
suite = suite.filter(tags={"hard"})
suite = suite.sample(100, seed=42)
```

```jsonl
{"id": "q1", "input": "2+2", "expected": "4", "tags": ["easy"], "weight": 1.0}
```

Unknown keys land in `task.metadata`, so you can carry whatever context your system needs.

---

## Reproducibility

Every run records seed, git SHA, Python version, and platform. Seeded runs, bootstraps, and
BCa intervals all reproduce exactly.

```python
run.metadata.as_dict()
# {'run_id': '40ac9e812a7f', 'seed': 42, 'git_sha': 'a1b2c3d', ...}
```

---

## Design notes

**Pure Python statistics.** Wilson intervals, bootstrap and BCa, paired permutation tests,
McNemar's exact test, Cohen's d, Cliff's delta, and power analysis, with an inverse-normal-CDF
approximation instead of a SciPy dependency. Every routine was checked against hand-computed
closed forms (Wilson 8/10 = [0.4902, 0.9433]; McNemar with 10 discordant pairs = 2/2¹⁰;
n = 388 to detect 50% → 60% at 95%/80%).

**Errors are typed.** Everything derives from `AgentEvalError`, so you can catch library failures
without swallowing bugs in your own system.

**Logging, not printing.** The library attaches a `NullHandler` and never configures logging on
import. Call `configure_logging()` if you want quick output.

**Progress goes to stderr** and disables itself when not attached to a TTY, so piping stdout to a
file still gives you clean JSON and CI logs stay readable.

---

## License

MIT
