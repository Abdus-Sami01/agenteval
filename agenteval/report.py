from __future__ import annotations

import json
from typing import Any

from agenteval.stats import wilson_interval
from agenteval.types import EvalRun, Outcome, TaskResult


def run_to_dict(run: EvalRun, include_predictions: bool = True) -> dict[str, Any]:
    ci = wilson_interval(run.passed, run.passed + run.failed)
    return {
        "metadata": run.metadata.as_dict(),
        "summary": {
            "tasks": len(run),
            "passed": run.passed,
            "failed": run.failed,
            "errored": run.errored,
            "pass_rate": round(run.pass_rate, 4),
            "pass_rate_ci": [round(ci.low, 4), round(ci.high, 4)],
            "mean_score": round(run.mean_score, 4),
            "total_ms": round(run.total_ms, 1),
        },
        "results": [_result_to_dict(r, include_predictions) for r in run.results],
    }


def run_to_json(run: EvalRun, indent: int = 2, include_predictions: bool = True) -> str:
    return json.dumps(run_to_dict(run, include_predictions), indent=indent, default=str)


def run_to_text(run: EvalRun, show_failures: int = 10) -> str:
    ci = wilson_interval(run.passed, run.passed + run.failed)
    lines = [
        f"Suite:  {run.metadata.suite_name}",
        f"System: {run.metadata.system_name}",
        f"Run:    {run.metadata.run_id}  seed={run.metadata.seed}"
        + (f"  git={run.metadata.git_sha}" if run.metadata.git_sha else ""),
        "",
        f"  tasks      {len(run)}",
        f"  passed     {run.passed}",
        f"  failed     {run.failed}",
        f"  errored    {run.errored}",
        f"  pass rate  {run.pass_rate:.1%}  (95% CI {ci.low:.1%} - {ci.high:.1%})",
        f"  mean score {run.mean_score:.4f}",
        f"  wall time  {run.total_ms:.0f}ms",
    ]

    by_tag = run.by_tag()
    if by_tag:
        lines.append("")
        lines.append("  by tag:")
        for tag in sorted(by_tag):
            graded = [r for r in by_tag[tag] if r.outcome in (Outcome.PASS, Outcome.FAIL)]
            if not graded:
                continue
            rate = sum(1 for r in graded if r.is_pass) / len(graded)
            lines.append(f"    {tag:<20}{rate:>7.1%}  ({len(graded)} tasks)")

    failures = [r for r in run.results if r.outcome != Outcome.PASS]
    if failures:
        lines.append("")
        lines.append(f"  failures ({len(failures)} total, showing up to {show_failures}):")
        for r in failures[:show_failures]:
            reason = r.error or (r.score.detail if r.score else "")
            lines.append(f"    [{r.outcome.value:<7}] {r.task_id}: {reason[:100]}")

    return "\n".join(lines)


def run_to_markdown(run: EvalRun, show_failures: int = 10) -> str:
    ci = wilson_interval(run.passed, run.passed + run.failed)
    lines = [
        f"# {run.metadata.suite_name} - {run.metadata.system_name}",
        "",
        f"Run `{run.metadata.run_id}` | seed `{run.metadata.seed}`"
        + (f" | git `{run.metadata.git_sha}`" if run.metadata.git_sha else ""),
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| tasks | {len(run)} |",
        f"| passed | {run.passed} |",
        f"| failed | {run.failed} |",
        f"| errored | {run.errored} |",
        f"| pass rate | {run.pass_rate:.1%} (95% CI {ci.low:.1%} - {ci.high:.1%}) |",
        f"| mean score | {run.mean_score:.4f} |",
        f"| wall time | {run.total_ms:.0f}ms |",
    ]

    by_tag = run.by_tag()
    if by_tag:
        lines += ["", "## By tag", "", "| tag | pass rate | tasks |", "| --- | --- | --- |"]
        for tag in sorted(by_tag):
            graded = [r for r in by_tag[tag] if r.outcome in (Outcome.PASS, Outcome.FAIL)]
            if not graded:
                continue
            rate = sum(1 for r in graded if r.is_pass) / len(graded)
            lines.append(f"| {tag} | {rate:.1%} | {len(graded)} |")

    failures = [r for r in run.results if r.outcome != Outcome.PASS]
    if failures:
        lines += ["", "## Failures", "", "| task | outcome | reason |", "| --- | --- | --- |"]
        for r in failures[:show_failures]:
            reason = (r.error or (r.score.detail if r.score else "")).replace("|", "\\|")
            lines.append(f"| {r.task_id} | {r.outcome.value} | {reason[:120]} |")

    return "\n".join(lines)


def tag_breakdown(run: EvalRun, level: float = 0.95) -> str:
    by_tag = run.by_tag()
    if not by_tag:
        return "No tags on this suite."

    rows = []
    for tag in sorted(by_tag):
        graded = [r for r in by_tag[tag] if r.outcome in (Outcome.PASS, Outcome.FAIL)]
        if not graded:
            continue
        passed = sum(1 for r in graded if r.is_pass)
        ci = wilson_interval(passed, len(graded), level)
        rows.append((tag, passed, len(graded), ci))

    if not rows:
        return "No graded tasks with tags."

    width = max(len(r[0]) for r in rows) + 2
    lines = [f"{'tag':<{width}}{'pass rate':>11}{'95% CI':>20}{'n':>6}", "-" * (width + 37)]
    for tag, passed, total, ci in sorted(rows, key=lambda r: r[3].point):
        interval = f"[{ci.low:.1%}, {ci.high:.1%}]"
        lines.append(f"{tag:<{width}}{passed / total:>10.1%}{interval:>20}{total:>6}")

    thin = [r for r in rows if r[2] < 20]
    if thin:
        lines.append("")
        lines.append(f"  note: {', '.join(r[0] for r in thin)} have fewer than 20 tasks, "
                     "so those intervals are very wide")
    return "\n".join(lines)


def leaderboard_tiers(runs: dict[str, EvalRun], level: float = 0.95) -> str:
    """Group systems whose intervals overlap into statistically tied tiers."""
    if not runs:
        return "No runs to compare."

    entries = []
    for name, run in runs.items():
        ci = wilson_interval(run.passed, run.passed + run.failed, level)
        entries.append((name, ci))
    entries.sort(key=lambda e: -e[1].point)

    tiers: list[list[tuple[str, Any]]] = []
    for entry in entries:
        placed = False
        for tier in tiers:
            if any(entry[1].low <= other[1].high and other[1].low <= entry[1].high for other in tier):
                tier.append(entry)
                placed = True
                break
        if not placed:
            tiers.append([entry])

    lines = [f"Systems grouped into statistically indistinguishable tiers ({int(level * 100)}% CI)", ""]
    for i, tier in enumerate(tiers, 1):
        names = ", ".join(n for n, _ in tier)
        lines.append(f"  Tier {i}: {names}")
        for name, ci in tier:
            lines.append(f"      {name:<16}{ci.point:>7.1%}  [{ci.low:.1%}, {ci.high:.1%}]")
    lines.append("")
    if len(tiers) == 1:
        lines.append("  All systems are within noise of each other on this suite.")
    else:
        lines.append("  Tier 1 is separated from Tier 2 by non-overlapping intervals.")
    return "\n".join(lines)


def leaderboard(runs: dict[str, EvalRun], level: float = 0.95) -> str:
    if not runs:
        return "No runs to compare."

    rows = []
    for name, run in runs.items():
        ci = wilson_interval(run.passed, run.passed + run.failed, level)
        rows.append((name, run.pass_rate, ci, run.mean_score, run.errored, run.total_ms))
    rows.sort(key=lambda r: -r[1])

    width = max(len(r[0]) for r in rows) + 2
    lines = [
        f"{'system':<{width}}{'pass rate':>11}{'95% CI':>20}{'mean':>9}{'err':>6}{'ms':>9}",
        "-" * (width + 55),
    ]
    for name, rate, ci, mean_score, errors, ms in rows:
        interval = f"[{ci.low:.1%}, {ci.high:.1%}]"
        lines.append(f"{name:<{width}}{rate:>10.1%}{interval:>20}{mean_score:>9.3f}{errors:>6}{ms:>8.0f}m")

    if len(rows) > 1:
        best, runner_up = rows[0], rows[1]
        overlap = best[2].low <= runner_up[2].high
        lines.append("")
        if overlap:
            lines.append(f"note: {best[0]} and {runner_up[0]} have overlapping confidence intervals - "
                         "the ranking between them is not statistically established")
        else:
            lines.append(f"note: {best[0]} leads {runner_up[0]} with non-overlapping intervals")

    return "\n".join(lines)


def failure_digest(run: EvalRun, group_by_prefix: bool = True) -> str:
    failures = [r for r in run.results if r.outcome != Outcome.PASS]
    if not failures:
        return "No failures."

    buckets: dict[str, list[TaskResult]] = {}
    for r in failures:
        if r.error:
            key = r.error.split(":")[0][:60]
        elif r.score and r.score.detail:
            key = r.score.detail.split(",")[0][:60] if group_by_prefix else r.score.detail[:60]
        else:
            key = "failed grading"
        buckets.setdefault(key, []).append(r)

    lines = [f"{len(failures)} failures across {len(buckets)} distinct causes", ""]
    for key, group in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  [{len(group):>3}x] {key}")
        lines.append(f"         e.g. {', '.join(r.task_id for r in group[:4])}")
    return "\n".join(lines)


def _result_to_dict(r: TaskResult, include_predictions: bool) -> dict[str, Any]:
    d: dict[str, Any] = {
        "task_id": r.task_id,
        "outcome": r.outcome.value,
        "elapsed_ms": round(r.elapsed_ms, 2),
        "attempts": r.attempts,
        "tags": list(r.tags),
    }
    if r.score:
        d["score"] = round(r.score.value, 4)
        d["grader"] = r.score.grader
        if r.score.detail:
            d["detail"] = r.score.detail
        if r.score.subscores:
            d["subscores"] = {k: round(v, 4) for k, v in r.score.subscores.items()}
    if r.error:
        d["error"] = r.error
    if include_predictions:
        d["prediction"] = str(r.prediction)[:500] if r.prediction is not None else None
        d["expected"] = str(r.expected)[:500] if r.expected is not None else None
    return d
