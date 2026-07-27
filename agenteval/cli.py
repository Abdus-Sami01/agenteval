from __future__ import annotations

import argparse
import sys

from agenteval.graders.base import GraderRegistry
from agenteval.report import run_to_markdown, run_to_text
from agenteval.stats import (
    minimum_detectable_effect,
    required_sample_size,
)
from agenteval.suites import load_suite, validate_suite


def cmd_validate(args) -> int:
    suite = load_suite(args.suite)
    problems = validate_suite(suite)

    if problems:
        print("INVALID:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"Suite '{suite.name}' is valid.")
    print(f"  tasks: {len(suite)}")
    if suite.all_tags:
        print(f"  tags:  {', '.join(sorted(suite.all_tags))}")
    return 0


def cmd_describe(args) -> int:
    suite = load_suite(args.suite)
    print(f"Suite:       {suite.name}")
    if suite.description:
        print(f"Description: {suite.description}")
    print(f"Tasks:       {len(suite)}")

    tag_counts: dict[str, int] = {}
    for task in suite.tasks:
        for tag in task.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if tag_counts:
        print("Tags:")
        for tag, count in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {tag:<24}{count:>5}")

    print()
    print("Sample tasks:")
    for task in suite.tasks[:args.limit]:
        print(f"  [{task.id}] {str(task.input)[:70]}")
        print(f"      expected: {str(task.expected)[:70]}")
    return 0


def cmd_power(args) -> int:
    n = required_sample_size(args.baseline, args.delta, args.level, args.power)
    print(f"To detect a {args.delta:+.1%} change from a {args.baseline:.1%} baseline")
    print(f"at {args.level:.0%} confidence with {args.power:.0%} power:")
    print()
    print(f"  required tasks: {n}")
    print()
    for size in [50, 100, 200, 500, 1000]:
        mde = minimum_detectable_effect(size, args.baseline, args.level, args.power)
        print(f"  with {size:>5} tasks you can detect a change of {mde:.1%} or larger")
    return 0


def cmd_report(args) -> int:
    from agenteval.report import load_run, tag_breakdown

    run = load_run(args.run)
    print(run_to_text(run))
    if run.by_tag():
        print()
        print(tag_breakdown(run))
    return 0


def load_object(spec: str):
    """Import 'module:attribute' and return the attribute.

    Importing runs the target module, so only ever point this at code you
    control. The spec must be given explicitly - nothing is auto-discovered.
    """
    import importlib

    if ":" not in spec:
        raise ValueError(f"expected 'module:attribute', got {spec!r}")

    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"expected 'module:attribute', got {spec!r}")

    module = importlib.import_module(module_name)
    if not hasattr(module, attribute):
        raise AttributeError(f"{module_name!r} has no attribute {attribute!r}")
    return getattr(module, attribute)


def cmd_run(args) -> int:
    import sys as _sys

    from agenteval.report import run_to_json, tag_breakdown
    from agenteval.runner import RateLimiter, RetryPolicy, evaluate

    if args.path:
        _sys.path.insert(0, args.path)

    suite = load_suite(args.suite)
    problems = validate_suite(suite)
    if problems:
        print("suite is invalid:", file=_sys.stderr)
        for p in problems:
            print(f"  - {p}", file=_sys.stderr)
        return 1

    if args.tags:
        suite = suite.filter(tags=set(args.tags.split(",")))
    if args.limit:
        suite = suite.sample(args.limit, seed=args.seed)

    system = load_object(args.system)
    grader = load_object(args.grader) if ":" in args.grader else GraderRegistry.create(args.grader)
    if callable(grader) and not hasattr(grader, "grade"):
        grader = grader()

    policy = None
    if args.retry_backoff > 0:
        policy = RetryPolicy(max_attempts=args.retries + 1, base_delay_s=args.retry_backoff)
    limiter = RateLimiter(args.rate_limit) if args.rate_limit > 0 else None

    run = evaluate(
        system, suite, grader,
        max_parallel=args.parallel, timeout_s=args.timeout, retries=args.retries,
        seed=args.seed, system_name=args.name or args.system,
        retry_policy=policy, rate_limiter=limiter,
    )

    if args.format == "json":
        print(run_to_json(run))
    elif args.format == "markdown":
        print(run_to_markdown(run))
    else:
        print(run_to_text(run))
        if suite.all_tags:
            print()
            print(tag_breakdown(run))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(run_to_json(run))
        print(f"\nwrote {args.out}")

    if args.html:
        from agenteval.html import run_to_html, write_html
        write_html(run_to_html(run), args.html)
        print(f"wrote {args.html}")

    if args.min_pass_rate is not None:
        from agenteval.compare import gate
        report = gate(run, min_pass_rate=args.min_pass_rate)
        print()
        print(report.summary())
        return 0 if report.passed else 1

    return 0


def cmd_compare(args) -> int:
    from agenteval.compare import compare, regression_gate, tag_regression_gate
    from agenteval.report import load_run

    base = load_run(args.baseline)
    cand = load_run(args.candidate)

    comparison = compare(base, cand, iterations=args.iterations, seed=args.seed)
    if not comparison.paired_ids:
        print("no overlapping task ids between the two runs")
        return 2

    print(comparison.summary())

    failed = False
    if args.max_tag_drop is not None:
        tags = tag_regression_gate(
            base, cand, max_drop=args.max_tag_drop, min_tasks=args.min_tag_tasks,
            iterations=args.iterations, seed=args.seed,
        )
        if tags.gates:
            print()
            print("Per-tag regression gates:")
            print(tags.summary())
        failed = failed or not tags.passed

    if args.fail_on_regression:
        gates = regression_gate(comparison, max_broken=args.max_broken)
        failed = failed or not gates.passed

    return 1 if failed else 0


def cmd_graders(args) -> int:
    print("Available graders:")
    for key in sorted(GraderRegistry.available()):
        print(f"  {key}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenteval", description="Evaluation harness for agents and LLM systems")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Check a task suite for structural problems")
    p.add_argument("suite")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("describe", help="Summarize a task suite")
    p.add_argument("suite")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("power", help="Compute how many tasks you need to detect a change")
    p.add_argument("--baseline", type=float, default=0.7, help="current pass rate, e.g. 0.7")
    p.add_argument("--delta", type=float, default=0.05, help="change you want to detect, e.g. 0.05")
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--power", type=float, default=0.8)
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("report", help="Summarize a saved run JSON")
    p.add_argument("run")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("graders", help="List available graders")
    p.set_defaults(func=cmd_graders)

    p = sub.add_parser("run", help="Run a suite against a system and report results")
    p.add_argument("suite")
    p.add_argument("--system", required=True,
                   help="'module:function' to evaluate. Importing runs that module, so point it only at code you control.")
    p.add_argument("--grader", default="exact",
                   help="a registered grader name, or 'module:factory' for a custom one")
    p.add_argument("--name", default="", help="label for this system in the report")
    p.add_argument("--path", default="", help="directory to add to sys.path before importing")
    p.add_argument("--tags", default="", help="comma-separated tags to filter the suite")
    p.add_argument("--limit", type=int, default=0, help="evaluate a random sample of N tasks")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--timeout", type=float, default=0)
    p.add_argument("--retries", type=int, default=0)
    p.add_argument("--retry-backoff", type=float, default=0.0,
                   help="seconds before the first retry, doubling with jitter after that")
    p.add_argument("--rate-limit", type=float, default=0.0,
                   help="cap system calls at this many per second across all workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    p.add_argument("--out", default="", help="write the run JSON here (for later comparison)")
    p.add_argument("--html", default="", help="write a standalone HTML report here")
    p.add_argument("--min-pass-rate", type=float, default=None,
                   help="exit non-zero if the pass rate falls below this")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="Compare two saved run JSON files")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fail-on-regression", action="store_true",
                   help="exit non-zero on a significant regression or too many broken tasks")
    p.add_argument("--max-broken", type=int, default=0)
    p.add_argument("--max-tag-drop", type=float, default=None,
                   help="exit non-zero if any tag's mean score drops by more than this")
    p.add_argument("--min-tag-tasks", type=int, default=5,
                   help="tags with fewer paired tasks are reported but not gated")
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"file not found: {e.filename}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
