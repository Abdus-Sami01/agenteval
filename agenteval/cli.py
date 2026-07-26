from __future__ import annotations

import argparse
import json
import sys

from agenteval.graders.base import GraderRegistry
from agenteval.report import leaderboard, run_to_markdown, run_to_text
from agenteval.stats import (
    minimum_detectable_effect,
    required_sample_size,
    wilson_interval,
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
    with open(args.run) as f:
        payload = json.load(f)

    summary = payload.get("summary", {})
    meta = payload.get("metadata", {})

    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    ci = wilson_interval(passed, passed + failed)

    print(f"Suite:  {meta.get('suite', '?')}")
    print(f"System: {meta.get('system', '?')}")
    print(f"Run:    {meta.get('run_id', '?')}  seed={meta.get('seed', '?')}")
    print()
    print(f"  tasks      {summary.get('tasks', 0)}")
    print(f"  passed     {passed}")
    print(f"  failed     {failed}")
    print(f"  errored    {summary.get('errored', 0)}")
    print(f"  pass rate  {summary.get('pass_rate', 0):.1%}  (95% CI {ci.low:.1%} - {ci.high:.1%})")
    print(f"  mean score {summary.get('mean_score', 0):.4f}")
    return 0


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
