"""Deterministic stand-in systems used across the test suite."""

from __future__ import annotations

import random

from agenteval import Task


def adder(task: Task) -> str:
    """Correct system: sums the two operands in the input."""
    a, b = str(task.input).split("+")
    return str(int(a) + int(b))


def always_wrong(task: Task) -> str:
    return "0"


def raises(task: Task):
    raise RuntimeError("model offline")


def flaky(rate: float, salt: str = ""):
    """System that answers correctly with the given probability, seeded per task."""

    def system(task: Task) -> str:
        if random.Random(task.id + salt).random() < rate:
            return adder(task)
        return "0"

    return system


def safety_regressor(task: Task) -> str:
    """Correct except on tasks tagged 'safety', for per-tag gate tests."""
    if "safety" in task.tags:
        return "0"
    return adder(task)
