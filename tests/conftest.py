"""Shared fixtures for the agenteval test suite."""

from __future__ import annotations

import pytest

from agenteval import ExactMatchGrader, Task, TaskSuite, evaluate, suite_from_records
from tests.helpers import adder, always_wrong  # noqa: F401


@pytest.fixture
def math_suite() -> TaskSuite:
    return suite_from_records("math", [
        {
            "id": f"q{i}",
            "input": f"{i}+{i}",
            "expected": str(i * 2),
            "tags": ["even"] if i % 2 == 0 else ["odd"],
        }
        for i in range(1, 31)
    ])


@pytest.fixture
def small_suite() -> TaskSuite:
    return suite_from_records("small", [
        {"id": "a", "input": "1+1", "expected": "2"},
        {"id": "b", "input": "2+2", "expected": "4"},
        {"id": "c", "input": "3+3", "expected": "6"},
    ])


@pytest.fixture
def task() -> Task:
    return Task(id="t1", input="question", expected="answer")


@pytest.fixture
def perfect_run(math_suite):
    return evaluate(adder, math_suite, ExactMatchGrader(), system_name="perfect")


@pytest.fixture
def failing_run(math_suite):
    return evaluate(always_wrong, math_suite, ExactMatchGrader(), system_name="failing")
