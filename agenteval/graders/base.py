from __future__ import annotations

from typing import Any

from agenteval.types import Score, Task


class Grader:
    name = "grader"

    def grade(self, prediction: Any, task: Task) -> Score:
        raise NotImplementedError

    def _score(self, value: float, passed: bool | None = None, detail: str = "", **subscores) -> Score:
        value = max(0.0, min(1.0, float(value)))
        return Score(
            value=value,
            passed=bool(value >= 1.0) if passed is None else passed,
            grader=self.name,
            detail=detail,
            subscores=subscores,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class GraderRegistry:
    _registry: dict[str, type[Grader]] = {}

    @classmethod
    def register(cls, key: str, grader_cls: type[Grader]) -> None:
        cls._registry[key] = grader_cls

    @classmethod
    def create(cls, key: str, **kwargs) -> Grader:
        grader_cls = cls._registry.get(key)
        if grader_cls is None:
            raise ValueError(f"unknown grader {key!r}. Available: {sorted(cls._registry)}")
        return grader_cls(**kwargs)

    @classmethod
    def available(cls) -> set[str]:
        return set(cls._registry)

    @classmethod
    def has(cls, key: str) -> bool:
        return key in cls._registry
