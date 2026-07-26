from __future__ import annotations


class AgentEvalError(Exception):
    """Base class for every error raised by agenteval.

    Catch this to handle any library failure without also catching
    unrelated exceptions from user code.
    """


class SuiteError(AgentEvalError):
    """A task suite could not be loaded, parsed, or validated."""


class SuiteFormatError(SuiteError):
    """A suite file was malformed or in an unsupported format."""

    def __init__(self, path: str, detail: str, line: int | None = None):
        self.path = path
        self.detail = detail
        self.line = line
        location = f"{path}:{line}" if line else path
        super().__init__(f"{location}: {detail}")


class GraderError(AgentEvalError):
    """A grader could not be created or failed while scoring."""


class UnknownGraderError(GraderError):
    def __init__(self, name: str, available: set[str]):
        self.name = name
        self.available = available
        super().__init__(
            f"unknown grader {name!r}. Available: {', '.join(sorted(available))}"
        )


class StatisticsError(AgentEvalError):
    """A statistical routine was given inputs it cannot handle."""


class PairedLengthError(StatisticsError):
    def __init__(self, left: int, right: int, context: str = "paired comparison"):
        self.left = left
        self.right = right
        super().__init__(
            f"{context} requires equal-length samples, got {left} and {right}"
        )


class BudgetExceededError(AgentEvalError):
    """An evaluation was stopped because its cost budget was exhausted."""


class ConfigurationError(AgentEvalError):
    """A component was configured with values that cannot work together."""
