from agenteval.graders.base import Grader, GraderRegistry
from agenteval.graders.numeric import NumericGrader, RangeGrader
from agenteval.graders.programmatic import (
    CallableGrader,
    OutcomeGrader,
    PredicateGrader,
    StepBudgetGrader,
    ToolSequenceGrader,
)
from agenteval.graders.rubric import LLMJudgeGrader, RubricGrader, WeightedGrader
from agenteval.graders.structured import JSONSchemaGrader, SetGrader, StructuralGrader
from agenteval.graders.text import (
    ContainsGrader,
    EditDistanceGrader,
    ExactMatchGrader,
    F1TokenGrader,
    RegexGrader,
    normalize_text,
)

GraderRegistry.register("exact", ExactMatchGrader)
GraderRegistry.register("contains", ContainsGrader)
GraderRegistry.register("regex", RegexGrader)
GraderRegistry.register("edit_distance", EditDistanceGrader)
GraderRegistry.register("f1", F1TokenGrader)
GraderRegistry.register("numeric", NumericGrader)
GraderRegistry.register("range", RangeGrader)
GraderRegistry.register("set", SetGrader)
GraderRegistry.register("json_schema", JSONSchemaGrader)
GraderRegistry.register("structural", StructuralGrader)
GraderRegistry.register("callable", CallableGrader)
GraderRegistry.register("predicate", PredicateGrader)
GraderRegistry.register("rubric", RubricGrader)
GraderRegistry.register("llm_judge", LLMJudgeGrader)
GraderRegistry.register("weighted", WeightedGrader)
GraderRegistry.register("outcome", OutcomeGrader)
GraderRegistry.register("tool_sequence", ToolSequenceGrader)
GraderRegistry.register("step_budget", StepBudgetGrader)

__all__ = [
    "CallableGrader",
    "ContainsGrader",
    "EditDistanceGrader",
    "ExactMatchGrader",
    "F1TokenGrader",
    "Grader",
    "GraderRegistry",
    "JSONSchemaGrader",
    "LLMJudgeGrader",
    "normalize_text",
    "NumericGrader",
    "OutcomeGrader",
    "PredicateGrader",
    "RangeGrader",
    "RegexGrader",
    "RubricGrader",
    "SetGrader",
    "StepBudgetGrader",
    "StructuralGrader",
    "ToolSequenceGrader",
    "WeightedGrader",
]
