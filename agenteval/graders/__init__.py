from agenteval.graders.base import Grader, GraderRegistry
from agenteval.graders.text import (
    ContainsGrader,
    EditDistanceGrader,
    ExactMatchGrader,
    F1TokenGrader,
    RegexGrader,
    normalize_text,
)
from agenteval.graders.numeric import NumericGrader, RangeGrader
from agenteval.graders.structured import JSONSchemaGrader, SetGrader, StructuralGrader
from agenteval.graders.programmatic import CallableGrader, PredicateGrader
from agenteval.graders.rubric import LLMJudgeGrader, RubricGrader, WeightedGrader

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
    "PredicateGrader",
    "RangeGrader",
    "RegexGrader",
    "RubricGrader",
    "SetGrader",
    "StructuralGrader",
    "WeightedGrader",
]
