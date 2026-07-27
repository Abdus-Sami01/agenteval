from __future__ import annotations

import json
from typing import Any

from agenteval.graders.base import Grader
from agenteval.types import Score, Task


def coerce_json(value: Any) -> Any:
    if isinstance(value, dict | list):
        return value
    text = str(value).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start == -1:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


class SetGrader(Grader):
    name = "set"

    def __init__(self, threshold: float = 1.0, order_matters: bool = False, normalize: bool = True):
        self._threshold = threshold
        self._order = order_matters
        self._normalize = normalize

    def grade(self, prediction: Any, task: Task) -> Score:
        pred_items = self._as_items(prediction)
        gold_items = self._as_items(task.expected)

        if pred_items is None or gold_items is None:
            return self._score(0.0, passed=False, detail="could not parse a collection")

        if self._order:
            matches = sum(1 for a, b in zip(pred_items, gold_items, strict=False) if a == b)
            score = matches / max(len(gold_items), 1)
            return self._score(score, passed=score >= self._threshold,
                               detail=f"{matches}/{len(gold_items)} positions match")

        pred_set, gold_set = set(pred_items), set(gold_items)
        if not pred_set and not gold_set:
            return self._score(1.0, precision=1.0, recall=1.0)

        overlap = len(pred_set & gold_set)
        precision = overlap / len(pred_set) if pred_set else 0.0
        recall = overlap / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        missing = sorted(gold_set - pred_set)[:5]
        spurious = sorted(pred_set - gold_set)[:5]
        detail = "" if f1 >= self._threshold else f"missing={missing} extra={spurious}"
        return self._score(f1, passed=f1 >= self._threshold, detail=detail,
                           precision=precision, recall=recall)

    def _as_items(self, value: Any) -> list[Any] | None:
        parsed = value if isinstance(value, list | tuple | set) else coerce_json(value)
        if parsed is None:
            parsed = [p.strip() for p in str(value).split(",") if p.strip()]
        if isinstance(parsed, dict):
            parsed = list(parsed.keys())
        if not isinstance(parsed, list | tuple | set):
            return None
        items = list(parsed)
        if self._normalize:
            items = [str(i).strip().lower() for i in items]
        return items


class JSONSchemaGrader(Grader):
    name = "json_schema"

    def __init__(self, schema: dict | None = None, use_expected: bool = False):
        self._schema = schema
        self._use_expected = use_expected

    def grade(self, prediction: Any, task: Task) -> Score:
        schema = task.expected if self._use_expected else self._schema
        if not isinstance(schema, dict):
            schema = task.metadata.get("schema")
        if not isinstance(schema, dict):
            return self._score(0.0, passed=False, detail="no schema configured")

        parsed = coerce_json(prediction)
        if parsed is None:
            return self._score(0.0, passed=False, detail="output is not valid JSON")

        try:
            import jsonschema
        except ImportError:
            ok = _shallow_schema_check(parsed, schema)
            return self._score(
                1.0 if ok else 0.0,
                passed=ok,
                detail="" if ok else "failed shallow schema check (install jsonschema for full validation)",
            )

        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.absolute_path))
        if not errors:
            return self._score(1.0)

        messages = [f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors[:3]]
        return self._score(0.0, passed=False, detail="; ".join(messages))


class StructuralGrader(Grader):
    name = "structural"

    def __init__(self, ignore_order: bool = True, partial_credit: bool = True):
        self._ignore_order = ignore_order
        self._partial = partial_credit

    def grade(self, prediction: Any, task: Task) -> Score:
        pred = coerce_json(prediction)
        gold = coerce_json(task.expected)

        if pred is None or gold is None:
            return self._score(0.0, passed=False, detail="could not parse both sides as JSON")

        matched, total, mismatches = _compare(pred, gold, self._ignore_order, "")
        score = matched / total if total else 1.0
        if not self._partial:
            score = 1.0 if score == 1.0 else 0.0

        detail = "" if score == 1.0 else "; ".join(mismatches[:3])
        return self._score(score, passed=score == 1.0, detail=detail,
                           matched=float(matched), total=float(total))


def _compare(pred: Any, gold: Any, ignore_order: bool, path: str) -> tuple[int, int, list[str]]:
    if isinstance(gold, dict):
        if not isinstance(pred, dict):
            return 0, 1, [f"{path or '<root>'}: expected object, got {type(pred).__name__}"]
        matched = total = 0
        problems: list[str] = []
        for key, gold_value in gold.items():
            child = f"{path}.{key}" if path else key
            if key not in pred:
                total += _leaf_count(gold_value)
                problems.append(f"{child}: missing")
                continue
            m, t, p = _compare(pred[key], gold_value, ignore_order, child)
            matched += m
            total += t
            problems.extend(p)
        return matched, total, problems

    if isinstance(gold, list):
        if not isinstance(pred, list):
            return 0, 1, [f"{path or '<root>'}: expected array, got {type(pred).__name__}"]
        if ignore_order:
            gold_repr = sorted(json.dumps(g, sort_keys=True, default=str) for g in gold)
            pred_repr = sorted(json.dumps(p, sort_keys=True, default=str) for p in pred)
            matched = sum(1 for g in gold_repr if g in pred_repr)
            problems = [] if matched == len(gold_repr) else [f"{path or '<root>'}: {matched}/{len(gold_repr)} items matched"]
            return matched, len(gold_repr), problems
        matched = total = 0
        problems = []
        for idx, gold_item in enumerate(gold):
            child = f"{path}[{idx}]"
            if idx >= len(pred):
                total += _leaf_count(gold_item)
                problems.append(f"{child}: missing")
                continue
            m, t, p = _compare(pred[idx], gold_item, ignore_order, child)
            matched += m
            total += t
            problems.extend(p)
        return matched, total, problems

    if pred == gold:
        return 1, 1, []
    return 0, 1, [f"{path or '<root>'}: expected {gold!r}, got {pred!r}"]


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_leaf_count(v) for v in value.values()) or 1
    if isinstance(value, list):
        return sum(_leaf_count(v) for v in value) or 1
    return 1


def _shallow_schema_check(value: Any, schema: dict) -> bool:
    expected_type = schema.get("type")
    type_map = {
        "object": dict, "array": list, "string": str,
        "number": (int, float), "integer": int, "boolean": bool,
    }
    if expected_type and expected_type in type_map:
        if not isinstance(value, type_map[expected_type]):
            return False
        if expected_type in ("number", "integer") and isinstance(value, bool):
            return False
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return False
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in value and isinstance(subschema, dict) and not _shallow_schema_check(value[key], subschema):
                    return False
    return True
