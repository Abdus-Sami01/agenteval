from __future__ import annotations

import csv
import json
from typing import Any

from agenteval.types import Task, TaskSuite


def suite_from_records(
    name: str,
    records: list[dict[str, Any]],
    input_key: str = "input",
    expected_key: str = "expected",
    id_key: str = "id",
    description: str = "",
) -> TaskSuite:
    tasks = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {i} is not a mapping")
        if input_key not in record:
            raise ValueError(f"record {i} missing required key {input_key!r}")

        known = {input_key, expected_key, id_key, "tags", "weight"}
        tasks.append(Task(
            id=str(record.get(id_key, f"task_{i}")),
            input=record[input_key],
            expected=record.get(expected_key),
            tags=tuple(record.get("tags", ()) or ()),
            weight=float(record.get("weight", 1.0)),
            metadata={k: v for k, v in record.items() if k not in known},
        ))
    return TaskSuite(name=name, tasks=tasks, description=description)


def load_jsonl(path: str, name: str = "", **kwargs) -> TaskSuite:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {e}")
    return suite_from_records(name or _stem(path), records, **kwargs)


def load_json(path: str, name: str = "", **kwargs) -> TaskSuite:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        records = payload.get("tasks", [])
        name = name or payload.get("name", _stem(path))
        description = payload.get("description", "")
    else:
        records = payload
        name = name or _stem(path)
        description = ""

    return suite_from_records(name, records, description=description, **kwargs)


def load_csv(path: str, name: str = "", **kwargs) -> TaskSuite:
    with open(path, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    for record in records:
        if record.get("tags"):
            record["tags"] = tuple(t.strip() for t in str(record["tags"]).split(";") if t.strip())
    return suite_from_records(name or _stem(path), records, **kwargs)


def load_yaml(path: str, name: str = "", **kwargs) -> TaskSuite:
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required for YAML suites: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if isinstance(payload, dict):
        records = payload.get("tasks", [])
        name = name or payload.get("name", _stem(path))
        description = payload.get("description", "")
    else:
        records = payload or []
        name = name or _stem(path)
        description = ""

    return suite_from_records(name, records, description=description, **kwargs)


def load_suite(path: str, **kwargs) -> TaskSuite:
    lowered = path.lower()
    if lowered.endswith(".jsonl"):
        return load_jsonl(path, **kwargs)
    if lowered.endswith(".json"):
        return load_json(path, **kwargs)
    if lowered.endswith(".csv"):
        return load_csv(path, **kwargs)
    if lowered.endswith((".yaml", ".yml")):
        return load_yaml(path, **kwargs)
    raise ValueError(f"unsupported suite format: {path}")


def save_suite(suite: TaskSuite, path: str) -> None:
    payload = {
        "name": suite.name,
        "description": suite.description,
        "tasks": [
            {
                "id": t.id,
                "input": t.input,
                "expected": t.expected,
                "tags": list(t.tags),
                "weight": t.weight,
                **t.metadata,
            }
            for t in suite.tasks
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def validate_suite(suite: TaskSuite) -> list[str]:
    problems = []
    if not suite.tasks:
        problems.append("suite has no tasks")

    seen: set[str] = set()
    for task in suite.tasks:
        if task.id in seen:
            problems.append(f"duplicate task id: {task.id!r}")
        seen.add(task.id)
        if task.input is None:
            problems.append(f"task {task.id!r} has no input")
        if task.weight <= 0:
            problems.append(f"task {task.id!r} has non-positive weight {task.weight}")
    return problems


def _stem(path: str) -> str:
    base = path.replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0]
