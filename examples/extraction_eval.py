import json
import random

from agenteval import (
    ExactMatchGrader,
    JSONSchemaGrader,
    StructuralGrader,
    WeightedGrader,
    compare,
    evaluate_many,
    gate,
    leaderboard,
    regression_gate,
    run_to_text,
    suite_from_records,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "amount": {"type": "number"},
    },
    "required": ["name", "amount"],
}

RECORDS = [
    {"id": "inv1", "input": "Invoice from Acme Corp for $1200",
     "expected": {"name": "Acme Corp", "amount": 1200}, "tags": ["invoice"]},
    {"id": "inv2", "input": "Globex billed us 450 dollars",
     "expected": {"name": "Globex", "amount": 450}, "tags": ["invoice"]},
    {"id": "inv3", "input": "Initech charge: $99.50",
     "expected": {"name": "Initech", "amount": 99.5}, "tags": ["invoice", "decimal"]},
    {"id": "inv4", "input": "Umbrella Inc invoice total 7800",
     "expected": {"name": "Umbrella Inc", "amount": 7800}, "tags": ["invoice"]},
    {"id": "inv5", "input": "Soylent LLC owes 25.25",
     "expected": {"name": "Soylent LLC", "amount": 25.25}, "tags": ["invoice", "decimal"]},
]


def strong_extractor(task):
    return json.dumps(task.expected)


def weak_extractor(task):
    expected = task.expected
    if isinstance(expected["amount"], float):
        return json.dumps({"name": expected["name"], "amount": round(expected["amount"])})
    return json.dumps(expected)


def unreliable_extractor(task):
    if random.Random(task.id).random() < 0.4:
        return "sorry, I could not parse that document"
    return json.dumps(task.expected)


def main():
    suite = suite_from_records("invoice-extraction", RECORDS,
                               description="Extract vendor and amount as JSON")

    grader = WeightedGrader(
        {"schema": JSONSchemaGrader(schema=SCHEMA), "fields": StructuralGrader()},
        weights={"fields": 2.0},
        threshold=1.0,
    )

    runs = evaluate_many(
        {"strong": strong_extractor, "weak": weak_extractor, "unreliable": unreliable_extractor},
        suite, grader, seed=7,
    )

    print(leaderboard(runs))
    print()
    print(run_to_text(runs["weak"]))
    print()

    result = compare(runs["strong"], runs["weak"], iterations=5000, seed=1)
    print(result.summary())
    print()

    print("QUALITY GATES (strong)")
    print(gate(runs["strong"], min_pass_rate=0.9, max_error_rate=0.0).summary())
    print()
    print("REGRESSION GATES (strong -> weak)")
    print(regression_gate(result, max_broken=0).summary())


if __name__ == "__main__":
    main()
