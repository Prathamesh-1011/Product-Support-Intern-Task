from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AcceptanceCriteria:
    """Rule-based checks on pipeline output."""

    required_fields: list[str] = field(default_factory=list)
    field_in: dict[str, list[Any]] = field(default_factory=dict)
    field_contains: dict[str, str] = field(default_factory=dict)
    min_list_length: dict[str, int] = field(default_factory=dict)
    custom: list[Callable[[dict[str, Any]], tuple[bool, str]]] = field(default_factory=list)


@dataclass
class EvalTestCase:
    id: str
    task: str  # "triage" | "tam"
    description: str
    input: dict[str, Any]
    criteria: AcceptanceCriteria
    adversarial: bool = False
    llm_judge_prompt: str | None = None


def _check_criteria(output: dict[str, Any], criteria: AcceptanceCriteria) -> tuple[float, list[str]]:
    """Return score in [0,1] and list of failure messages."""
    checks: list[tuple[bool, str]] = []

    for fld in criteria.required_fields:
        present = fld in output and output[fld] is not None
        checks.append((present, f"required field '{fld}' present"))

    for fld, allowed in criteria.field_in.items():
        val = output.get(fld)
        ok = val in allowed
        checks.append((ok, f"{fld} in {allowed} (got {val!r})"))

    for fld, needle in criteria.field_contains.items():
        val = str(output.get(fld, ""))
        ok = needle.lower() in val.lower()
        checks.append((ok, f"{fld} contains '{needle}'"))

    for fld, min_len in criteria.min_list_length.items():
        val = output.get(fld, [])
        ok = isinstance(val, list) and len(val) >= min_len
        checks.append((ok, f"{fld} length >= {min_len}"))

    for fn in criteria.custom:
        ok, msg = fn(output)
        checks.append((ok, msg))

    if not checks:
        return 1.0, []

    passed = sum(1 for ok, _ in checks if ok)
    failures = [msg for ok, msg in checks if not ok]
    return passed / len(checks), failures
