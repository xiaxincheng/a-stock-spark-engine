"""Family keys and overlap metrics for deduplicating related candidates."""
from __future__ import annotations

import re
from collections import defaultdict

from spark_core.indicator_registry import FUNCTION_REGISTRY, PATTERN_FUNCTIONS
from spark_core.unified_engine import parse_boolean


def _indicator_names(condition: str) -> tuple[str, ...]:
    names = {
        name for name in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", condition)
        if name in FUNCTION_REGISTRY or name in PATTERN_FUNCTIONS
    }
    return tuple(sorted(names))


def structure_key(condition: str, direction: str) -> tuple[str, str, str]:
    node = parse_boolean(condition)
    operators: set[str] = set()

    def visit(current) -> None:
        if current.op != "leaf":
            operators.add(current.op)
            for child in current.children or []:
                visit(child)

    visit(node)
    return (direction, "+".join(sorted(operators)) or "leaf", "+".join(_indicator_names(condition)))


def signal_jaccard(left: list[dict], right: list[dict]) -> float:
    def keys(signals: list[dict]) -> set[tuple[str, str]]:
        return {(str(item.get("code", "")), str(item.get("signal_date", ""))[:10]) for item in signals}

    a, b = keys(left), keys(right)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def group_families(
    candidates: list[dict],
    max_jaccard: float = 0.85,
) -> dict[tuple[str, str, str], list[dict]]:
    """Group structural families, then split near-duplicate signal sets."""
    structural: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for candidate in candidates:
        structural[structure_key(candidate["condition_expr"], candidate["direction"])].append(candidate)

    families: dict[tuple[str, str, str], list[dict]] = {}
    for key, members in structural.items():
        representatives: list[dict] = []
        for candidate in sorted(members, key=lambda item: item.get("id", ""), reverse=True):
            duplicate = any(
                signal_jaccard(candidate.get("signals", []), prior.get("signals", [])) >= max_jaccard
                for prior in representatives
            )
            if not duplicate:
                representatives.append(candidate)
        families[key] = representatives
    return families
