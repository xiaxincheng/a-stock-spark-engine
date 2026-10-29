"""Layered, reproducible candidate generation for public Spark Engine."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from spark_core.indicator_registry import indicator_manifest
from spark_core.spark_l1_contrastive import ExpressionCompiler
from spark_core.types import Direction, Hypothesis
from spark_core.unified_engine import (
    _recipe_expression,
    describe_condition,
    generate_unified_hypotheses,
    parse_boolean,
)


@dataclass(frozen=True)
class GenerationOptions:
    max_initial_conditions: int = 3
    l2_perturbations: tuple[float, ...] = (-0.20, -0.10, 0.10, 0.20)
    l3_mutations: tuple[str, ...] = ("threshold", "add_filter", "remove_filter", "replace_indicator")


def _direction_value(direction: str) -> str:
    return "short" if direction == "short" else "long"


def _leaf_texts(condition: str) -> list[str]:
    node = parse_boolean(condition)
    output: list[str] = []

    def visit(current) -> None:
        if current.op == "leaf":
            output.append(current.text.strip())
            return
        for child in current.children or []:
            visit(child)

    visit(node)
    return output


def _join(operator: str, leaves: list[str]) -> str:
    if len(leaves) == 1:
        return leaves[0]
    return f" {operator} ".join(f"({leaf})" for leaf in leaves)


def generate_l1(count: int = 3, direction: str = "both", seed: int | None = None) -> list[Hypothesis]:
    if seed is not None:
        random.seed(seed)
    choices = ("long", "short") if direction == "both" else (_direction_value(direction),)
    return generate_unified_hypotheses(
        max_hypotheses=count,
        generation_layer=1,
        direction_choices=choices,
    )


def _mutate_numbers(text: str, ratio: float) -> str:
    pattern = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)(?![A-Za-z_])")

    def replace(match: re.Match) -> str:
        value = float(match.group(1))
        if value == 0:
            return match.group(0)
        mutated = value * (1 + ratio)
        return str(int(mutated)) if "." not in match.group(1) else f"{mutated:g}"

    return pattern.sub(replace, text)


def generate_l2(parents: list[Hypothesis], max_children: int = 4, seed: int | None = None) -> list[Hypothesis]:
    """Create bounded parameter neighborhoods around surviving expressions."""
    rng = random.Random(seed)
    children: list[Hypothesis] = []
    seen: set[str] = set()
    for parent in parents:
        if not parent.condition_expr:
            continue
        for perturbation in GenerationOptions.l2_perturbations:
            if len(children) >= max_children:
                return children
            condition = _mutate_numbers(parent.condition_expr, perturbation)
            if condition == parent.condition_expr or condition in seen:
                continue
            if not ExpressionCompiler().validate(condition)[0]:
                continue
            seen.add(condition)
            children.append(Hypothesis(
                direction=parent.direction,
                name=f"L2 parameter neighborhood {perturbation:+.0%}: {parent.name}",
                description=f"Threshold/period neighborhood around parent {parent.id}; quality is decided only by the unified verifier.",
                condition_expr=condition,
                signal_logic_desc=describe_condition(condition, parent.direction.value),
                source_chain=[parent.id, "l2_parameter_neighborhood"],
                generation_layer=2,
                combo_type="parameter_neighborhood",
            ))
    return children


def generate_l3(
    parents: list[Hypothesis],
    max_children: int = 4,
    seed: int | None = None,
) -> list[Hypothesis]:
    """Mutate surviving structures without falling back to blind random search."""
    rng = random.Random(seed)
    specs = indicator_manifest()
    children: list[Hypothesis] = []
    seen: set[str] = set()
    for parent in parents:
        leaves = _leaf_texts(parent.condition_expr or "")
        if not leaves:
            continue
        for mutation in GenerationOptions.l3_mutations:
            if len(children) >= max_children:
                return children
            changed = list(leaves)
            combo = mutation
            if mutation == "threshold":
                changed = [_mutate_numbers(item, rng.choice((-0.15, 0.15))) for item in changed]
            elif mutation == "add_filter" and len(changed) < 5:
                spec = rng.choice(specs)
                changed.append(_recipe_expression(spec, parent.direction.value))
            elif mutation == "remove_filter" and len(changed) > 1:
                changed.pop(rng.randrange(len(changed)))
            elif mutation == "replace_indicator" and changed:
                spec = rng.choice(specs)
                changed[rng.randrange(len(changed))] = _recipe_expression(spec, parent.direction.value)
            else:
                continue

            condition = _join("AND", changed)
            if condition == parent.condition_expr or condition in seen:
                continue
            if not ExpressionCompiler().validate(condition)[0]:
                continue
            seen.add(condition)
            children.append(Hypothesis(
                direction=parent.direction,
                name=f"L3 {mutation}: {parent.name}",
                description=f"Single-step survivor mutation ({mutation}) from parent {parent.id}.",
                condition_expr=condition,
                signal_logic_desc=describe_condition(condition, parent.direction.value),
                source_chain=[parent.id, f"l3_{mutation}"],
                generation_layer=3,
                combo_type=f"survivor_{mutation}",
            ))
    return children


def generate_l4(
    parents: list[Hypothesis],
    max_children: int = 3,
    seed: int | None = None,
) -> list[Hypothesis]:
    """Fuse same-direction survivors while retaining executable semantics."""
    rng = random.Random(seed)
    by_direction: dict[str, list[Hypothesis]] = {"long": [], "short": []}
    for parent in parents:
        if parent.condition_expr:
            by_direction.setdefault(parent.direction.value, []).append(parent)

    children: list[Hypothesis] = []
    seen: set[str] = set()
    for direction, pool in by_direction.items():
        if len(pool) < 2:
            continue
        for _ in range(max(0, max_children - len(children))):
            left, right = rng.sample(pool, 2)
            operator = "OR" if rng.random() < 0.7 else "AND"
            condition = f"({left.condition_expr}) {operator} ({right.condition_expr})"
            if condition in seen:
                continue
            seen.add(condition)
            children.append(Hypothesis(
                direction=Direction.SHORT if direction == "short" else Direction.LONG,
                name=f"L4 same-direction {operator}: {left.name} / {right.name}",
                description=f"Same-direction {operator} fusion of {left.id} and {right.id}; conflict handling is explicit in the unified verifier.",
                condition_expr=condition,
                signal_logic_desc=describe_condition(condition, direction),
                source_chain=[left.id, right.id, f"l4_{operator.lower()}"],
                generation_layer=4,
                combo_type=f"same_direction_{operator}",
            ))
    return children[:max_children]
