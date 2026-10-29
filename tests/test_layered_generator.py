from spark_core.layered_generator import generate_l1, generate_l2, generate_l3, generate_l4
from spark_core.types import Direction, Hypothesis


def _parent(condition="RSI(14) < 30 AND close > MA(60)", layer=1):
    return Hypothesis(
        id=f"parent-{layer}",
        name="test parent",
        direction=Direction.LONG,
        condition_expr=condition,
        generation_layer=layer,
    )


def test_l1_generates_executable_text_and_condition():
    result = generate_l1(count=3, direction="long", seed=2026)
    assert result
    assert all(item.generation_layer == 1 for item in result)
    assert all(item.condition_expr and item.signal_logic_desc for item in result)


def test_l2_creates_parameter_neighborhood_not_random_new_structure():
    result = generate_l2([_parent()], max_children=4, seed=7)
    assert result
    assert all(item.generation_layer == 2 for item in result)
    assert all("RSI" in item.condition_expr and "MA" in item.condition_expr for item in result)


def test_l3_single_step_mutations_have_parent_chain():
    result = generate_l3([_parent()], max_children=4, seed=11)
    assert result
    assert all(item.generation_layer == 3 for item in result)
    assert all("parent-1" in item.source_chain for item in result)


def test_l4_requires_same_direction_parents():
    short = Hypothesis(
        id="short-parent", name="short", direction=Direction.SHORT,
        condition_expr="RSI(14) > 70", generation_layer=1,
    )
    long = _parent()
    assert generate_l4([short, long], max_children=2, seed=3) == []

    result = generate_l4([long, _parent("KDJ_D(9,3,3) < 20", 2)], max_children=2, seed=3)
    assert result
    assert all(item.generation_layer == 4 and item.direction == Direction.LONG for item in result)
