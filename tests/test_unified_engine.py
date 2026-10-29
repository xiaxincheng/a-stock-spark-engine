import numpy as np
import pandas as pd

from spark_core.indicator_registry import indicator_manifest
from spark_core.spark_l1_contrastive import ExpressionCompiler
from spark_core.unified_engine import (
    FULL_BACKTEST_END,
    FULL_BACKTEST_YEARS,
    _GENERATED_EXPRESSIONS,
    _recipe_expression,
    _LONG_PATTERNS,
    _SHORT_PATTERNS,
    describe_condition,
    development_robustness_stats,
    generate_unified_hypotheses,
    parse_boolean,
)


def test_creative_engine_development_window_is_sealed():
    assert FULL_BACKTEST_END == "2024-12-31"
    assert FULL_BACKTEST_YEARS == ("2020", "2021", "2022", "2023", "2024")


def _trade(year: str, profit: float):
    return {"buy_date": f"{year}-06-01", "profit_pct": profit}


def _df():
    n = 400
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.4, n)
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + rng.random(n),
        "low": np.minimum(open_, close) - rng.random(n),
        "close": close,
        "volume": rng.lognormal(10, 0.2, n),
    })


def test_every_registered_indicator_has_executable_recipe():
    compiler = ExpressionCompiler()
    df = _df()
    for spec in indicator_manifest():
        for direction in ("long", "short"):
            expr = _recipe_expression(spec, direction)
            valid, error = compiler.validate(expr)
            assert valid, (spec["name"], expr, error)
            mask = compiler.compile_and_eval(expr, df.copy())
            assert len(mask) == len(df)


def test_parse_nested_l4_boolean_expression():
    expr = "(strategy=A, sl=0.08, tp=0.1, hold=10) OR (RSI(14) < 30 AND CCI(14) > -100)"
    node = parse_boolean(expr)
    assert node.op == "OR"
    assert [child.op for child in node.children] == ["leaf", "AND"]
    text = describe_condition(expr, "long")
    assert "或" in text and "且" in text


def test_l1_and_l3_use_same_registry_generator():
    l1 = generate_unified_hypotheses(max_hypotheses=1, generation_layer=1)
    l3 = generate_unified_hypotheses(max_hypotheses=1, generation_layer=3)
    assert l1 and l3
    assert all(h.generation_layer == 1 and h.condition_expr and h.signal_logic_desc for h in l1)
    assert all(h.generation_layer == 3 and h.condition_expr and h.signal_logic_desc for h in l3)


def test_random_search_is_capped_at_three_initial_conditions():
    import random
    from spark_core.spark_l1_contrastive import ExpressionCompiler

    _GENERATED_EXPRESSIONS.clear()
    random.seed(20261027)
    hypotheses = generate_unified_hypotheses(max_hypotheses=100, generation_layer=1)

    def leaf_count(node) -> int:
        if node.op == "leaf":
            return 1
        return sum(leaf_count(child) for child in node.children or [])

    factor_counts = []
    for hypothesis in hypotheses:
        assert ExpressionCompiler().validate(hypothesis.condition_expr)[0]
        factor_counts.append(leaf_count(parse_boolean(hypothesis.condition_expr)))
    histogram = {count: factor_counts.count(count) for count in set(factor_counts)}
    assert {2, 3}.issubset(histogram)
    assert set(histogram).issubset({1, 2, 3})
    assert any(" OR " in h.condition_expr for h in hypotheses)


def test_development_robustness_requires_cross_year_support():
    passing = [
        _trade("2020", 2), _trade("2020", -1),
        _trade("2021", 2), _trade("2021", -1),
        _trade("2022", 2), _trade("2022", -1),
        _trade("2023", 2), _trade("2023", -1),
        _trade("2024", 1), _trade("2024", -1),
    ]
    failing = [dict(trade, profit_pct=-2) if trade["buy_date"].startswith("2024") else trade
               for trade in passing]

    passed = development_robustness_stats(passing)
    failed = development_robustness_stats(failing)
    assert passed["passed"] is True
    assert passed["positive_year_ratio"] == 0.8
    assert failed["passed"] is False
    assert failed["worst_year_pf"] == 0.0


def test_price_scaled_indicators_are_normalized_and_pattern_direction_is_consistent():
    specs = {item["name"]: item for item in indicator_manifest()}
    assert "close" in _recipe_expression(specs["BBANDS"], "long")
    assert "/ close" in _recipe_expression(specs["STDDEV"], "long")
    if "MOM" in specs:
        assert "/ close" in _recipe_expression(specs["MOM"], "long")

    rng_state = __import__("random").getstate()
    __import__("random").seed(42)
    try:
        generated = generate_unified_hypotheses(max_hypotheses=10, generation_layer=1)
    finally:
        __import__("random").setstate(rng_state)

    for h in generated:
        expr = h.condition_expr
        if h.direction.value == "long":
            assert not any(name in expr for name in _SHORT_PATTERNS)
        else:
            assert not any(name in expr for name in _LONG_PATTERNS)
