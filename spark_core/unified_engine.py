"""Unified strategy semantics for Spark L1-L4.

Every Spark strategy is represented by one canonical ``condition_expr``.  This
module turns that expression into signals, a human explanation, and one full
market backtest.  L1/L3 indicator expressions, L2 strategy expressions and L4
AND/OR fusions therefore use the same executable semantics.
"""
from __future__ import annotations

import ast
import gc
import json
import multiprocessing
import os
import random
import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from spark_core.indicator_registry import FUNCTION_REGISTRY, PATTERN_FUNCTIONS, indicator_manifest
from spark_core.spark_l1_contrastive import ExpressionCompiler
from spark_core.types import Direction, Hypothesis


FULL_BACKTEST_START = "2020-01-01"
FULL_BACKTEST_END = "2024-12-31"
FULL_BACKTEST_YEARS = tuple(str(year) for year in range(2020, 2025))
MIN_SIGNALS_PER_FULL_YEAR = 300
MAX_SIGNALS_PER_FULL_YEAR = int(os.environ.get("SPARK_MAX_SIGNALS_PER_YEAR", "100000"))
MAX_INITIAL_CONDITION_COUNT = int(os.environ.get("SPARK_MAX_INITIAL_CONDITIONS", "3"))

_MAX_EXPRESSION_CACHE = int(os.environ.get("SPARK_MAX_EXPRESSION_CACHE", "2"))
_MAX_STRATEGY_CACHE = int(os.environ.get("SPARK_MAX_STRATEGY_CACHE", "8"))
_EXPRESSION_SIGNAL_CACHE: OrderedDict[str, list[dict]] = OrderedDict()
_STRATEGY_SIGNAL_CACHE: OrderedDict[tuple, list[dict]] = OrderedDict()
_GENERATED_EXPRESSIONS: set[str] = set()
_FULL_MARKET_DF_MAP = None
_FULL_MARKET_DF_KEY = None
_LAST_EVALUATION_TRUNCATED = False
_MAX_SIGNALS_PER_SCAN_WORKER = int(os.environ.get("SPARK_MAX_SIGNALS_PER_WORKER", "100000"))
_DIRECTIONAL_POOL_SIZE = int(os.environ.get("SPARK_DIRECTIONAL_POOL_SIZE", "36"))
_DIRECTIONAL_EXPLORE_RATIO = float(os.environ.get("SPARK_DIRECTIONAL_EXPLORE_RATIO", "0.20"))
_DIRECTIONAL_PRIORS: dict[str, dict[str, float]] | None = None


@dataclass
class BoolNode:
    op: str  # "leaf", "AND", "OR"
    text: str = ""
    children: list["BoolNode"] | None = None


def _strip_outer_parens(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        closes_at_end = False
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i == len(text) - 1:
                    closes_at_end = True
                elif depth == 0:
                    closes_at_end = False
                    break
        if closes_at_end:
            text = text[1:-1].strip()
        else:
            break
    return text


def split_top_level(text: str, operator: str) -> list[str] | None:
    """Split a top-level boolean operator while ignoring nested parentheses."""
    text = _strip_outer_parens(text)
    depth = 0
    token = f" {operator} "
    start = 0
    parts: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        elif depth == 0 and text[i:i + len(token)].upper() == token:
            parts.append(text[start:i])
            i += len(token)
            start = i
            continue
        i += 1
    if not parts:
        return None
    parts.append(text[start:])
    return parts


def parse_boolean(text: str) -> BoolNode:
    """Parse top-level AND/OR expressions used by Spark L1-L4."""
    text = _strip_outer_parens(text or "")
    for op in ("AND", "OR"):
        parts = split_top_level(text, op)
        if parts:
            return BoolNode(op=op, children=[parse_boolean(part) for part in parts])
    return BoolNode(op="leaf", text=text)


def _fmt_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.6g}"
    return f"{value:.4g}"


def _default_params(spec: dict) -> tuple:
    if spec["name"] == "MAMA":
        return (0.5, 0.05)
    if spec["name"] == "BBANDS":
        return (20, 2, 2, 0)
    preferred = {
        "N": 14, "D": 5, "M": 20, "M1": 11, "M2": 9,
        "N1": 3, "N2": 6, "N3": 12, "N4": 24,
        "FAST": 12, "SLOW": 26, "SIGNAL": 9, "P1": 7, "P2": 14, "P3": 28,
        "K": 2, "MULT": 3, "ACCEL": 0.02, "MAX": 0.2,
    }
    values = []
    if spec["name"] == "STOCH_K":
        return (5, 3, 3)
    if spec["name"] == "STOCH_D":
        return (5, 3, 3)
    if spec["name"] == "STOCHF_K":
        return (5, 3)
    if spec["name"] == "STOCHF_D":
        return (5, 3)
    if spec["name"] == "STOCHRSI":
        return (14, 3, 3)
    for name, low, high in spec.get("params", []):
        if name in preferred:
            value = preferred[name]
        elif name in ("N", "PERIOD"):
            value = 14
        else:
            value = (float(low) + float(high)) / 2
        if isinstance(low, int) and isinstance(high, int):
            value = int(max(low, min(high, round(value))))
        values.append(value)
    return tuple(values)


def _call(spec: dict) -> str:
    params = _default_params(spec)
    return f"{spec['name']}({','.join(_fmt_number(x) for x in params)})"


_PRICE_OVERLAYS = {
    "MA", "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT",
    "MIDPRICE", "HT_TRENDLINE", "LINEARREG", "LINEARREG_INTERCEPT", "TSF", "BBI",
    "HMA", "BOLL_MID", "BOLL_UPPER", "BOLL_LOWER", "ENE_UPPER", "ENE_MID", "ENE_LOWER",
    "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE", "SAR", "BBANDS", "MAMA",
}
_DIRECTION_INDICATORS = {"SUPERTREND", "HT_TRENDMODE"}
_BOUNDED_0_100 = {
    "RSI", "KDJ_K", "KDJ_D", "KDJ_J", "SKDJ_K", "SKDJ_D", "SKDJ_J", "STOCH_K",
    "STOCH_D", "STOCHF_K", "STOCHF_D", "MFI", "PSY", "CHOP", "ADX", "ADXR",
    "AROON_UP", "AROON_DOWN", "AROONOSC", "CMO",
}
_NEGATIVE_100_0 = {"WR", "WILLR"}
_ZERO_CENTERED = {
    "MACD_DIF": 0.0, "MACD_DEA": 0.0, "MACD_HIST": 0.0, "CCI": -100.0,
    "TRIX": 0.0, "APO": 0.0, "PPO": 0.0, "BOP": 0.0, "ULTOSC": 35.0,
    "ROC": -0.1, "ROCP": -0.1, "ROCR": 0.9, "MOM": -1.0,
}
_PRICE_NORMALIZED_INDICATORS = {
    "STDDEV": 0.08,
    "VAR": 0.01,
    "LINEARREG_SLOPE": 0.0,
    "LINEARREG_ANGLE": 30.0,
    "MOM": 0.1,
    "APO": 0.0,
    "MACDFIX": 0.0,
}
_LONG_PATTERNS = {
    "is_bullish_engulfing", "is_hammer", "is_morning_star", "gap_up", "consecutive_up",
    "CDL3INSIDE", "CDL3WHITESOLDIERS", "CDLCOUNTERATTACK", "CDLDRAGONFLYDOJI",
    "CDLHAMMER", "CDLHOMINGPIGEON", "CDLINVERTEDHAMMER", "CDLKICKING",
    "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM", "CDLMATCHINGLOW",
    "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR", "CDLPIERCING", "CDLSTICKSANDWICH",
    "CDLTAKURI", "CDLUNIQUE3RIVER",
}
_SHORT_PATTERNS = {
    "is_hanging_man", "is_evening_star", "is_shooting_star", "gap_down", "consecutive_down",
    "CDL2CROWS", "CDL3BLACKCROWS", "CDLABANDONEDBABY", "CDLADVANCEBLOCK",
    "CDLDARKCLOUDCOVER", "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR", "CDLGRAVESTONEDOJI",
    "CDLHANGINGMAN", "CDLIDENTICAL3CROWS", "CDLINNECK", "CDLONNECK",
    "CDLRISEFALL3METHODS", "CDLSHOOTINGSTAR", "CDLSTALLEDPATTERN", "CDLTHRUSTING",
    "CDLUPSIDEGAP2CROWS", "CDLXSIDEGAP3METHODS",
}
_RATIO_INDICATORS = {
    "volume_ratio": 1.5, "volume_trend": 0.0, "turnover_rate": 3.0,
    "VR": 120.0, "AR": 120.0, "BR": 120.0, "CR": 100.0,
    "VSTD": 0.8, "ATR": 0.05, "NATR": 5.0, "std": 0.05, "amplitude": 0.08,
    "bollinger_width": 0.12, "CHOP": 45.0,
}


def _recipe_expression(spec: dict, direction: str = "long") -> str:
    name = spec["name"]
    call = _call(spec)
    short = direction == "short"

    if name in PATTERN_FUNCTIONS or name in {"gap_up", "gap_down", "consecutive_up", "consecutive_down"}:
        return f"{name}() == 1"
    if name in _PRICE_NORMALIZED_INDICATORS:
        threshold = _PRICE_NORMALIZED_INDICATORS[name]
        denominator = "close * close" if name == "VAR" else "close"
        if name == "STDDEV":
            return f"({call} / {denominator}) {'>' if short else '<'} {threshold}"
        if name in {"VAR", "LINEARREG_SLOPE", "MOM", "APO", "MACDFIX"}:
            return f"({call} / {denominator}) {'<' if short else '>'} {threshold}"
        if name == "LINEARREG_ANGLE":
            return f"{call} {'<' if short else '>'} {threshold}"
    if name in _PRICE_OVERLAYS:
        return f"close {'<' if short else '>'} {call}"
    if name in _DIRECTION_INDICATORS:
        return f"{call} {'<' if short else '>'} 0"
    if name in _BOUNDED_0_100:
        if name in {"ADX", "ADXR", "CHOP"}:
            threshold = 45 if name == "CHOP" else 22
            return f"{call} {'<' if short else '>'} {threshold}"
        threshold = 70 if short else 30
        return f"{call} {'>' if short else '<'} {threshold}"
    if name in _NEGATIVE_100_0:
        threshold = -20 if short else -80
        return f"{call} {'>' if short else '<'} {threshold}"
    if name in _ZERO_CENTERED:
        threshold = -_ZERO_CENTERED[name] if short else _ZERO_CENTERED[name]
        return f"{call} {'<' if short else '>'} {_fmt_number(threshold)}"
    if name in _RATIO_INDICATORS:
        threshold = _RATIO_INDICATORS[name]
        if name in {"volume_ratio", "turnover_rate", "VR", "AR", "BR", "CR"}:
            return f"{call} > {threshold}"
        if name in {"CHOP", "NATR", "std", "amplitude", "bollinger_width", "VSTD"}:
            return f"{call} {'>' if short else '<'} {threshold}"
        return f"{call} {'<' if short else '>'} {threshold}"
    if name in {"up_days", "down_days"}:
        return f"{name}(20) {'>' if short else '<'} 8"
    if name in {"PVT", "WVAD", "OBV", "AD", "ADOSC"}:
        return f"{call} > 0"

    # Remaining TA-Lib statistics/cycle series use a stable 30/70 percentile
    # threshold measured on a representative stock. This keeps every registered
    # indicator executable while avoiding a fixed absolute-price threshold.
    threshold = _representative_percentile(spec, 70 if short else 30)
    return f"{call} {'<' if short else '>'} {_fmt_number(threshold)}"


def _pattern_direction(spec: dict) -> str | None:
    name = spec["name"]
    if name in _LONG_PATTERNS:
        return "long"
    if name in _SHORT_PATTERNS:
        return "short"
    return None


_REPRESENTATIVE_CACHE: dict[str, float] = {}


def seed_generated_expressions(expressions: Iterable[str]) -> None:
    """Persist expression dedup across engine restarts."""
    _GENERATED_EXPRESSIONS.update(expr for expr in expressions if expr)


def _representative_percentile(spec: dict, percentile: float) -> float:
    key = f"{spec['name']}@{percentile}"
    if key in _REPRESENTATIVE_CACHE:
        return _REPRESENTATIVE_CACHE[key]
    path = os.environ.get("SPARK_REFERENCE_CSV", "")
    if not path:
        return 0.0
    try:
        from backtest_framework import load_stock_data
        df = load_stock_data(path)
        params = _default_params(spec)
        values = FUNCTION_REGISTRY[spec["name"]][1](df, *params).replace([np.inf, -np.inf], np.nan).dropna()
        threshold = float(values.quantile(percentile / 100.0))
    except Exception:
        threshold = 0.0
    _REPRESENTATIVE_CACHE[key] = threshold
    return threshold


def describe_condition(condition: str, direction: str = "long") -> str:
    """Turn a canonical expression into concise Chinese trading logic."""
    node = parse_boolean(condition)

    spec_by_name = {item["name"]: item for item in indicator_manifest()}

    def render_leaf(text: str) -> str:
        strategy = parse_strategy_condition(text)
        if strategy:
            name, sl, tp, hold = strategy
            return (
                f"执行策略「{name}」的原始入场信号；"
                f"止损{sl:.0%}，止盈{tp:.0%}，最多持有{hold}天"
            )
        if text.startswith("indicator="):
            match = re.search(r"indicator=([^,]+),\s*method=([^,]+)", text)
            if match:
                return f"执行L3指标「{match.group(1)}」的{match.group(2)}信号"
        for name in sorted(spec_by_name, key=len, reverse=True):
            if name in text:
                return f"{text}（{spec_by_name[name]['description']}）"
        return text

    def render(current: BoolNode) -> str:
        if current.op == "leaf":
            return render_leaf(current.text)
        op = "且" if current.op == "AND" else "或"
        rendered = [
            f"({render(child)})" if child.op != "leaf" else render(child)
            for child in current.children or []
        ]
        return f" {op} ".join(rendered)

    joined = render(node) or condition
    dir_text = "做空" if direction == "short" else "做多"
    return f"{dir_text}信号：{joined}。信号次日开盘入场，回测使用统一止损/止盈/持有规则。"


def _condition_indicator_names(condition: str) -> set[str]:
    return {
        name for name in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", condition)
        if name in FUNCTION_REGISTRY or name in PATTERN_FUNCTIONS
    }


def _load_directional_priors() -> dict[str, dict[str, float]]:
    """Mine historical validation outcomes into long/short indicator priors.

    Verified hypotheses contribute positively using PF, win rate and signal
    stability. Rejected hypotheses get a smaller negative weight so shared
    indicators in a random AND expression are not overly punished. All scores
    are shrunk toward zero to keep this a search prior rather than a hard rule.
    """
    global _DIRECTIONAL_PRIORS
    if _DIRECTIONAL_PRIORS is not None:
        return _DIRECTIONAL_PRIORS

    aggregates: dict[str, dict[str, tuple[float, float]]] = {
        "long": {},
        "short": {},
    }
    try:
        db_path = os.environ.get("SPARK_DB_PATH", "")
        if not db_path:
            raise RuntimeError("directional priors disabled: SPARK_DB_PATH is not set")
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        epoch = conn.execute(
            "SELECT MAX(epoch) FROM spark_engine_epochs"
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT direction, status, condition_expr, backtest_pf,
                   backtest_win_rate, verification_result, rejection_reason
            FROM spark_hypotheses
            WHERE leaderboard_epoch=? AND status IN ('verified','rejected')
            """,
            (epoch,),
        ).fetchall()
        conn.close()

        for row in rows:
            direction = row["direction"] if row["direction"] in aggregates else "long"
            names = _condition_indicator_names(row["condition_expr"] or "")
            if not names:
                continue

            if row["status"] == "verified":
                pf = float(row["backtest_pf"] or 1.0)
                win_rate = float(row["backtest_win_rate"] or 0.5)
                score = max(0.0, pf - 1.0) + max(0.0, win_rate - 0.5) * 2.0
                try:
                    verification = json.loads(row["verification_result"] or "{}")
                    metrics = next(
                        (item.get("metrics", {}) for item in verification.get("results", [])
                         if item.get("gate") == "full_backtest"),
                        {},
                    )
                    yearly = metrics.get("yearly_signal_count") or {}
                    if min(yearly.values() or [0]) >= 1000:
                        score += 0.10
                    if max(yearly.values() or [0]) > 50_000:
                        score -= 0.20
                except (TypeError, json.JSONDecodeError):
                    pass
                weight = 1.0
            else:
                reason = row["rejection_reason"] or ""
                if "Signal-width gate" in reason or "Annual signal-width" in reason:
                    score = -0.18
                elif "Annual signal gate" in reason:
                    score = -0.12
                else:
                    score = -0.08
                weight = 0.20

            for name in names:
                total, total_weight = aggregates[direction].get(name, (0.0, 0.0))
                aggregates[direction][name] = (
                    total + score * weight,
                    total_weight + weight,
                )
    except Exception:
        aggregates = {"long": {}, "short": {}}

    result: dict[str, dict[str, float]] = {"long": {}, "short": {}}
    for direction, name_stats in aggregates.items():
        for name, (total, weight) in name_stats.items():
            raw = total / weight if weight else 0.0
            result[direction][name] = raw * weight / (weight + 2.0)
    _DIRECTIONAL_PRIORS = result
    return result


def _directional_pool(direction: str, eligible: list[dict]) -> list[dict]:
    """Return a compact prior-weighted pool while preserving exploration."""
    priors = _load_directional_priors().get(direction, {})
    ranked = sorted(
        eligible,
        key=lambda spec: (priors.get(spec["name"], 0.0), spec["name"]),
        reverse=True,
    )
    if len(ranked) < 24:
        return ranked

    pool = ranked[:max(12, _DIRECTIONAL_POOL_SIZE)]
    pool_names = {spec["name"] for spec in pool}
    # Keep enough continuous indicators so multi-factor expressions are not
    # dominated by rare candle patterns.
    continuous = [
        spec for spec in ranked
        if spec["name"] in FUNCTION_REGISTRY and _pattern_direction(spec) is None
    ]
    for spec in continuous:
        if len([x for x in pool if x["name"] in FUNCTION_REGISTRY and _pattern_direction(x) is None]) >= 12:
            break
        if spec["name"] not in pool_names:
            pool.append(spec)
            pool_names.add(spec["name"])
    return pool


def _prior_weighted_sample(
    direction: str,
    eligible: list[dict],
    pool: list[dict],
    exclude: set[str],
) -> dict:
    source = pool
    if random.random() < _DIRECTIONAL_EXPLORE_RATIO:
        source = eligible
    choices = [spec for spec in source if spec["name"] not in exclude]
    if not choices:
        choices = [spec for spec in eligible if spec["name"] not in exclude]
    return random.choice(choices)


def generate_unified_hypotheses(
    max_hypotheses: int = 3,
    generation_layer: int = 1,
    direction_choices: Iterable[str] = ("long", "short"),
    sample_csv_files: list[str] | None = None,
) -> list[Hypothesis]:
    """Randomly generate canonical indicator expressions from the full registry."""
    specs = indicator_manifest()
    if not specs:
        return []
    hypotheses: list[Hypothesis] = []
    attempts = 0
    max_attempts = max(20, max_hypotheses * 20)
    directions = list(direction_choices)

    def build_condition(chosen_specs: list[dict], selected_direction: str) -> tuple[str, str]:
        leaves = [_recipe_expression(spec, selected_direction) for spec in chosen_specs]
        count = len(leaves)
        if count <= 3:
            operator = random.choices(["AND", "OR"], weights=[70, 30], k=1)[0]
            return f" {operator} ".join(leaves), f"uniform-{operator}"
        if count == 4:
            if random.random() < 0.55:
                return " AND ".join(leaves), "uniform-AND"
            random.shuffle(leaves)
            return f"({leaves[0]} OR {leaves[1]}) AND ({leaves[2]} OR {leaves[3]})", "two-OR-groups"

        random.shuffle(leaves)
        if random.random() < 0.5:
            return (
                f"({leaves[0]} OR {leaves[1]}) AND ({leaves[2]} OR {leaves[3]} OR {leaves[4]})",
                "OR2-AND-OR3",
            )
        return (
            f"({leaves[0]} OR {leaves[1]}) AND ({leaves[2]} OR {leaves[3]}) AND {leaves[4]}",
            "OR2-AND-OR2-AND-filter",
        )

    while len(hypotheses) < max_hypotheses and attempts < max_attempts:
        attempts += 1
        direction = random.choice(directions)
        eligible = [
            spec for spec in specs
            if _pattern_direction(spec) in (None, direction)
        ]
        # Initial divergence stays simple. Higher-order expressions must be
        # earned later by the survivor-evolution layer, not random search.
        count_choices = list(range(1, max(1, MAX_INITIAL_CONDITION_COUNT) + 1))
        count_weights = [8, 32, 32, 18, 10][: len(count_choices)]
        count = random.choices(count_choices, weights=count_weights, k=1)[0]
        # At least one continuous indicator avoids sparse pattern-only L1/L3
        # candidates and makes the annual >=300 signal gate meaningful.
        continuous = [
            spec for spec in eligible
            if spec["name"] in FUNCTION_REGISTRY and _pattern_direction(spec) is None
        ]
        prior_pool = _directional_pool(direction, eligible)
        prior_continuous = [
            spec for spec in prior_pool
            if spec["name"] in FUNCTION_REGISTRY and _pattern_direction(spec) is None
        ]
        chosen = [_prior_weighted_sample(direction, continuous, prior_continuous, set())]
        if count > 1:
            min_continuous = max(1, (count + 1) // 2)
            continuous_count = 1
            used_names = {chosen[0]["name"]}
            remaining_slots = count - 1
            while len(chosen) < count:
                required_continuous = min_continuous - continuous_count
                must_choose_continuous = required_continuous >= remaining_slots
                source_eligible = continuous if must_choose_continuous else eligible
                source_pool = prior_continuous if must_choose_continuous else prior_pool
                spec = _prior_weighted_sample(
                    direction,
                    source_eligible,
                    source_pool,
                    used_names,
                )
                chosen.append(spec)
                used_names.add(spec["name"])
                if spec in continuous:
                    continuous_count += 1
                remaining_slots -= 1

        condition, structure = build_condition(chosen, direction)
        if condition in _GENERATED_EXPRESSIONS:
            continue

        compiler = ExpressionCompiler()
        valid, error = compiler.validate(condition)
        if not valid:
            continue
        if sample_csv_files:
            sampled_files = random.sample(sample_csv_files, min(240, len(sample_csv_files)))
            signals = evaluate_condition(condition, sampled_files)[0]
            recent_signals = [
                sig for sig in signals
                if FULL_BACKTEST_START <= str(sig.get("signal_date", ""))[:10] <= FULL_BACKTEST_END
            ]
            # 240 of ~5200 stocks is about 4.6%; 30 recent signals estimates a
            # lower-bound pace above 300/year before the expensive full scan.
            sample_count = len({
                (sig.get("code"), sig.get("signal_date"))
                for sig in recent_signals
            })
            sample_fraction = len(sampled_files) / max(1, len(sample_csv_files))
            estimated_per_year = sample_count / (len(FULL_BACKTEST_YEARS) * sample_fraction)
            if sample_count < 30 or estimated_per_year > MAX_SIGNALS_PER_FULL_YEAR:
                continue

        _GENERATED_EXPRESSIONS.add(condition)
        names = [item["name"] for item in chosen]
        name_prefix = "统一指标" if generation_layer == 1 else "指标演化"
        name = (
            f"[L{generation_layer}][{ '做空' if direction == 'short' else '做多'}] "
            f"{name_prefix}:{'+'.join(names)}"
        )
        description = (
            f"L{generation_layer}统一表达式策略，来自180项指标注册表。"
            f"指标：{', '.join(names)}。结构：{structure}。"
        )
        hypotheses.append(Hypothesis(
            direction=Direction.SHORT if direction == "short" else Direction.LONG,
            name=name,
            description=description[:300],
            condition_expr=condition,
            signal_logic_desc=describe_condition(condition, direction),
            source_chain=[f"unified_registry:{name}" for name in names],
            generation_layer=generation_layer,
            # Keep debate semantics aligned with legacy L1: registry-built
            # indicator expressions are genuinely new, not legacy pair combos.
            combo_type="new_strategy",
        ))
    return hypotheses


def _scan_expression_chunk(args: tuple) -> list[tuple[str, dict]]:
    condition, files = args
    scanner = ExpressionCompiler().make_scan_function(condition, None)
    output = []
    for fpath in files:
        try:
            signals = scanner(fpath)
            if signals:
                output.extend((fpath, sig) for sig in signals)
                if len(output) >= _MAX_SIGNALS_PER_SCAN_WORKER:
                    output.append((fpath, {"_signal_truncated": True}))
                    break
        except Exception:
            continue
    return output


def _scan_strategy_chunk(args: tuple) -> list[tuple[str, dict]]:
    scan_func, files = args
    output = []
    for fpath in files:
        try:
            signals = scan_func(fpath)
            if signals:
                output.extend((fpath, sig) for sig in signals)
                if len(output) >= _MAX_SIGNALS_PER_SCAN_WORKER:
                    output.append((fpath, {"_signal_truncated": True}))
                    break
        except Exception:
            continue
    return output


def _parallel_scan(scanner, payload, files: list[str], serial_key: str) -> list[dict]:
    configured = int(os.environ.get("SPARK_SCAN_WORKERS", "8"))
    worker_count = min(configured, 8, max(1, multiprocessing.cpu_count() - 1))
    if worker_count <= 1 or len(files) < 200:
        results = scanner((payload, files))
    else:
        size = (len(files) + worker_count - 1) // worker_count
        chunks = [(payload, files[i:i + size]) for i in range(0, len(files), size)]
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(min(worker_count, len(chunks))) as pool:
            results = [item for chunk in pool.map(scanner, chunks) for item in chunk]
    signals = []
    for fpath, sig in results:
        if sig.get("_signal_truncated"):
            return [{"_signal_truncated": True}]
        code = os.path.basename(fpath).replace(".csv", "")
        if "/sh/" in fpath or fpath.startswith("sh"):
            code = f"sh{code}"
        elif "/sz/" in fpath or fpath.startswith("sz"):
            code = f"sz{code}"
        sig["code"] = code
        signals.append(sig)
    return signals


def _strategy_scan_map() -> dict:
    # Public builds intentionally have no bundled, private strategy catalog.
    # Expressions generated in this process remain executable.
    result = {}
    try:
        from spark_core.spark_l1_contrastive import DYNAMIC_STRATEGIES
        result.update({name: value[0] for name, value in DYNAMIC_STRATEGIES.items()})
    except Exception:
        pass
    return result


def parse_strategy_condition(condition: str) -> tuple[str, float, float, int] | None:
    match = re.search(r"strategy=([^,]+),\s*sl=([0-9.]+),\s*tp=([0-9.]+),\s*hold=([0-9]+)", condition)
    if not match:
        return None
    return match.group(1).strip(), float(match.group(2)), float(match.group(3)), int(match.group(4))


def _legacy_l3_condition(condition: str, files: list[str]) -> list[dict]:
    match = re.search(r"indicator=([^,]+),\s*method=([^,]+),\s*params=(\{.*\})", condition)
    if not match:
        return []
    from backtest_framework import load_stock_data
    from spark_core.spark_l3_evolution import L3EvolutionLab, INDICATOR_POOL
    name, method = match.group(1), match.group(2)
    try:
        params = ast.literal_eval(match.group(3))
    except Exception:
        params = {}
    indicator = next((item for item in INDICATOR_POOL if item["name"] == name), None)
    if indicator is None:
        return []
    lab = L3EvolutionLab(None)
    output = []
    for fpath in files:
        try:
            df = load_stock_data(fpath)
            if df is None:
                continue
            indices = lab._detect_signals(df, indicator, method, params)
            code = os.path.basename(fpath).replace(".csv", "")
            code = ("sh" if "/sh/" in fpath else "sz") + code
            for idx in indices:
                if 0 <= idx < len(df):
                    output.append({"code": code, "signal_date": str(df.iloc[idx]["date"]), "signal_type": f"L3:{name}"})
        except Exception:
            continue
    return output


def _known_strategy_text_signals(text: str, files: list[str]) -> list[dict]:
    """Compatibility for old pseudo expressions such as 'ENE轨道线入场信号'."""
    scan_map = _strategy_scan_map()
    known = sorted((name for name in scan_map if name in text), key=len, reverse=True)
    if not known:
        return []
    signals = []
    for name in known:
        signals.extend(_strategy_signals(name, files))
    return signals


def _expression_signals(condition: str, files: list[str]) -> list[dict]:
    key = (condition, tuple(sorted(files)))
    if key not in _EXPRESSION_SIGNAL_CACHE:
        _EXPRESSION_SIGNAL_CACHE[key] = _parallel_scan(_scan_expression_chunk, condition, files, key)
        while len(_EXPRESSION_SIGNAL_CACHE) > _MAX_EXPRESSION_CACHE:
            _EXPRESSION_SIGNAL_CACHE.popitem(last=False)
    else:
        _EXPRESSION_SIGNAL_CACHE.move_to_end(key)
    return list(_EXPRESSION_SIGNAL_CACHE[key])


def _strategy_signals(name: str, files: list[str]) -> list[dict]:
    scan_map = _strategy_scan_map()
    scan_func = scan_map.get(name)
    if scan_func is None:
        return []
    key = (name, tuple(sorted(files)))
    if key not in _STRATEGY_SIGNAL_CACHE:
        _STRATEGY_SIGNAL_CACHE[key] = _parallel_scan(_scan_strategy_chunk, scan_func, files, name)
        while len(_STRATEGY_SIGNAL_CACHE) > _MAX_STRATEGY_CACHE:
            _STRATEGY_SIGNAL_CACHE.popitem(last=False)
    else:
        _STRATEGY_SIGNAL_CACHE.move_to_end(key)
    return list(_STRATEGY_SIGNAL_CACHE[key])


def evaluate_condition(condition: str, files: list[str]) -> tuple[list[dict], tuple[float, float, int]]:
    """Evaluate one canonical condition into deduplicated signals and exit params."""
    global _LAST_EVALUATION_TRUNCATED
    _LAST_EVALUATION_TRUNCATED = False
    _ensure_market_loader()
    node = parse_boolean(condition)

    # Pure indicator/pattern expressions must be compiled and scanned as one
    # expression. Scanning each AND leaf independently would reject a broad
    # indicator even when a rare candle pattern later narrows it substantially.
    pure_expression = all(
        not leaf.text.strip().startswith("strategy=")
        and not leaf.text.strip().startswith("indicator=")
        for leaf in (node.children or [node])
    ) if node.op != "leaf" else (
        not node.text.strip().startswith("strategy=")
        and not node.text.strip().startswith("indicator=")
    )
    if pure_expression:
        signals = _expression_signals(condition, files)
        if any(sig.get("_signal_truncated") for sig in signals):
            _LAST_EVALUATION_TRUNCATED = True
            return [], (0.08, 0.10, 10)
        return signals, (0.08, 0.10, 10)

    def eval_node(current: BoolNode) -> tuple[list[dict], tuple[float, float, int] | None]:
        if current.op == "leaf":
            text = current.text.strip()
            params = parse_strategy_condition(text)
            if params:
                _, sl, tp, hold = params
                return _strategy_signals(params[0], files), (sl, tp, hold)
            if text.startswith("indicator="):
                return _legacy_l3_condition(text, files), (0.08, 0.10, 10)
            if re.search(r"[><=]", text):
                return _expression_signals(text, files), (0.08, 0.10, 10)
            return _known_strategy_text_signals(text, files), (0.08, 0.10, 10)

        child_results = [eval_node(child) for child in current.children or []]
        if any(
            sig.get("_signal_truncated")
            for signals, _ in child_results
            for sig in signals
        ):
            return [{"_signal_truncated": True}], None
        all_signals: list[dict] = []
        params = None
        strategy_params = [item[1] for item in child_results if item[1] is not None]
        if strategy_params:
            params = strategy_params[0]
        if current.op == "OR":
            seen = set()
            for signals, child_params in child_results:
                for sig in signals:
                    key = (sig.get("code", ""), str(sig.get("signal_date", "")))
                    if key in seen:
                        continue
                    seen.add(key)
                    if child_params:
                        sig = dict(sig)
                        sig.update({"_sl": child_params[0], "_tp": child_params[1], "_hold": child_params[2]})
                    all_signals.append(sig)
        else:
            sets = []
            base_signals = None
            for signals, child_params in child_results:
                key_set = {(sig.get("code", ""), str(sig.get("signal_date", ""))) for sig in signals}
                sets.append(key_set)
                if child_params is not None:
                    base_signals = signals
            if not sets or any(not item for item in sets):
                return [], params
            intersection = set.intersection(*sets)
            source = base_signals if base_signals is not None else child_results[0][0]
            seen = set()
            for sig in source:
                key = (sig.get("code", ""), str(sig.get("signal_date", "")))
                if key in intersection and key not in seen:
                    seen.add(key)
                    if params:
                        sig = dict(sig)
                        sig.update({"_sl": params[0], "_tp": params[1], "_hold": params[2]})
                    all_signals.append(sig)
        return all_signals, params

    signals, params = eval_node(node)
    if any(sig.get("_signal_truncated") for sig in signals):
        _LAST_EVALUATION_TRUNCATED = True
        return [], (0.08, 0.10, 10)
    default = (0.08, 0.10, 10)
    if params is None:
        params = default
    return signals, params


def _full_market_map(files: list[str]) -> dict | None:
    global _FULL_MARKET_DF_MAP, _FULL_MARKET_DF_KEY
    key = tuple(sorted(files))
    if _FULL_MARKET_DF_MAP is not None and _FULL_MARKET_DF_KEY == key:
        return _FULL_MARKET_DF_MAP
    from market_data import MarketDataLoader
    loader = MarketDataLoader.get_instance()
    loader.ensure_loaded(start_date=FULL_BACKTEST_START, end_date=FULL_BACKTEST_END)
    # Hold the singleton's original frames directly. build_stock_df_map copies
    # every frame after filtering and would duplicate the full-market cache.
    result = loader.get_stock_df_map()
    if len(result) < 50:
        return None
    _FULL_MARKET_DF_MAP = result
    _FULL_MARKET_DF_KEY = key
    return result


def _ensure_market_loader() -> None:
    """Load once before fork scans so children inherit copy-on-write pages."""
    from market_data import MarketDataLoader
    MarketDataLoader.get_instance().ensure_loaded(
        start_date=FULL_BACKTEST_START,
        end_date=FULL_BACKTEST_END,
    )


def yearly_signal_stats(signals: list[dict]) -> dict:
    seen = set()
    yearly = {year: 0 for year in FULL_BACKTEST_YEARS}
    for sig in signals:
        date = str(sig.get("signal_date", ""))[:10]
        if date[:4] not in yearly:
            continue
        key = (sig.get("code", ""), date)
        if key in seen:
            continue
        seen.add(key)
        yearly[date[:4]] += 1
    return {
        "yearly_signal_count": yearly,
        "min_yearly_signal_count": min(yearly.values()) if yearly else 0,
        "signal_count": len(seen),
    }


def development_robustness_stats(trades: list[dict]) -> dict:
    """Measure whether development-period performance is distributed across years."""
    from backtest_framework import calc_yearly_stats

    yearly = calc_yearly_stats(trades)
    pfs = [float(item.get("pf", 0)) for item in yearly.values()]
    if not pfs:
        return {
            "years_with_trades": 0,
            "positive_year_ratio": 0.0,
            "worst_year_pf": 0.0,
            "passed": False,
        }

    positive_ratio = sum(1 for pf in pfs if pf > 1.0) / len(pfs)
    passed = positive_ratio >= 0.60 and min(pfs) > 0.75
    return {
        "yearly_trade_stats": yearly,
        "years_with_trades": len(pfs),
        "positive_year_ratio": round(positive_ratio, 3),
        "worst_year_pf": round(min(pfs), 2),
        "passed": bool(passed),
    }


def run_unified_backtest(
    condition: str,
    files: list[str],
    direction: str = "long",
) -> dict:
    """Run the single source of truth backtest for any L1-L4 condition."""
    from backtest_framework import calc_trade_stats, simulate_trades_v2

    raw, default_params = evaluate_condition(condition, files)
    signals = [
        sig for sig in raw
        if FULL_BACKTEST_START <= str(sig.get("signal_date", ""))[:10] <= FULL_BACKTEST_END
    ]
    seen = set()
    deduped = []
    for sig in signals:
        key = (sig.get("code", ""), str(sig.get("signal_date", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sig)
    annual = yearly_signal_stats(deduped)
    global _LAST_EVALUATION_TRUNCATED
    if _LAST_EVALUATION_TRUNCATED:
        gc.collect()
        return {
            "passed": False,
            "failure_reason": (
                f"Signal-width gate: one scan worker exceeded "
                f"{_MAX_SIGNALS_PER_SCAN_WORKER} signals; expression is too broad"
            ),
            "method": "unified_full_market_backtest",
            "signal_truncated": True,
        }
    if annual["min_yearly_signal_count"] < MIN_SIGNALS_PER_FULL_YEAR:
        gc.collect()
        return {
            "passed": False,
            "failure_reason": (
                f"Annual signal gate: min {annual['min_yearly_signal_count']}/year "
                f"< {MIN_SIGNALS_PER_FULL_YEAR}; sparse signal rejected"
            ),
            "method": "unified_full_market_backtest",
            **annual,
        }
    if max(annual["yearly_signal_count"].values()) > MAX_SIGNALS_PER_FULL_YEAR:
        gc.collect()
        return {
            "passed": False,
            "failure_reason": (
                f"Annual signal-width gate: max {max(annual['yearly_signal_count'].values())}/year "
                f"> {MAX_SIGNALS_PER_FULL_YEAR}; over-broad expression rejected"
            ),
            "method": "unified_full_market_backtest",
            **annual,
        }

    df_map = _full_market_map(files)
    if df_map is None:
        return {"passed": False, "failure_reason": "Full-market data unavailable", "method": "unified_full_market_backtest"}

    groups: dict[tuple[float, float, int], list[dict]] = {}
    for sig in deduped:
        params = (
            float(sig.get("_sl", default_params[0])),
            float(sig.get("_tp", default_params[1])),
            int(sig.get("_hold", default_params[2])),
        )
        clean = {k: v for k, v in sig.items() if not k.startswith("_")}
        groups.setdefault(params, []).append(clean)

    trades = []
    for (sl, tp, hold), group_signals in groups.items():
        if direction == "short":
            from backtest_framework import simulate_short_trades
            group_trades = simulate_short_trades(group_signals, df_map, hold_days=hold, tp=tp, sl=sl)
        else:
            group_trades = simulate_trades_v2(group_signals, df_map, hold_days=hold, tp=tp, sl=-sl, cooldown=10)
        trades.extend(group_trades)
    stats = calc_trade_stats(trades)
    robustness = development_robustness_stats(trades)
    gc.collect()
    pf = stats["pf"]
    passed = (
        pf >= 1.15
        and stats["count"] >= 100
        and robustness["passed"]
    )
    reasons = []
    if pf < 1.15:
        reasons.append(f"Unified backtest PF={pf:.2f}, below 1.15")
    if stats["count"] < 100:
        reasons.append(f"Only {stats['count']} trades (need >=100)")
    if not robustness["passed"]:
        reasons.append(
            "Annual robustness gate failed: "
            f"positive_years={robustness['positive_year_ratio']:.0%}, "
            f"worst_year_pf={robustness['worst_year_pf']:.2f}"
        )
    return {
        "passed": passed,
        "failure_reason": "; ".join(reasons) if reasons else None,
        "method": "unified_full_market_backtest",
        "test_pf": round(pf, 2),
        "win_rate": round(stats["win_rate"], 1),
        "trade_count": stats["count"],
        "development_robustness": robustness,
        "avg_return": round(stats["avg_return"], 2),
        **annual,
        "exit_groups": [
            {"sl": sl, "tp": tp, "hold": hold, "signals": len(group)}
            for (sl, tp, hold), group in groups.items()
        ],
        "logic_text": describe_condition(condition, direction),
    }


__all__ = [
    "parse_boolean", "describe_condition", "evaluate_condition",
    "generate_unified_hypotheses", "run_unified_backtest", "yearly_signal_stats",
]
