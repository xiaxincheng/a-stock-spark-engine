#!/usr/bin/env python3
"""Command-line interface for the A-Stock Spark Engine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from enum import Enum

from backtest_framework import (
    build_stock_df_map,
    calc_trade_stats,
    calc_yearly_stats,
    scan_all_csvs,
    simulate_short_trades,
    simulate_trades_v2,
)
from spark_core.indicator_registry import indicator_counts, indicator_manifest
from spark_core.layered_generator import generate_l1
from spark_core.strategy_family import group_families, structure_key
from spark_core.unified_engine import describe_condition, evaluate_condition, run_unified_backtest


def _set_data_dir(path: str | None) -> None:
    if path:
        os.environ["SPARK_DATA_DIR"] = path


def _dump_hypotheses(items: list) -> None:
    def encode(value):
        if isinstance(value, Enum):
            return value.value
        raise TypeError(type(value).__name__)

    print(json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2, default=encode))


def cmd_catalog(_: argparse.Namespace) -> None:
    print(json.dumps({
        "counts": indicator_counts(),
        "total": len(indicator_manifest()),
        "indicators": indicator_manifest(),
    }, ensure_ascii=False, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    _set_data_dir(args.data_dir)
    items = generate_l1(count=args.count, direction=args.direction, seed=args.seed)
    if args.ledger:
        from spark_ledger import ExperimentLedger
        ledger = ExperimentLedger(args.ledger)
        for item in items:
            ledger.save_hypothesis(item)
            ledger.record_event(item.id, "generated", {"layer": item.generation_layer})
    _dump_hypotheses(items)


def _simulate_groups(signals: list[dict], direction: str, frames: dict, defaults: tuple[float, float, int]) -> list[dict]:
    groups: dict[tuple[float, float, int], list[dict]] = defaultdict(list)
    for signal in signals:
        params = (
            float(signal.get("_sl", defaults[0])),
            float(signal.get("_tp", defaults[1])),
            int(signal.get("_hold", defaults[2])),
        )
        clean = {key: value for key, value in signal.items() if not key.startswith("_")}
        groups[params].append(clean)
    trades: list[dict] = []
    for stop, target, hold, group in groups.items():
        if direction == "short":
            trades.extend(simulate_short_trades(group, frames, hold_days=hold, tp=target, sl=stop))
        else:
            trades.extend(simulate_trades_v2(group, frames, hold_days=hold, tp=target, sl=-stop, cooldown=10))
    return trades


def cmd_backtest(args: argparse.Namespace) -> None:
    _set_data_dir(args.data_dir)
    files = scan_all_csvs()
    raw, defaults = evaluate_condition(args.expression, files)
    signals = [
        signal for signal in raw
        if (not args.start or str(signal["signal_date"])[:10] >= args.start)
        and (not args.end or str(signal["signal_date"])[:10] <= args.end)
    ]
    frames = build_stock_df_map(files, start_date=args.start, end_date=args.end)
    trades = _simulate_groups(signals, args.direction, frames, defaults)
    print(json.dumps({
        "expression": args.expression,
        "direction": args.direction,
        "stock_count": len(frames),
        "signal_count": len(signals),
        "stats": calc_trade_stats(trades),
        "yearly": calc_yearly_stats(trades),
        "logic": describe_condition(args.expression, args.direction),
    }, ensure_ascii=False, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    _set_data_dir(args.data_dir)
    files = scan_all_csvs()
    result = run_unified_backtest(args.expression, files, direction=args.direction)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_family(args: argparse.Namespace) -> None:
    candidates = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    families = group_families(candidates)
    print(json.dumps({
        str(key): [{"id": item.get("id"), "condition_expr": item.get("condition_expr")} for item in members]
        for key, members in families.items()
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Indicator-based A-share strategy research engine")
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog", help="show the executable indicator catalog")
    catalog.set_defaults(func=cmd_catalog)

    generate = commands.add_parser("generate", help="generate L1 indicator expressions")
    generate.add_argument("--count", type=int, default=3)
    generate.add_argument("--direction", choices=("long", "short", "both"), default="both")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--data-dir")
    generate.add_argument("--ledger", help="optional SQLite ledger path")
    generate.set_defaults(func=cmd_generate)

    backtest = commands.add_parser("backtest", help="evaluate one expression over local CSV data")
    backtest.add_argument("expression")
    backtest.add_argument("--direction", choices=("long", "short"), default="long")
    backtest.add_argument("--data-dir")
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.set_defaults(func=cmd_backtest)

    verify = commands.add_parser("verify", help="run annual signal and robustness gates")
    verify.add_argument("expression")
    verify.add_argument("--direction", choices=("long", "short"), default="long")
    verify.add_argument("--data-dir")
    verify.set_defaults(func=cmd_verify)

    family = commands.add_parser("family", help="group a JSON candidate array by strategy family")
    family.add_argument("candidates_json", type=Path)
    family.set_defaults(func=cmd_family)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
