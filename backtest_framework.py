"""Portable OHLCV loading and next-open backtest primitives.

The engine deliberately separates strategy research from brokerage execution.
This module only evaluates historical signals; it does not place, queue, or
manage live orders.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_DIR = Path(os.environ.get("SPARK_DATA_DIR", "data/daily"))
COST_BUY = float(os.environ.get("SPARK_COST_BUY", "0.0007"))
COST_SELL = float(os.environ.get("SPARK_COST_SELL", "0.0017"))
REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def stock_code(csv_path: str | Path) -> str:
    """Infer an exchange-prefixed code from a CSV path."""
    path = str(Path(csv_path))
    stem = Path(path).stem
    for market in ("sh", "sz", "bj"):
        if f"/{market}/" in path or path.startswith(market):
            return f"{market}{stem}"
    if stem.startswith(("6", "9")):
        return f"sh{stem}"
    if stem.startswith(("0", "2", "3")):
        return f"sz{stem}"
    return stem


def scan_all_csvs(data_dir: str | Path | None = None) -> list[str]:
    """Find CSV files in either ``sh/`` + ``sz/`` layout or a flat directory."""
    root = Path(data_dir or os.environ.get("SPARK_DATA_DIR") or DATA_DIR)
    if not root.exists():
        return []
    market_files = [file for market in ("sh", "sz", "bj") for file in sorted((root / market).glob("*.csv"))]
    if market_files:
        return [str(path) for path in market_files]
    return [str(path) for path in sorted(root.rglob("*.csv"))]


def apply_forward_adjustment(df: pd.DataFrame, code: str = "") -> pd.DataFrame:
    """Heuristically stitch large ex-rights open gaps.

        This is not a substitute for an exchange-provided adjustment factor. It is
        only used when ``SPARK_FORWARD_ADJUST=1`` and the user explicitly opts in.
    """
    if len(df) < 2:
        return df

    bare = code[2:] if code[:2] in {"sh", "sz", "bj"} else code
    if bare.startswith(("688", "689", "300", "301")):
        threshold = 0.205
    else:
        threshold = 0.105

    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    factors = np.ones(len(df), dtype=float)
    for i in range(1, len(df)):
        if close[i - 1] <= 0 or open_[i] <= 0:
            continue
        if abs((open_[i] - close[i - 1]) / close[i - 1]) > threshold:
            factors[i:] *= close[i - 1] / open_[i]

    result = df.copy()
    if np.any(factors != 1.0):
        for col in ("open", "high", "low", "close"):
            result[col] = result[col].astype(float) * factors
    return result


def load_stock_data(
    csv_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    fq: bool | None = None,
) -> pd.DataFrame | None:
    """Load one clean OHLCV frame.

    The repository does not bundle real market data. Users should normally
    provide already-adjusted OHLCV CSV files.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if not all(column in df.columns for column in REQUIRED_COLUMNS):
        return None

    df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[column for column in REQUIRED_COLUMNS if column != "volume"])
    df = df[(df["volume"] >= 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if df.empty:
        return None

    if fq is True or (fq is None and os.environ.get("SPARK_FORWARD_ADJUST", "").lower() in {"1", "true", "yes", "on"}):
        df = apply_forward_adjustment(df, stock_code(csv_path))
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    return df.reset_index(drop=True) if len(df) >= 20 else None


def build_stock_df_map(
    csv_files: Iterable[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for path in csv_files:
        frame = load_stock_data(path, start_date=start_date, end_date=end_date)
        if frame is not None:
            result[stock_code(path)] = frame
    return result


def calc_trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {
            "count": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "pf": 0.0,
            "avg_return": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "avg_hold": 0.0,
            "total_return": 0.0, "max_return": 0.0, "min_return": 0.0,
        }
    values = [float(trade["profit_pct"]) for trade in trades]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss else (99.0 if gross_profit else 0.0)
    return {
        "count": len(values),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(values) * 100, 2),
        "pf": round(pf, 2),
        "avg_return": round(float(np.mean(values)), 2),
        "avg_win": round(float(np.mean(wins)) if wins else 0, 2),
        "avg_loss": round(float(np.mean(losses)) if losses else 0, 2),
        "avg_hold": round(float(np.mean([trade.get("hold_days", 0) for trade in trades])), 1),
        "total_return": round(sum(values), 2),
        "max_return": round(max(values), 2),
        "min_return": round(min(values), 2),
    }


def calc_yearly_stats(trades: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade["buy_date"])[:4]].append(float(trade["profit_pct"]))
    result: dict[str, dict] = {}
    for year, values in sorted(grouped.items()):
        result[year] = calc_trade_stats([{"profit_pct": value, "hold_days": 1} for value in values])
    return result


def _limit_threshold(code: str) -> float:
    if code.startswith(("sh688", "sz30")):
        return 0.195
    return 0.095


def _can_buy_at_open(code: str, previous_close: float, open_price: float) -> bool:
    if not os.environ.get("SPARK_ENFORCE_LIMIT_UP", "1").lower() in {"0", "false", "no", "off"}:
        return True
    return previous_close <= 0 or (open_price - previous_close) / previous_close <= _limit_threshold(code)


def simulate_trades_v2(
    signals: list[dict],
    stock_df_map: dict[str, pd.DataFrame],
    hold_days: int = 10,
    tp: float = 0.10,
    sl: float = -0.08,
    cooldown: int = 10,
) -> list[dict]:
    """Simulate long entries at the next open with SL/TP/timeout exits."""
    trades: list[dict] = []
    next_allowed_signal: dict[str, str] = {}
    for signal in sorted(signals, key=lambda item: (str(item["signal_date"]), str(item["code"]))):
        code = str(signal["code"])
        signal_date = str(signal["signal_date"])[:10]
        if code in next_allowed_signal and signal_date < next_allowed_signal[code]:
            continue
        df = stock_df_map.get(code)
        if df is None or signal_date not in set(df["date"]):
            continue
        signal_idx = int(df.index[df["date"] == signal_date][0])
        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            continue
        buy_price = float(df.loc[entry_idx, "open"])
        previous_close = float(df.loc[signal_idx, "close"])
        if buy_price <= 0 or not _can_buy_at_open(code, previous_close, buy_price):
            continue

        end_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_idx = end_idx
        exit_price = float(df.loc[end_idx, "close"])
        exit_reason = "timeout"
        target = buy_price * (1 + tp)
        stop = buy_price * (1 + sl)
        for idx in range(entry_idx + 1, end_idx + 1):
            low = float(df.loc[idx, "low"])
            high = float(df.loc[idx, "high"])
            if low <= stop:
                exit_idx, exit_price, exit_reason = idx, stop, "sl"
                break
            if high >= target:
                exit_idx, exit_price, exit_reason = idx, target, "tp"
                break

        exit_date = str(df.loc[exit_idx, "date"])
        buy_date = str(df.loc[entry_idx, "date"])
        effective_buy = buy_price * (1 + COST_BUY)
        effective_sell = exit_price * (1 - COST_SELL)
        trades.append({
            "code": code,
            "signal_type": signal.get("signal_type", ""),
            "signal_date": signal_date,
            "buy_date": buy_date,
            "sell_date": exit_date,
            "buy_price": round(buy_price, 4),
            "sell_price": round(exit_price, 4),
            "profit_pct": round((effective_sell - effective_buy) / effective_buy * 100, 2),
            "hold_days": max(1, (pd.Timestamp(exit_date) - pd.Timestamp(buy_date)).days),
            "sell_reason": exit_reason,
        })
        next_allowed_signal[code] = exit_date
    return trades


def simulate_short_trades(
    signals: list[dict],
    stock_df_map: dict[str, pd.DataFrame],
    hold_days: int = 10,
    tp: float = 0.10,
    sl: float = -0.08,
    cooldown: int = 10,
) -> list[dict]:
    """Research-only short simulation; borrow availability and fees are excluded."""
    trades: list[dict] = []
    next_allowed_signal: dict[str, str] = {}
    sl_abs = abs(sl)
    for signal in sorted(signals, key=lambda item: (str(item["signal_date"]), str(item["code"]))):
        code = str(signal["code"])
        signal_date = str(signal["signal_date"])[:10]
        if code in next_allowed_signal and signal_date < next_allowed_signal[code]:
            continue
        df = stock_df_map.get(code)
        if df is None or signal_date not in set(df["date"]):
            continue
        signal_idx = int(df.index[df["date"] == signal_date][0])
        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            continue
        short_price = float(df.loc[entry_idx, "open"])
        if short_price <= 0:
            continue
        target = short_price * (1 - tp)
        stop = short_price * (1 + sl_abs)
        end_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_idx, exit_price, exit_reason = end_idx, float(df.loc[end_idx, "close"]), "timeout"
        for idx in range(entry_idx + 1, end_idx + 1):
            open_price = float(df.loc[idx, "open"])
            high = float(df.loc[idx, "high"])
            low = float(df.loc[idx, "low"])
            if open_price >= stop:
                exit_idx, exit_price, exit_reason = idx, open_price, "gap_sl"
                break
            if open_price <= target:
                exit_idx, exit_price, exit_reason = idx, open_price, "gap_tp"
                break
            if high >= stop:
                exit_idx, exit_price, exit_reason = idx, stop, "sl"
                break
            if low <= target:
                exit_idx, exit_price, exit_reason = idx, target, "tp"
                break

        exit_date = str(df.loc[exit_idx, "date"])
        entry_date = str(df.loc[entry_idx, "date"])
        effective_entry = short_price * (1 - COST_SELL)
        effective_cover = exit_price * (1 + COST_BUY)
        trades.append({
            "code": code,
            "signal_type": signal.get("signal_type", ""),
            "signal_date": signal_date,
            "buy_date": entry_date,
            "sell_date": exit_date,
            "buy_price": round(short_price, 4),
            "sell_price": round(exit_price, 4),
            "profit_pct": round((effective_entry - effective_cover) / effective_entry * 100, 2),
            "hold_days": max(1, (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days),
            "sell_reason": exit_reason,
        })
        next_allowed_signal[code] = exit_date
    return trades
