"""Executable indicator registry for Spark strategy generation.

The registry deliberately exposes indicator *functions*, not precomputed
columns.  L1 can therefore compose new expressions while the verifier uses one
implementation for generation and backtesting.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:  # pragma: no cover - TA-Lib is available in production.
    talib = None


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    params: tuple[tuple[str, float, float], ...]
    category: str
    description: str


FUNCTION_REGISTRY: dict[str, tuple[tuple[tuple[str, float, float], ...], object]] = {}
PATTERN_FUNCTIONS: dict[str, object] = {}
INDICATOR_META: dict[str, IndicatorSpec] = {}


def _register(
    name: str,
    params: tuple[tuple[str, float, float], ...],
    category: str,
    description: str,
    computer: object,
) -> None:
    FUNCTION_REGISTRY[name] = (params, computer)
    INDICATOR_META[name] = IndicatorSpec(name, params, category, description)


def _register_pattern(name: str, category: str, description: str, computer: object) -> None:
    PATTERN_FUNCTIONS[name] = computer
    INDICATOR_META[name] = IndicatorSpec(name, (), category, description)


def _required_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"indicator input column missing: {col}")
    return pd.to_numeric(df[col], errors="coerce")


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = _required_col(df, "high"), _required_col(df, "low"), _required_col(df, "close")
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(int(n), min_periods=1).mean()


def _rsi(df: pd.DataFrame, n: int) -> pd.Series:
    close = _required_col(df, "close")
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / int(n), adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / int(n), adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _cci(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = _required_col(df, "high"), _required_col(df, "low"), _required_col(df, "close")
    tp = (high + low + close) / 3
    ma = tp.rolling(int(n), min_periods=1).mean()
    md = tp.rolling(int(n), min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _willr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = _required_col(df, "high"), _required_col(df, "low"), _required_col(df, "close")
    hh = high.rolling(int(n), min_periods=1).max()
    ll = low.rolling(int(n), min_periods=1).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def _consecutive(df: pd.DataFrame, direction: str) -> pd.Series:
    close = _required_col(df, "close")
    result = pd.Series(0, index=df.index, dtype=float)
    count = 0
    values = close.to_numpy()
    for i, value in enumerate(values):
        if i == 0 or (direction == "up" and value > values[i - 1]) or (
            direction == "down" and value < values[i - 1]
        ):
            count += 1
        else:
            count = 1
        result.iloc[i] = count
    return result


_register("MA", (("N", 1, 610),), "均线", "N日简单移动平均", lambda df, N: _required_col(df, "close").rolling(int(N), min_periods=1).mean())
_register("MA_slope", (("N", 1, 250), ("D", 1, 60)), "均线", "N日均线D日变化率", lambda df, N, D: _required_col(df, "close").rolling(int(N), min_periods=1).mean().pct_change(int(D)))
_register("volume_ratio", (("N", 1, 120),), "量能", "成交量/N日均量", lambda df, N: _required_col(df, "volume") / _required_col(df, "volume").rolling(int(N), min_periods=1).mean().replace(0, np.nan))
_register("ATR", (("N", 1, 250),), "波动", "平均真实波幅", _atr)
_register("amplitude", (("N", 1, 120),), "波动", "N日高低振幅/均价", lambda df, N: (_required_col(df, "high").rolling(int(N), min_periods=1).max() - _required_col(df, "low").rolling(int(N), min_periods=1).min()) / _required_col(df, "close").rolling(int(N), min_periods=1).mean())
_register("std", (("N", 1, 120),), "波动", "N日收盘价变异系数", lambda df, N: _required_col(df, "close").rolling(int(N), min_periods=1).std() / _required_col(df, "close").rolling(int(N), min_periods=1).mean())
_register("RSI", (("N", 2, 120),), "动量", "相对强弱指标", _rsi)
_register("CCI", (("N", 2, 120),), "动量", "商品通道指标", _cci)
_register("WR", (("N", 2, 120),), "动量", "威廉指标，-100到0", _willr)
_register("BIAS", (("N", 1, 250),), "均值回归", "N日乖离率", lambda df, N: _required_col(df, "close").pct_change(0) / _required_col(df, "close").rolling(int(N), min_periods=1).mean() - 1)
_register("pct_change", (("N", 1, 120),), "动量", "N日涨跌幅", lambda df, N: _required_col(df, "close").pct_change(int(N)))
_register("ma_spread", (("N", 1, 610), ("M", 1, 610)), "均线", "N/M均线距离百分比", lambda df, N, M: (_required_col(df, "close").rolling(int(N), min_periods=1).mean() - _required_col(df, "close").rolling(int(M), min_periods=1).mean()).abs() / _required_col(df, "close").rolling(int(M), min_periods=1).mean().replace(0, np.nan))
_register("up_days", (("N", 1, 120),), "资金流", "近N日上涨天数", lambda df, N: (_required_col(df, "close") > _required_col(df, "close").shift(1)).rolling(int(N), min_periods=1).sum())
_register("down_days", (("N", 1, 120),), "资金流", "近N日下跌天数", lambda df, N: (_required_col(df, "close") < _required_col(df, "close").shift(1)).rolling(int(N), min_periods=1).sum())
_register("consecutive_up", (), "资金流", "连续上涨天数", lambda df: _consecutive(df, "up"))
_register("consecutive_down", (), "资金流", "连续下跌天数", lambda df: _consecutive(df, "down"))
_register("volume_trend", (("N", 1, 60),), "量能", "N日均量变化率", lambda df, N: _required_col(df, "volume").rolling(int(N), min_periods=1).mean().pct_change(1))
_register("bollinger_width", (("N", 1, 120),), "波动", "布林带宽度/中轨", lambda df, N: 4 * _required_col(df, "close").rolling(int(N), min_periods=1).std() / _required_col(df, "close").rolling(int(N), min_periods=1).mean())
_register("turnover_rate", (), "量能", "换手率原始字段", lambda df: df["turnover_rate"] if "turnover_rate" in df else (df["turnover"] if "turnover" in df else pd.Series(np.nan, index=df.index)))
_register("gap_up", (), "形态", "向上跳空超过0.5%", lambda df: _required_col(df, "open") > _required_col(df, "close").shift(1) * 1.005)
_register("gap_down", (), "形态", "向下跳空超过0.5%", lambda df: _required_col(df, "open") < _required_col(df, "close").shift(1) * 0.995)


def _is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    return (prev["close"] < prev["open"]) & (df["close"] > df["open"]) & (df["open"] <= prev["close"]) & (df["close"] >= prev["open"])


def _candle_body_shadow(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    body = (_required_col(df, "close") - _required_col(df, "open")).abs()
    lower = pd.concat([_required_col(df, "open"), _required_col(df, "close")], axis=1).min(axis=1) - _required_col(df, "low")
    upper = _required_col(df, "high") - pd.concat([_required_col(df, "open"), _required_col(df, "close")], axis=1).max(axis=1)
    return body, lower, upper


def _is_hammer(df: pd.DataFrame) -> pd.Series:
    body, lower, upper = _candle_body_shadow(df)
    return (lower > body * 2) & (upper < body * 0.5) & (body > 0)


def _is_morning_star(df: pd.DataFrame) -> pd.Series:
    prev2, prev1 = df.shift(2), df.shift(1)
    return (prev2["close"] < prev2["open"]) & ((prev1["close"] - prev1["open"]).abs() < prev1["close"] * 0.01) & (df["close"] > df["open"]) & (df["close"] > (prev2["open"] + prev2["close"]) / 2)


def _is_hanging_man(df: pd.DataFrame) -> pd.Series:
    body, lower, upper = _candle_body_shadow(df)
    return _is_hammer(df) & (df["close"].shift(1) > df["open"].shift(1))


def _is_evening_star(df: pd.DataFrame) -> pd.Series:
    prev2, prev1 = df.shift(2), df.shift(1)
    return (prev2["close"] > prev2["open"]) & ((prev1["close"] - prev1["open"]).abs() < prev1["close"] * 0.01) & (df["close"] < df["open"]) & (df["close"] < (prev2["open"] + prev2["close"]) / 2)


def _is_shooting_star(df: pd.DataFrame) -> pd.Series:
    body, lower, upper = _candle_body_shadow(df)
    return (upper > body * 2) & (lower < body * 0.5) & (body > 0)


for _name, _fn, _desc in [
    ("is_bullish_engulfing", _is_bullish_engulfing, "看涨吞没"),
    ("is_hammer", _is_hammer, "锤子线"),
    ("is_morning_star", _is_morning_star, "启明星"),
    ("is_hanging_man", _is_hanging_man, "上吊线"),
    ("is_evening_star", _is_evening_star, "黄昏星"),
    ("is_shooting_star", _is_shooting_star, "射击之星"),
]:
    _register_pattern(_name, "K线形态", _desc, _fn)


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=int(n), adjust=False, min_periods=1).mean()


def _kdj_part(part: str):
    def compute(df: pd.DataFrame, N: float, M1: float, M2: float) -> pd.Series:
        high, low, close = _required_col(df, "high"), _required_col(df, "low"), _required_col(df, "close")
        hh = high.rolling(int(N), min_periods=1).max()
        ll = low.rolling(int(N), min_periods=1).min()
        rsv = (close - ll) / (hh - ll).replace(0, np.nan) * 100
        k = rsv.ewm(alpha=1 / int(M1), adjust=False).mean()
        d = k.ewm(alpha=1 / int(M2), adjust=False).mean()
        j = 3 * k - 2 * d
        return {"K": k, "D": d, "J": j}[part]
    return compute


for _part in ("K", "D", "J"):
    _register(
        f"KDJ_{_part}",
        (("N", 2, 120), ("M1", 2, 60), ("M2", 2, 60)),
        "A股动量",
        f"KDJ-{_part}线",
        _kdj_part(_part),
    )


def _skdj_part(part: str):
    def compute(df: pd.DataFrame, N: float, M: float) -> pd.Series:
        high, low, close = _required_col(df, "high"), _required_col(df, "low"), _required_col(df, "close")
        hh = high.rolling(int(N), min_periods=1).max()
        ll = low.rolling(int(N), min_periods=1).min()
        rsv = (close - ll) / (hh - ll).replace(0, np.nan) * 100
        k = rsv.rolling(int(M), min_periods=1).mean()
        d = k.rolling(int(M), min_periods=1).mean()
        j = 3 * k - 2 * d
        return {"K": k, "D": d, "J": j}[part]
    return compute


for _part in ("K", "D", "J"):
    _register(f"SKDJ_{_part}", (("N", 2, 120), ("M", 2, 60)), "A股动量", f"慢速KDJ-{_part}线", _skdj_part(_part))


_register("BBI", (("N1", 2, 120), ("N2", 2, 180), ("N3", 2, 240), ("N4", 2, 360)), "A股均线", "多空指标BBI", lambda df, N1, N2, N3, N4: sum(_required_col(df, "close").rolling(int(x), min_periods=1).mean() for x in (N1, N2, N3, N4)) / 4)


def _ene_band(kind: str):
    def compute(df: pd.DataFrame, N: float, M1: float, M2: float) -> pd.Series:
        close, high, low = _required_col(df, "close"), _required_col(df, "high"), _required_col(df, "low")
        mid = (high + low + close).rolling(int(N), min_periods=1).mean() / 3
        return {"UPPER": mid * (1 + M1 / 100), "MID": mid, "LOWER": mid * (1 - M2 / 100)}[kind]
    return compute


for _band in ("UPPER", "MID", "LOWER"):
    _register(f"ENE_{_band}", (("N", 2, 120), ("M1", 1, 50), ("M2", 1, 50)), "A股轨道", f"ENE轨道-{_band}", _ene_band(_band))


_register("PSY", (("N", 2, 120),), "A股情绪", "心理线，上涨天数占比*100", lambda df, N: (_required_col(df, "close") > _required_col(df, "close").shift(1)).rolling(int(N), min_periods=1).mean() * 100)
_register("VR", (("N", 2, 180),), "A股量能", "成交量比率", lambda df, N: (_required_col(df, "volume").where(_required_col(df, "close") > _required_col(df, "close").shift(1)).rolling(int(N), min_periods=1).sum() / _required_col(df, "volume").where(_required_col(df, "close") < _required_col(df, "close").shift(1)).rolling(int(N), min_periods=1).sum().replace(0, np.nan)) * 100)
_register("CR", (("N", 2, 180),), "A股量能", "中间意愿指标", lambda df, N: (_required_col(df, "high").sub(_required_col(df, "close").shift(1)).clip(lower=0).rolling(int(N), min_periods=1).sum() / _required_col(df, "close").shift(1).sub(_required_col(df, "low")).clip(lower=0).rolling(int(N), min_periods=1).sum().replace(0, np.nan)) * 200)
_register("AR", (("N", 2, 180),), "A股情绪", "人气指标", lambda df, N: (_required_col(df, "high").sub(_required_col(df, "open")).clip(lower=0).rolling(int(N), min_periods=1).sum() / _required_col(df, "open").sub(_required_col(df, "low")).clip(lower=0).rolling(int(N), min_periods=1).sum().replace(0, np.nan)) * 100)
_register("BR", (("N", 2, 180),), "A股情绪", "意愿指标", lambda df, N: (_required_col(df, "high").sub(_required_col(df, "close").shift(1)).clip(lower=0).rolling(int(N), min_periods=1).sum() / _required_col(df, "close").shift(1).sub(_required_col(df, "low")).clip(lower=0).rolling(int(N), min_periods=1).sum().replace(0, np.nan)) * 100)
_register("PVT", (), "A股量能", "价量趋势指标", lambda df: (_required_col(df, "close").pct_change().fillna(0) * _required_col(df, "volume")).cumsum())
_register("WVAD", (("N", 2, 180),), "A股量能", "威廉变异离散量", lambda df, N: (((_required_col(df, "close") - _required_col(df, "low")) - (_required_col(df, "high") - _required_col(df, "close"))) / (_required_col(df, "high") - _required_col(df, "low")).replace(0, np.nan) * _required_col(df, "volume")).rolling(int(N), min_periods=1).sum())
_register("VSTD", (("N", 2, 180),), "量能", "成交量变异系数", lambda df, N: _required_col(df, "volume").rolling(int(N), min_periods=2).std() / _required_col(df, "volume").rolling(int(N), min_periods=1).mean().replace(0, np.nan))


def _hma(df: pd.DataFrame, n: int) -> pd.Series:
    close = _required_col(df, "close")

    def wma(x: pd.Series, period: int) -> pd.Series:
        return x.rolling(int(period), min_periods=1).apply(lambda v: np.average(v, weights=np.arange(1, len(v) + 1)), raw=True)

    half, sqrt_n = max(1, int(n) // 2), max(1, int(np.sqrt(n)))
    return wma(2 * wma(close, half) - wma(close, int(n)), sqrt_n)


_register("HMA", (("N", 4, 250),), "均线", "Hull均线", _hma)


def _supertrend(df: pd.DataFrame, n: float, mult: float) -> pd.Series:
    close, high, low = _required_col(df, "close"), _required_col(df, "high"), _required_col(df, "low")
    atr = _atr(df, int(n))
    mid = (high + low) / 2
    upper, lower = mid + mult * atr, mid - mult * atr
    direction = pd.Series(np.nan, index=df.index)
    trend_up = True
    for i in range(len(df)):
        if i == 0 or np.isnan(atr.iloc[i]):
            direction.iloc[i] = 1 if trend_up else -1
            continue
        if close.iloc[i] > upper.iloc[i - 1]:
            trend_up = True
        elif close.iloc[i] < lower.iloc[i - 1]:
            trend_up = False
        direction.iloc[i] = 1 if trend_up else -1
    return direction


_register("SUPERTREND", (("N", 4, 120), ("MULT", 0.5, 10)), "趋势", "超级趋势方向，1多/-1空", _supertrend)


def _chop(df: pd.DataFrame, n: int) -> pd.Series:
    atr = _atr(df, 1)
    tr_sum = atr.rolling(int(n), min_periods=1).sum()
    hh = _required_col(df, "high").rolling(int(n), min_periods=1).max()
    ll = _required_col(df, "low").rolling(int(n), min_periods=1).min()
    return 100 * np.log10(tr_sum / (hh - ll).replace(0, np.nan))


_register("CHOP", (("N", 5, 120),), "趋势", "盘整指数，高值震荡/低值趋势", _chop)
_register("DONCHIAN_POS", (("N", 5, 250),), "趋势", "收盘在N日唐奇安通道中的位置0-1", lambda df, N: (_required_col(df, "close") - _required_col(df, "low").rolling(int(N), min_periods=1).min()) / (_required_col(df, "high").rolling(int(N), min_periods=1).max() - _required_col(df, "low").rolling(int(N), min_periods=1).min()).replace(0, np.nan))


if talib is not None:
    _TA_EXCLUDED = {
        # Arithmetic transforms are not standalone stock indicators here.
        "ADD", "DIV", "MAX", "MAXINDEX", "MIN", "MININDEX", "MINMAX", "MINMAXINDEX", "MULT", "SUB", "SUM",
        "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH", "EXP", "FLOOR", "LN", "LOG10", "SIN", "SINH", "SQRT", "TAN", "TANH",
        # Vector-period and extended SAR variants are awkward for LLM DSL.
        "MAVP", "SAREXT", "MA", "CCI", "RSI", "ATR", "MACDEXT",
    }
    _TA_SINGLE_PERIOD = {
        "ADX", "ADXR", "APO", "AROONOSC", "CMO", "DX", "MFI", "MINUS_DI", "MINUS_DM",
        "MOM", "PLUS_DI", "PLUS_DM", "PPO", "ROC", "ROCP", "ROCR", "ROCR100",
        "TRIX", "WILLR", "DEMA", "EMA", "HT_TRENDLINE", "KAMA", "MIDPOINT", "MIDPRICE",
        "SMA", "T3", "TEMA", "TRIMA", "WMA", "BETA", "CORREL", "LINEARREG",
        "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", "LINEARREG_SLOPE", "STDDEV", "TSF", "VAR", "NATR",
    }
    _TA_NO_PARAM = {"AD", "OBV", "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE", "TRANGE", "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDMODE", "HT_TRENDLINE"}

    _TA_INPUTS = {
        "real": ["close"], "real0": ["close"], "real1": ["volume"],
        "open": ["open"], "high": ["high"], "low": ["low"], "close": ["close"], "volume": ["volume"],
    }

    def _ta_inputs(func_name: str, df: pd.DataFrame) -> list[np.ndarray]:
        params = inspect.signature(getattr(talib, func_name)).parameters
        result: list[np.ndarray] = []
        for name in params:
            if name not in _TA_INPUTS:
                break
            result.extend(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64) for col in _TA_INPUTS[name])
        if not result:
            result = [_required_col(df, "close").to_numpy(dtype=np.float64)]
        return result

    # TA-Lib aliases with clear scalar outputs and A-stock friendly names.
    def _macd_part(part: int):
        def compute(df: pd.DataFrame, fast: float, slow: float, signal: float) -> pd.Series:
            close = _required_col(df, "close").to_numpy(dtype=np.float64)
            return pd.Series(talib.MACD(close, int(fast), int(slow), int(signal))[part], index=df.index)
        return compute

    for _part, _idx, _label in [("DIF", 0, "快线"), ("DEA", 1, "慢线"), ("HIST", 2, "柱")]:
        _register(f"MACD_{_part}", (("FAST", 2, 100), ("SLOW", 3, 200), ("SIGNAL", 2, 100)), "MACD", f"MACD-{_label}", _macd_part(_idx))

    def _bband(kind: int, label: str):
        def compute(df: pd.DataFrame, N: float, K: float) -> pd.Series:
            close = _required_col(df, "close").to_numpy(dtype=np.float64)
            return pd.Series(talib.BBANDS(close, int(N), float(K), float(K))[kind], index=df.index)
        return compute

    for _kind, _idx in [("UPPER", 0), ("MID", 1), ("LOWER", 2)]:
        _register(f"BOLL_{_kind}", (("N", 2, 200), ("K", 0.5, 5)), "布林轨道", f"布林带{_kind}", _bband(_idx, _kind))

    def _stoch_part(part: int):
        def compute(df: pd.DataFrame, fast: float, slow: float, slow_d: float) -> pd.Series:
            arrays = [_required_col(df, c).to_numpy(dtype=np.float64) for c in ("high", "low", "close")]
            return pd.Series(talib.STOCH(*arrays, int(fast), int(slow), 0, int(slow_d), 0)[part], index=df.index)
        return compute

    for _part, _idx in [("K", 0), ("D", 1)]:
        _register(f"STOCH_{_part}", (("FAST", 2, 100), ("SLOW", 2, 100), ("SLOWD", 2, 100)), "随机指标", f"TA-Lib STOCH-{_part}", _stoch_part(_idx))

    def _aroon_part(part: int):
        def compute(df: pd.DataFrame, N: float) -> pd.Series:
            arrays = [_required_col(df, c).to_numpy(dtype=np.float64) for c in ("high", "low")]
            return pd.Series(talib.AROON(*arrays, int(N))[part], index=df.index)
        return compute

    for _part, _idx in [("UP", 1), ("DOWN", 0)]:
        _register(f"AROON_{_part}", (("N", 2, 200),), "趋势", f"AROON-{_part}", _aroon_part(_idx))

    _register("SAR", (("ACCEL", 0.001, 0.2), ("MAX", 0.05, 1)), "趋势", "抛物线SAR", lambda df, ACCEL, MAX: pd.Series(talib.SAR(_required_col(df, "high").to_numpy(np.float64), _required_col(df, "low").to_numpy(np.float64), float(ACCEL), float(MAX)), index=df.index))
    _register("ADOSC", (("FAST", 2, 30), ("SLOW", 3, 100)), "量能", "Chaikin A/D Oscillator", lambda df, FAST, SLOW: pd.Series(talib.ADOSC(*[_required_col(df, c).to_numpy(np.float64) for c in ("high", "low", "close", "volume")], int(FAST), int(SLOW)), index=df.index))
    _register("ULTOSC", (("P1", 2, 60), ("P2", 2, 100), ("P3", 2, 200)), "动量", "终极震荡指标", lambda df, P1, P2, P3: pd.Series(talib.ULTOSC(*[_required_col(df, c).to_numpy(np.float64) for c in ("high", "low", "close")], int(P1), int(P2), int(P3)), index=df.index))

    _TA_PARAM_MAP = {
        "MACDEXT": (("FAST", 2, 100), ("SLOW", 3, 200), ("SIGNAL", 2, 100)),
        "MACDFIX": (("SIGNAL", 2, 100),),
        "MAMA": (("FAST", 0.01, 0.99), ("SLOW", 0.01, 0.99)),
        "STOCHF": (("FAST", 2, 100), ("SLOW", 2, 100)),
        "STOCHRSI": (("N", 2, 100), ("FAST", 2, 100), ("SLOW", 2, 100)),
        "BBANDS": (("N", 2, 200), ("NBDEVUP", 0.5, 5), ("NBDEVDN", 0.5, 5), ("MATYPE", 0, 8)),
    }
    _TA_DESCRIPTIONS = {
        "ADX": "平均趋向指数", "ADXR": "ADX速率", "APO": "绝对价格振荡", "AROONOSC": "AROON振荡",
        "BOP": "买卖力度", "CMO": "钱德动量振荡", "DX": "动向指数", "MFI": "资金流量指标",
        "OBV": "能量潮", "TRIX": "三重指数平滑均线", "ULTOSC": "终极震荡", "WILLR": "威廉指标",
        "SAR": "抛物线转向", "STDDEV": "滚动标准差", "VAR": "滚动方差", "LINEARREG_SLOPE": "线性回归斜率",
    }

    for _func_name in talib.get_functions():
        if _func_name in _TA_EXCLUDED or _func_name in INDICATOR_META:
            continue
        try:
            inspect.signature(getattr(talib, _func_name))
        except (TypeError, ValueError):
            continue
        if _func_name.startswith("CDL"):
            def _pattern_computer(name: str):
                def compute(df: pd.DataFrame) -> pd.Series:
                    arrays = [_required_col(df, c).to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")]
                    out = getattr(talib, name)(*arrays)
                    return pd.Series(out != 0, index=df.index)
                return compute
            _register_pattern(_func_name, "TA-Lib形态", f"TA-Lib形态 {_func_name}", _pattern_computer(_func_name))
            continue
        if _func_name in _TA_NO_PARAM:
            params: tuple[tuple[str, float, float], ...] = ()
        elif _func_name in _TA_PARAM_MAP:
            params = _TA_PARAM_MAP[_func_name]
        elif _func_name in _TA_SINGLE_PERIOD:
            params = (("N", 2, 250),)
        else:
            continue

        def _computer(name: str, arg_count: int):
            def compute(df: pd.DataFrame, *args: float) -> pd.Series:
                inputs = _ta_inputs(name, df)
                output = getattr(talib, name)(*inputs, *args)
                if isinstance(output, tuple):
                    output = output[0]
                return pd.Series(output, index=df.index)
            return compute

        category = "TA-Lib指标"
        description = _TA_DESCRIPTIONS.get(_func_name, f"TA-Lib {_func_name}")
        FUNCTION_REGISTRY[_func_name] = (params, _computer(_func_name, len(params)))
        INDICATOR_META[_func_name] = IndicatorSpec(_func_name, params, category, description)


def indicator_manifest() -> list[dict]:
    """Return a JSON-safe manifest for prompts and the capabilities API."""
    return [
        {
            "name": spec.name,
            "params": [list(p) for p in spec.params],
            "category": spec.category,
            "description": spec.description,
        }
        for spec in INDICATOR_META.values()
    ]


def indicator_catalog_text() -> str:
    grouped: dict[str, list[str]] = {}
    for spec in INDICATOR_META.values():
        sig = spec.name
        if spec.params:
            sig += "(" + ",".join(p[0] for p in spec.params) + ")"
        grouped.setdefault(spec.category, []).append(sig)
    return "\n".join(f"[{category}]\n  " + ", ".join(names) for category, names in grouped.items())


def indicator_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in INDICATOR_META.values():
        counts[spec.category] = counts.get(spec.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


__all__ = [
    "FUNCTION_REGISTRY",
    "PATTERN_FUNCTIONS",
    "INDICATOR_META",
    "indicator_manifest",
    "indicator_catalog_text",
    "indicator_counts",
]
