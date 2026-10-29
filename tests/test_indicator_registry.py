import numpy as np
import pandas as pd

from spark_core.indicator_registry import (
    FUNCTION_REGISTRY,
    PATTERN_FUNCTIONS,
    indicator_counts,
)
from spark_core.spark_l1_contrastive import ExpressionCompiler


def _df(n=300):
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.4, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + rng.random(n),
            "low": np.minimum(open_, close) - rng.random(n),
            "close": close,
            "volume": rng.lognormal(10, 0.2, n),
        }
    )


def test_registry_covers_open_source_indicator_families():
    try:
        import talib  # noqa: F401
    except ImportError:
        talib_available = False
    else:
        talib_available = True

    minimum = 180 if talib_available else 35
    assert len(FUNCTION_REGISTRY) + len(PATTERN_FUNCTIONS) >= minimum
    for name in [
        "MACD_DIF", "MACD_DEA", "MACD_HIST", "KDJ_K", "KDJ_D", "KDJ_J",
        "ADX", "AROON_UP", "OBV", "MFI", "TRIX", "SAR", "SMA", "STDDEV",
        "BOLL_UPPER", "ENE_LOWER", "VR", "PSY",
    ]:
        assert name in FUNCTION_REGISTRY or name in PATTERN_FUNCTIONS


def test_compiles_extended_indicator_expression():
    expr = (
        "MACD_HIST(12,26,9) < 0 AND KDJ_K(9,3,3) > KDJ_D(9,3,3) "
        "AND ADX(14) > 20 AND OBV() > 0"
    )
    compiler = ExpressionCompiler()
    assert compiler.validate(expr)[0]
    mask = compiler.compile_and_eval(expr, _df())
    assert len(mask) == 300
    assert mask.dtype == bool


def test_indicator_manifest_counts():
    counts = indicator_counts()
    try:
        import talib  # noqa: F401
    except ImportError:
        return
    assert counts.get("TA-Lib形态", 0) == 61
    assert counts.get("TA-Lib指标", 0) >= 55
