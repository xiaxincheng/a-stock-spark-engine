import pandas as pd

from backtest_framework import calc_trade_stats, load_stock_data, simulate_short_trades, simulate_trades_v2


def _frame():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "open": [100, 101, 110, 111, 112],
        "high": [101, 102, 116, 117, 118],
        "low": [99, 100, 109, 110, 111],
        "close": [100, 101, 112, 113, 114],
        "volume": [1000, 1100, 1200, 1300, 1400],
    })


def test_long_entry_uses_next_open_and_later_exit():
    frames = {"sh600001": _frame()}
    trades = simulate_trades_v2(
        [{"code": "sh600001", "signal_date": "2024-01-02", "signal_type": "test"}],
        frames,
        hold_days=3,
        tp=0.05,
        sl=-0.10,
    )
    assert len(trades) == 1
    assert trades[0]["buy_date"] == "2024-01-03"
    assert trades[0]["buy_price"] == 110
    assert trades[0]["sell_reason"] == "tp"
    assert calc_trade_stats(trades)["count"] == 1


def test_short_gap_through_target_fills_at_open():
    frame = _frame()
    frame.loc[4, "open"] = 90
    frames = {"sz000001": frame}
    trades = simulate_short_trades(
        [{"code": "sz000001", "signal_date": "2024-01-02", "signal_type": "test"}],
        frames,
        hold_days=3,
        tp=0.15,
        sl=-0.10,
    )
    assert len(trades) == 1
    assert trades[0]["buy_date"] == "2024-01-03"
    assert trades[0]["sell_reason"] == "gap_tp"


def test_csv_loader_requires_ohlcv_columns(tmp_path):
    path = tmp_path / "sh600001.csv"
    frame = pd.concat([_frame()] * 5, ignore_index=True)
    frame["date"] = pd.date_range("2024-01-01", periods=len(frame)).strftime("%Y-%m-%d")
    frame.to_csv(path, index=False)
    loaded = load_stock_data(path)
    assert loaded is not None
    assert loaded["date"].tolist()[0] == "2024-01-01"
