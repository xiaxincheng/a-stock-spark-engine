"""Small singleton cache so one engine run loads each CSV once."""
from __future__ import annotations

from pathlib import Path

from backtest_framework import build_stock_df_map, scan_all_csvs


class MarketDataLoader:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self._frames: dict | None = None

    @classmethod
    def get_instance(cls) -> "MarketDataLoader":
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
        return _INSTANCE

    def ensure_loaded(self, start_date=None, end_date=None, verbose: bool = False) -> None:
        files = scan_all_csvs(self.data_dir)
        if verbose:
            print(f"[data] loading {len(files)} CSV files")
        self._frames = build_stock_df_map(files, start_date=start_date, end_date=end_date)
        if verbose:
            print(f"[data] loaded {len(self._frames)} valid series")

    def is_loaded(self) -> bool:
        return self._frames is not None

    def get_stock_df_map(self) -> dict:
        return self._frames or {}

    def get_df(self, code: str):
        return (self._frames or {}).get(code)


_INSTANCE: MarketDataLoader | None = None
