"""Optional SQLite experiment ledger for reproducible research."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict

from spark_core.types import Hypothesis


class ExperimentLedger:
    def __init__(self, path: str):
        self.path = path
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS hypotheses (
                    id TEXT PRIMARY KEY, name TEXT, direction TEXT,
                    generation_layer INTEGER, condition_expr TEXT,
                    description TEXT, status TEXT, result_json TEXT,
                    created_at TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS experiment_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id TEXT, event_type TEXT, detail_json TEXT,
                    created_at TEXT
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def save_hypothesis(self, hypothesis: Hypothesis, status: str = "proposed", result: dict | None = None) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO hypotheses
                   (id,name,direction,generation_layer,condition_expr,description,status,result_json,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    hypothesis.id, hypothesis.name, hypothesis.direction.value,
                    hypothesis.generation_layer, hypothesis.condition_expr,
                    hypothesis.description, status, json.dumps(result or {}, ensure_ascii=False), now,
                ),
            )

    def record_event(self, hypothesis_id: str, event_type: str, detail: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiment_events(hypothesis_id,event_type,detail_json,created_at) VALUES (?,?,?,?)",
                (hypothesis_id, event_type, json.dumps(detail or {}, ensure_ascii=False), time.strftime("%Y-%m-%d %H:%M:%S")),
            )

    def hypotheses(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM hypotheses ORDER BY created_at,id")]

    def unused_natural_law_ideas(self, limit: int = 3) -> list[dict]:
        """Compatibility hook for optional externally generated idea pools."""
        return []
