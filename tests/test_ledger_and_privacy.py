import re
from pathlib import Path

from spark_core.types import Direction, Hypothesis
from spark_ledger import ExperimentLedger


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_ledger_records_hypotheses_and_events(tmp_path):
    ledger = ExperimentLedger(str(tmp_path / "spark.sqlite"))
    hypothesis = Hypothesis(
        id="test-id", name="test", direction=Direction.LONG,
        condition_expr="RSI(14) < 30", generation_layer=1,
    )
    ledger.save_hypothesis(hypothesis)
    ledger.record_event(hypothesis.id, "generated", {"layer": 1})
    rows = ledger.hypotheses()
    assert len(rows) == 1
    assert rows[0]["condition_expr"] == "RSI(14) < 30"


def test_public_tree_has_no_private_local_or_live_trading_coupling():
    forbidden = [
        "/Users/", "WorkBuddy", "trading.db", "camp_",
        "trade_engine", "PortfolioManager", "LaunchAgents", "run_all_full",
    ]
    findings = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or path.name == Path(__file__).name
        ):
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".txt", ".json", ""}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                findings.append(f"{path.relative_to(ROOT)}: {token}")
    assert not findings, findings


def test_no_committed_credentials_or_real_market_data():
    assert not list((ROOT / "data").glob("**/*"))
    secret_patterns = [r"gh[pousr]_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{20,}", r"(?i)password\s*=\s*['\"][^'\"]+"]
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".txt", ".json", ""}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(re.search(pattern, text) for pattern in secret_patterns), path
