# A-Stock Spark Engine

A-share indicator strategy research engine for generating executable conditions, evaluating full-market signals, and running reproducible long/short research backtests.

This project is research software only. It does **not** connect to a broker, place orders, manage a portfolio, or provide investment advice.

## Why This Exists

Random strategy ideas are cheap; reproducible evidence is not. Spark Engine keeps generation, expression evaluation, and backtesting in one deterministic pipeline:

- Every idea has a text explanation and an executable `condition_expr`.
- The same indicator implementation is used for generation and verification.
- Signals are deduplicated by stock and date.
- Research backtests enter at the next open to avoid same-bar lookahead.
- Annual signal-count and robustness gates reject sparse, over-fitted candidates.
- Related candidates can be grouped by structural family and signal overlap.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+ is required.

### Optional TA-Lib

The baseline registry uses pandas/numpy and works without TA-Lib. If TA-Lib is installed, Spark Engine automatically expands the catalog with additional TA-Lib functions and candlestick patterns.

## Data Format

Set `SPARK_DATA_DIR` to a directory of daily CSV files. A common layout is:

```text
data/daily/
  sh/sh600000.csv
  sz/sz000001.csv
```

A flat directory also works. Each CSV needs these columns:

```text
date,open,high,low,close,volume
```

Dates should use `YYYY-MM-DD`. The repository deliberately ships no real market data. For accurate research, use a survivorship-bias-aware, adjusted OHLCV universe that includes delisted stocks where possible.

For a few generic TA-Lib statistics without a natural fixed threshold, L1 initialization can use `SPARK_REFERENCE_CSV` to measure a representative 30th/70th percentile on one long, adjusted series. This only initializes sensible thresholds; verification still runs across the complete data directory.

If raw files contain large ex-rights gaps and you do not have an official adjustment factor, you can enable the built-in heuristic with `SPARK_FORWARD_ADJUST=1`. An exchange-provided adjusted series is preferred.

## Quick Start

Show the executable indicator catalog:

```bash
python spark_engine.py catalog
```

Generate initial hypotheses:

```bash
python spark_engine.py generate --count 5 --direction both --seed 42
```

Backtest one expression:

```bash
python spark_engine.py backtest \
  'RSI(14) < 35 AND close > MA(60)' \
  --direction long \
  --data-dir /path/to/csv \
  --start 2020-01-01 \
  --end 2024-12-31
```

Run annual signal and robustness gates:

```bash
python spark_engine.py verify \
  'KDJ_D(9,3,3) > 70 AND close < ENE_LOWER(10,11,9)' \
  --direction short \
  --data-dir /path/to/csv
```

## Layers

### L1: New Structure

Generates simple 1-3 condition expressions from the indicator registry. It prefers at least one continuous indicator and keeps initial search broad but constrained.

### L2: Parameter Neighborhood

For a surviving parent, creates bounded threshold/period neighborhoods. A parameter island is not treated as evidence; neighboring variants should remain economically sensible.

### L3: Survivor Evolution

Mutates verified parents with one structural step at a time:

- threshold perturbation
- add one filter
- remove one redundant filter
- replace one same-role indicator

L3 does not silently fall back to blind random generation when no parent survives.

### L4: Same-Direction Fusion

Combines two surviving parents only when their directions match. The expression remains executable and the verifier resolves entry and exit semantics explicitly.

## Verification

The unified verifier:

1. Parses the canonical boolean expression.
2. Evaluates it across every CSV in `SPARK_DATA_DIR`.
3. Deduplicates signals by `(code, signal_date)`.
4. Rejects fewer than 300 signals per development year by default.
5. Rejects excessively broad signal sets.
6. Simulates next-open entries.
7. Measures PF, trade count, win rate, yearly stability, and annual robustness.

Default development dates are currently `2020-01-01` through `2024-12-31`. Change source constants only with a documented reason; changing windows repeatedly without recording experiments introduces data snooping.

## Backtest Assumptions

Research defaults:

- Long entry: next trading day open.
- Short entry: next trading day open.
- Long stop, target, and timeout exits use later bars.
- If stop and target hit the same bar, stop is evaluated first.
- Short gaps through a level fill at the open.
- Default round-trip research cost: buy 0.07%, sell 0.17%.
- A-share opening limit-up entries are skipped for longs.
- Borrow cost, borrow availability, partial fills, market impact, and live slippage are not modeled.

Short results are statistical candidates only and are not executable A-share strategies by themselves.

## Experiment Ledger

Set `SPARK_DB_PATH` to a SQLite path to record generated hypotheses and lifecycle events. The public ledger keeps an auditable trial count without bundling any private historical results.

## Limitations

- Results can contain survivorship bias if delisted stocks are absent.
- High PF after many trials is not independent evidence.
- Multiple-testing correction is not automatically applied.
- Heuristic forward adjustment is not equivalent to official adjustment factors.
- Past performance does not predict future returns.
- This is not investment advice or executable trading software.

## License

MIT License. See [LICENSE](LICENSE).
