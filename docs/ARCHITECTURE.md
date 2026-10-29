# Architecture

```text
Indicator registry
  -> expression compiler
  -> L1/L2/L3/L4 candidate generator
  -> canonical condition_expr
  -> full-market signal scan
  -> annual signal gates
  -> next-open long/short research simulator
  -> robustness and family deduplication
  -> optional experiment ledger
```

## Key Boundaries

- `spark_core/indicator_registry.py`: executable indicators and patterns.
- `spark_core/spark_l1_contrastive.py`: safe expression parser/compiler.
- `spark_core/unified_engine.py`: canonical expression semantics and verifier.
- `spark_core/layered_generator.py`: public L1-L4 generation operators.
- `spark_core/strategy_family.py`: structural families and signal overlap.
- `backtest_framework.py`: data loading and research simulation only.
- `spark_ledger.py`: optional reproducibility database.

The public package has no web server, daemon, watcher, launch agent, brokerage adapter, portfolio manager, or live order path.
