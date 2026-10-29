# Examples

This directory intentionally contains no real market data.

1. Point `SPARK_DATA_DIR` to your own OHLCV CSV directory.
2. Generate expressions with `python ../spark_engine.py generate --count 5`.
3. Verify one expression with `python ../spark_engine.py verify '<expr>' --direction long`.
