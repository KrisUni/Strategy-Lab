"""
Compute and commit expected trade-log hashes for the regression harness.

Run once after generating fixtures to produce expected_hashes.json:
    python -m tests.regression.generate_hashes

Re-run only when a phase intentionally changes backtest behavior
(which should never happen — every phase must be behavior-preserving).
"""
import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.strategy import StrategyParams
from src.backtest import BacktestEngine
from tests.regression.configs import CONFIGS

FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXPECTED_PATH = Path(__file__).parent / "expected_hashes.json"


def compute_trade_hash(trades) -> str:
    if not trades:
        return hashlib.sha256(b"").hexdigest()
    rows = []
    for t in trades:
        rows.append({
            "entry_time":  t.entry_date,
            "exit_time":   t.exit_date,
            "direction":   t.direction,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "size":        t.size_dollars,
            "pnl":         t.pnl,
            "exit_reason": t.exit_reason,
        })
    trades_df = pd.DataFrame(rows)[
        ["entry_time", "exit_time", "direction", "entry_price",
         "exit_price", "size", "pnl", "exit_reason"]
    ]
    return hashlib.sha256(
        pd.util.hash_pandas_object(trades_df, index=False).values.tobytes()
    ).hexdigest()


def generate_hashes() -> None:
    hashes = {}
    print("Computing trade-log hashes...\n")

    for config_name, config_dict, fixture_file in CONFIGS:
        fixture_path = FIXTURE_DIR / fixture_file
        if not fixture_path.exists():
            print(f"  MISSING fixture: {fixture_file}  — run generate_fixtures.py first")
            sys.exit(1)

        df = pd.read_parquet(fixture_path)
        params = StrategyParams.from_dict(config_dict)
        engine = BacktestEngine(params)
        results = engine.run(df)
        n = len(results.trades)
        h = compute_trade_hash(results.trades)
        hashes[config_name] = h
        status = "OK" if n >= 5 else f"WARN: only {n} trades — hash may be fragile"
        print(f"  {config_name}: {n} trades  [{status}]")
        print(f"    hash: {h}")

    EXPECTED_PATH.write_text(json.dumps(hashes, indent=2))
    print(f"\nWrote {EXPECTED_PATH}")


if __name__ == "__main__":
    generate_hashes()
