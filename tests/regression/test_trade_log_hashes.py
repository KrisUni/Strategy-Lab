"""
Regression harness: pins backtest behavior via trade-log SHA-256 hashes.

Any phase of the indicator-registry refactor that changes these hashes
has introduced a behavioral regression and must not be merged.

Expected hashes live in expected_hashes.json (committed).
Fixtures are committed parquet files under fixtures/.

To regenerate (only when a phase intentionally changes behavior):
    python -m tests.regression.generate_fixtures   # if fixtures need refresh
    python -m tests.regression.generate_hashes     # recompute hashes
"""
import hashlib
import json
import pandas as pd
import pytest
from pathlib import Path

from src.strategy import StrategyParams
from src.backtest import BacktestEngine
from tests.regression.configs import CONFIGS

FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXPECTED_HASHES_PATH = Path(__file__).parent / "expected_hashes.json"


def _compute_trade_hash(trades) -> str:
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


@pytest.fixture(scope="session")
def expected_hashes() -> dict:
    assert EXPECTED_HASHES_PATH.exists(), (
        f"expected_hashes.json not found at {EXPECTED_HASHES_PATH}.\n"
        "Run: python -m tests.regression.generate_hashes"
    )
    return json.loads(EXPECTED_HASHES_PATH.read_text())


@pytest.mark.parametrize(
    "config_name,config_dict,fixture_file",
    CONFIGS,
    ids=[c[0] for c in CONFIGS],
)
def test_trade_log_hash(config_name, config_dict, fixture_file, expected_hashes):
    fixture_path = FIXTURE_DIR / fixture_file
    assert fixture_path.exists(), (
        f"Fixture missing: {fixture_path}\n"
        "Run: python -m tests.regression.generate_fixtures"
    )

    df = pd.read_parquet(fixture_path)
    params = StrategyParams.from_dict(config_dict)
    engine = BacktestEngine(params)
    results = engine.run(df)

    n = len(results.trades)
    assert n >= 5, (
        f"{config_name}: only {n} trades — hash is meaningless. "
        "Config may be too restrictive or fixture too short."
    )

    actual_hash = _compute_trade_hash(results.trades)
    expected_hash = expected_hashes[config_name]

    assert actual_hash == expected_hash, (
        f"{config_name}: trade log changed!\n"
        f"  expected: {expected_hash}\n"
        f"  actual:   {actual_hash}\n"
        f"  trades:   {n}\n"
        "If this phase intentionally changes behavior, re-run generate_hashes.py "
        "and justify the change in the PR description."
    )
