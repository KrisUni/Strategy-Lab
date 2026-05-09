"""
Generate (or regenerate) regression fixture parquet files.

Run once before the first regression test run:
    python -m tests.regression.generate_fixtures

Uses yfinance for daily data. Config 3 (ETH-USD 1h) uses synthetic data
because yfinance 1h history is limited to ~730 rolling calendar days
and would produce a non-reproducible fixture.
"""
import sys
from pathlib import Path

# Allow running from project root: python -m tests.regression.generate_fixtures
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.data import fetch_yfinance, generate_sample_data

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(exist_ok=True)


def _save(df: pd.DataFrame, filename: str) -> None:
    path = FIXTURE_DIR / filename
    df.to_parquet(path)
    print(f"  saved {filename}  ({len(df)} bars)")


def generate_all() -> None:
    print("Generating regression fixtures...\n")

    # ── Fixture 1: SPY daily 2020-2024 ────────────────────────────────────────
    print("1. SPY 1d 2020-01-01 → 2024-12-31")
    df = fetch_yfinance("SPY", "2020-01-01", "2024-12-31", "1d")
    _save(df, "spy_1d_2020_2024.parquet")

    # ── Fixture 2: QQQ daily 2018-2024 ────────────────────────────────────────
    print("2. QQQ 1d 2018-01-01 → 2024-12-31")
    df = fetch_yfinance("QQQ", "2018-01-01", "2024-12-31", "1d")
    _save(df, "qqq_1d_2018_2024.parquet")

    # ── Fixture 3: Synthetic ETH-1h-style data (see configs.py for why) ───────
    print("3. ETH-1h synthetic (seed=7, vol=0.025, drift=0.0002)")
    df = generate_sample_data(
        days=2000,
        volatility=0.025,  # ~2.5% daily vol ≈ crypto-like
        drift=0.0002,      # slight positive drift (avoids permanent open positions)
        seed=7,
        start_date="2018-01-01",
    )
    _save(df, "eth_1h_synthetic.parquet")

    # ── Fixture 4: AAPL daily 2015-2024 ───────────────────────────────────────
    print("4. AAPL 1d 2015-01-01 → 2024-12-31")
    df = fetch_yfinance("AAPL", "2015-01-01", "2024-12-31", "1d")
    _save(df, "aapl_1d_2015_2024.parquet")

    # ── Fixture 5: TSLA daily 2020-2024 ───────────────────────────────────────
    print("5. TSLA 1d 2020-01-01 → 2024-12-31")
    df = fetch_yfinance("TSLA", "2020-01-01", "2024-12-31", "1d")
    _save(df, "tsla_1d_2020_2024.parquet")

    print("\nAll fixtures generated.")


if __name__ == "__main__":
    generate_all()
