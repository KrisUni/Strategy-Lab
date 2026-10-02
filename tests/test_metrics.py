"""
Return-based metrics: shared pure functions and engine integration.

Verifies that:
1. Sortino uses downside deviation over all observations
2. Calmar sign follows CAGR
3. CAGR uses n-1 periods and reports ruin as -100%
4. The entry bar counts as in-market for Sharpe and exposure
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestEngine
from src.backtest.metrics import (
    METRIC_CAP,
    annualized_sharpe,
    annualized_sortino,
    cagr_pct,
    calmar_ratio,
    downside_deviation,
)
from src.data import generate_sample_data
from src.strategy import StrategyParams


def test_downside_deviation():
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert downside_deviation(r) == pytest.approx(0.011180339887498949)
    assert downside_deviation(pd.Series([], dtype=float)) == 0.0


def test_sortino():
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert annualized_sortino(r, 252) == pytest.approx(3.549647869859769)
    assert annualized_sortino(pd.Series([0.01, 0.02]), 252) == METRIC_CAP
    assert annualized_sortino(pd.Series([0.0, 0.0]), 252) == 0.0
    assert annualized_sortino(pd.Series([0.01]), 252) == 0.0


def test_sharpe_zero_std():
    assert annualized_sharpe(pd.Series([0.01, 0.01, 0.01]), 252) == 0.0
    assert annualized_sharpe(pd.Series([0.01]), 252) == 0.0


def test_calmar():
    assert calmar_ratio(-26.49, -85.95) == pytest.approx(-0.3082024432809773)
    assert calmar_ratio(10.0, 0.0) == METRIC_CAP
    assert calmar_ratio(-5.0, 0.0) == 0.0


def test_cagr():
    assert cagr_pct(10_000, 12_000, 253, 252) == pytest.approx(20.0)
    assert cagr_pct(10_000, 0.0, 100, 252) == -100.0
    assert cagr_pct(10_000, 12_000, 1, 252) == 0.0


def test_sharpe_mask_includes_entry_bar():
    df = generate_sample_data(days=1500, seed=3)
    r = BacktestEngine(StrategyParams()).run(df)
    assert len(r.trades) > 0
    mask = np.zeros(len(df), dtype=bool)
    for t in r.trades:
        mask[t.entry_idx : t.exit_idx + 1] = True
    rets = r.equity_curve.pct_change()
    valid = rets.notna().to_numpy()
    active = rets[mask & valid]
    n_total = int(valid.sum())
    expected = active.mean() / active.std() * np.sqrt(len(active) * r.bars_per_year / n_total)
    assert r.sharpe_ratio == pytest.approx(expected, rel=1e-12)
    assert r.pct_time_in_market == pytest.approx(mask.sum() / len(df) * 100)
