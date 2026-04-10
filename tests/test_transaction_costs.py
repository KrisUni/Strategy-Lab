import inspect

import pandas as pd
import pytest

from src.backtest import (
    BacktestEngine,
    BacktestResults,
    DEFAULT_COMMISSION_PCT,
    DEFAULT_SLIPPAGE_PCT,
)
from src.optimize import BayesianOptimizer, optimize_strategy
from src.permutation import run_permutation_test
from src.strategy import StrategyParams


def test_transaction_cost_defaults_are_shared():
    backtest_sig = inspect.signature(BacktestEngine.__init__)
    optimize_sig = inspect.signature(optimize_strategy)
    permutation_sig = inspect.signature(run_permutation_test)

    assert backtest_sig.parameters['commission_pct'].default == DEFAULT_COMMISSION_PCT
    assert backtest_sig.parameters['slippage_pct'].default == DEFAULT_SLIPPAGE_PCT
    assert optimize_sig.parameters['commission_pct'].default == DEFAULT_COMMISSION_PCT
    assert optimize_sig.parameters['slippage_pct'].default == DEFAULT_SLIPPAGE_PCT
    assert permutation_sig.parameters['commission_pct'].default == DEFAULT_COMMISSION_PCT
    assert permutation_sig.parameters['slippage_pct'].default == DEFAULT_SLIPPAGE_PCT


def test_backtest_results_record_transaction_cost_settings():
    idx = pd.date_range('2024-01-01', periods=2, freq='D')
    equity = pd.Series([10_000.0, 10_000.0], index=idx)
    results = BacktestResults(
        trades=[],
        equity_curve=equity,
        realized_equity=equity,
        commission_pct=0.25,
        slippage_pct=0.40,
    )

    assert results.commission_pct == pytest.approx(0.25)
    assert results.slippage_pct == pytest.approx(0.40)


def test_optimizer_passes_slippage_to_backtest_engine(monkeypatch):
    captured = {}

    class DummyEngine:
        def __init__(self, params, initial_capital, commission_pct, slippage_pct):
            captured['initial_capital'] = initial_capital
            captured['commission_pct'] = commission_pct
            captured['slippage_pct'] = slippage_pct

        def run(self, df):
            return BacktestResults(
                trades=[],
                equity_curve=pd.Series(dtype=float),
                realized_equity=pd.Series(dtype=float),
                initial_capital=captured['initial_capital'],
                commission_pct=captured['commission_pct'],
                slippage_pct=captured['slippage_pct'],
            )

    monkeypatch.setattr('src.optimize.BacktestEngine', DummyEngine)

    df = pd.DataFrame(
        {
            'open': [100.0, 101.0],
            'high': [101.0, 102.0],
            'low': [99.0, 100.0],
            'close': [100.5, 101.5],
            'volume': [1_000, 1_100],
        },
        index=pd.date_range('2024-01-01', periods=2, freq='D'),
    )

    opt = BayesianOptimizer(
        df=df,
        enabled_filters={},
        initial_capital=25_000,
        commission_pct=0.33,
        slippage_pct=0.44,
    )
    result = opt._run_backtest(StrategyParams(), df)

    assert captured == {
        'initial_capital': 25_000,
        'commission_pct': 0.33,
        'slippage_pct': 0.44,
    }
    assert result.commission_pct == pytest.approx(0.33)
    assert result.slippage_pct == pytest.approx(0.44)
