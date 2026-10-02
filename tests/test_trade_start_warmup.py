"""
trade_start gives backtests indicator warmup history.

Verifies that:
1. trade_start=None and trade_start=first bar are identical
2. trade_start slices simulation, equity and trades
3. trade_start after the last bar raises
4. Walk-forward fold backtests receive full history up to window end
5. The simple-split test run receives full history, trading from the test start
"""
import pandas as pd
import pytest

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

from src.backtest import BacktestEngine
from src.data import generate_sample_data
from src.indicators.registry import INDICATOR_REGISTRY
from src.optimize import BayesianOptimizer
from src.strategy import StrategyParams


@pytest.fixture(scope="module")
def df():
    return generate_sample_data(days=800, seed=3)


@pytest.fixture(scope="module")
def filters():
    """PAMRP entry + PAMRP exit on; every other indicator off."""
    f = {spec.enable_param: False for spec in INDICATOR_REGISTRY}
    f["pamrp_enabled"] = True
    f["pamrp_exit_enabled"] = True
    return f


@pytest.fixture
def recorder(monkeypatch):
    """Record (len(df), df.index[0], trade_start) for every BacktestEngine.run call."""
    calls = []
    original = BacktestEngine.run

    def rec(self, df, trade_start=None):
        calls.append((len(df), df.index[0], trade_start))
        return original(self, df, trade_start=trade_start)

    monkeypatch.setattr(BacktestEngine, "run", rec)
    return calls


def _trade_tuples(res):
    return [
        (t.entry_date, t.exit_date, t.direction, t.entry_price, t.exit_price,
         t.size_dollars, t.pnl, t.exit_reason)
        for t in res.trades
    ]


def test_trade_start_none_and_first_bar_are_identical(df):
    a = BacktestEngine(StrategyParams()).run(df)
    b = BacktestEngine(StrategyParams()).run(df, trade_start=df.index[0])
    assert _trade_tuples(a) == _trade_tuples(b)
    pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)


def test_trade_start_slices_simulation(df):
    ts = df.index[300]
    r = BacktestEngine(StrategyParams()).run(df, trade_start=ts)
    assert r.equity_curve.index[0] == ts
    assert len(r.equity_curve) == 500
    assert r.equity_curve.iloc[0] == r.initial_capital
    assert all(t.entry_date > ts for t in r.trades)


def test_trade_start_after_end_raises(df):
    with pytest.raises(ValueError):
        BacktestEngine(StrategyParams()).run(
            df, trade_start=df.index[-1] + pd.Timedelta(days=1)
        )


def test_walkforward_backtests_receive_warmup_history(df, filters, recorder, monkeypatch):
    monkeypatch.setattr(
        BayesianOptimizer, "_optimize_on_data",
        lambda self, *a, **k: (StrategyParams(), 0.0),
    )
    opt = BayesianOptimizer(df, filters, use_walkforward=True, n_folds=4, min_trades=0)
    opt.optimize(n_trials=5, show_progress=False)

    windows = opt._get_fold_windows()
    assert windows
    for tr_start, tr_end, ts_start, ts_end in windows:
        assert (tr_end, df.index[0], df.index[tr_start]) in recorder
        assert (ts_end, df.index[0], df.index[ts_start]) in recorder


def test_simple_split_test_run_receives_warmup_history(df, filters, recorder):
    opt = BayesianOptimizer(df, filters, use_walkforward=False, min_trades=0)
    opt.optimize(n_trials=2, show_progress=False)

    assert (len(df), df.index[0], opt.test_df.index[0]) in recorder
    assert all(c[1] != opt.test_df.index[0] for c in recorder)
