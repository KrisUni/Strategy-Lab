"""end_of_data force-closes stay in trades/equity but are excluded from trade statistics."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy import StrategyParams
from src.backtest import BacktestEngine


def _run(open_, close, entry_long, exit_long=None):
    """Run a stubbed-signal long-only backtest with zero costs and no stop-loss."""
    n = len(open_)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        'open': [float(x) for x in open_],
        'high': [max(o, c) + 1.0 for o, c in zip(open_, close)],
        'low': [min(o, c) - 1.0 for o, c in zip(open_, close)],
        'close': [float(x) for x in close],
        'volume': [1000] * n,
    }, index=idx)
    exit_long = exit_long if exit_long is not None else [False] * n

    def add_signals(frame):
        frame = frame.copy()
        frame['entry_long'] = entry_long
        frame['entry_short'] = [False] * n
        frame['exit_long_signal'] = exit_long
        frame['exit_short_signal'] = [False] * n
        return frame

    engine = BacktestEngine(
        StrategyParams(stop_loss_enabled=False),
        commission_pct=0.0,
        slippage_pct=0.0,
    )
    engine.signal_gen.generate_all_signals = add_signals
    return engine.run(df)


def test_only_end_of_data_trade():
    T, F = True, False
    r = _run(
        open_=[100, 100, 102, 104, 106],
        close=[100, 101, 103, 105, 107],
        entry_long=[T, F, F, F, F],
    )
    assert len(r.trades) == 1
    assert r.trades[0].exit_reason == 'end_of_data'
    assert r.num_trades == 0
    assert r.open_trades_at_end == 1
    assert r.win_rate == 0
    assert r.profit_factor == 0
    assert r.total_return_pct == pytest.approx(7.0)


def test_end_of_data_loser_excluded_from_stats():
    T, F = True, False
    r = _run(
        open_=[100, 100, 101, 102, 103, 104, 103, 102],
        close=[100.5, 100.5, 101.5, 102.5, 103.5, 103.5, 102.5, 100.0],
        entry_long=[T, F, F, F, T, F, F, F],
        exit_long=[F, F, T, F, F, F, F, F],
    )
    assert len(r.trades) == 2
    assert [t.exit_reason for t in r.trades] == ['signal', 'end_of_data']
    assert r.num_trades == 1
    assert r.open_trades_at_end == 1
    assert r.winners == 1
    assert r.losers == 0
    assert r.win_rate == 100.0
    assert r.profit_factor == 999.99
    assert r.total_return_pct < 2.0
