"""
Tests for same-bar entry+exit signal skip behavior.

When the exit signal is True on the same bar the entry signal fired,
the engine must skip the entry. Entering into an already-active exit
produces a guaranteed 1-bar round-trip loss (full costs, no upside).
"""
import pandas as pd
import pytest

from src.backtest import BacktestEngine
from src.strategy import StrategyParams


class _NoopSignalGen:
    """Stub that returns df unchanged, letting tests inject signal columns directly."""

    def generate_all_signals(self, df):
        return df


def _make_engine():
    params = StrategyParams(
        pamrp_enabled=False,
        bbwp_enabled=False,
        adx_enabled=False,
        ma_trend_enabled=False,
        rsi_enabled=False,
        volume_enabled=False,
        supertrend_enabled=False,
        vwap_enabled=False,
        macd_enabled=False,
        stoch_rsi_entry_enabled=False,
        pamrp_exit_enabled=False,
        stop_loss_enabled=False,
        take_profit_enabled=False,
        trailing_stop_enabled=False,
        atr_trailing_enabled=False,
        time_exit_enabled=False,
        ma_exit_enabled=False,
        bbwp_exit_enabled=False,
        stoch_rsi_exit_enabled=False,
        rsi_exit_enabled=False,
        allow_same_bar_exit=False,
        allow_same_bar_reversal=False,
    )
    engine = BacktestEngine(params, commission_pct=0.0, slippage_pct=0.0)
    engine.signal_gen = _NoopSignalGen()
    return engine


def _make_df(n, entry_long, exit_long_signal,
             entry_short=None, exit_short_signal=None, price=100.0):
    """Build a minimal OHLCV + signal DataFrame for engine tests."""
    idx = pd.date_range('2024-01-01', periods=n, freq='1D')
    df = pd.DataFrame({
        'open':   [price] * n,
        'high':   [price * 1.01] * n,
        'low':    [price * 0.99] * n,
        'close':  [price] * n,
        'volume': [1_000] * n,
    }, index=idx)
    df['entry_long']        = entry_long
    df['entry_short']       = entry_short        if entry_short        is not None else [False] * n
    df['exit_long_signal']  = exit_long_signal
    df['exit_short_signal'] = exit_short_signal  if exit_short_signal  is not None else [False] * n
    return df


class TestSkipEntryWhenExitActive:
    def test_skip_long_entry_when_exit_signal_active_same_bar(self):
        """When entry_long[i] and exit_long_signal[i] are both True on the same bar,
        no trade is opened — the entry must be skipped."""
        df = _make_df(
            n=3,
            entry_long=[True, False, False],
            exit_long_signal=[True, False, False],
        )
        results = _make_engine().run(df)
        assert len(results.trades) == 0

    def test_skip_short_entry_when_exit_signal_active_same_bar(self):
        """Symmetric for shorts: short entry skipped when exit_short_signal is True
        on the same bar the entry signal fires."""
        df = _make_df(
            n=3,
            entry_long=[False, False, False],
            exit_long_signal=[False, False, False],
            entry_short=[True, False, False],
            exit_short_signal=[True, False, False],
        )
        results = _make_engine().run(df)
        assert len(results.trades) == 0

    def test_entry_proceeds_when_exit_signal_inactive_at_entry_bar(self):
        """Entry proceeds normally when exit_long_signal is False on the entry bar.
        Bar 0: entry fires, exit False → enter at bar 1's open.
        Bar 2: exit fires → close at bar 3's open."""
        df = _make_df(
            n=4,
            entry_long=[True, False, False, False],
            exit_long_signal=[False, False, True, False],
        )
        results = _make_engine().run(df)
        assert len(results.trades) == 1

    def test_entry_proceeds_on_next_bar_when_exit_clears(self):
        """Bar 0: entry+exit both True → skip.
        Bar 1: entry still True, exit now False → enter at bar 2's open.
        Bar 3: exit fires → close at bar 4's open."""
        df = _make_df(
            n=5,
            entry_long=[True, True, False, False, False],
            exit_long_signal=[True, False, False, True, False],
        )
        results = _make_engine().run(df)
        assert len(results.trades) == 1
        assert results.trades[0].entry_idx == 2

    def test_one_bar_round_trip_does_not_occur(self):
        """Regression: before the fix, entry_long[0]=True + exit_long_signal[0]=True
        produced a trade entered at bar 1 and exited at bar 2 (1-bar round-trip).
        With the fix, no trade is produced."""
        df = _make_df(
            n=4,
            entry_long=[True, False, False, False],
            exit_long_signal=[True, True, False, False],
        )
        results = _make_engine().run(df)
        assert len(results.trades) == 0

    def test_metrics_not_polluted_by_skipped_entry(self):
        """A skipped entry must not appear as a trade or cost in the results.
        num_trades=0 and equity is flat (no costs applied)."""
        df = _make_df(
            n=3,
            entry_long=[True, False, False],
            exit_long_signal=[True, False, False],
        )
        results = _make_engine().run(df)
        assert results.num_trades == 0
        assert results.total_return_pct == pytest.approx(0.0, abs=1e-9)
