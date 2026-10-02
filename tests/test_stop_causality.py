"""
Trailing and ATR-trailing stop levels must be computable at the bar's open.

A resting stop order for bar i can only use extremes through bar i-1 and
ATR[i-1]. Changing only the exit bar's high (trailing %) or low (ATR trailing)
must therefore not change the fill price of the exit.
"""
import pytest

from src.backtest import BacktestEngine
from src.data import generate_sample_data
from src.strategy import StrategyParams


@pytest.fixture(scope="module")
def df():
    return generate_sample_data(days=1500, seed=3)


def _checkable_exits(params, df, reason, limit=5):
    """Trades exiting with `reason` at a price other than the exit bar's open."""
    r = BacktestEngine(params).run(df)
    out = [t for t in r.trades
           if t.exit_reason == reason and abs(t.exit_price - df["open"].iloc[t.exit_idx]) > 1e-9]
    return out[:limit]


def _assert_exit_unchanged(params, df, reason, column, factor):
    """Perturb only `column` on each exit bar by `factor` and assert the fill is unchanged."""
    exits = _checkable_exits(params, df, reason)
    assert len(exits) >= 3
    for t in exits:
        perturbed = df.copy()
        perturbed.iloc[t.exit_idx, perturbed.columns.get_loc(column)] *= factor
        r2 = BacktestEngine(params).run(perturbed)
        match = [x for x in r2.trades if x.entry_idx == t.entry_idx]
        assert len(match) == 1
        assert match[0].exit_idx == t.exit_idx
        assert match[0].exit_reason == reason
        assert match[0].exit_price == pytest.approx(t.exit_price, abs=1e-12)


def test_atr_stop_level_ignores_exit_bar_range(df):
    p = StrategyParams(atr_trailing_enabled=True, atr_length=14, atr_multiplier=2.0,
                       pamrp_exit_enabled=False, stop_loss_enabled=False)
    _assert_exit_unchanged(p, df, "atr_trailing", "low", 0.97)


def test_trailing_stop_level_ignores_exit_bar_high(df):
    p = StrategyParams(trailing_stop_enabled=True, trailing_stop_pct=3.0,
                       trailing_stop_activation=1.0,
                       pamrp_exit_enabled=False, stop_loss_enabled=False)
    _assert_exit_unchanged(p, df, "trailing_stop", "high", 1.03)
