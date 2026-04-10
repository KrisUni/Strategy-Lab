from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest import Trade
from ui.tabs.backtest import _build_bh_comparison_window, _calculate_curve_stats


def _make_results(index):
    return SimpleNamespace(
        equity_curve=pd.Series([10_000.0, 10_000.0, 11_000.0], index=index),
        bars_per_year=252,
        trades=[
            Trade(
                entry_idx=1,
                entry_date=index[1],
                entry_price=100.0,
                direction='long',
            )
        ],
    )


def test_bh_comparison_full_window_uses_full_series():
    index = pd.date_range('2024-01-01', periods=3, freq='D')
    df = pd.DataFrame({'close': [100.0, 120.0, 130.0]}, index=index)
    comparison = _build_bh_comparison_window(_make_results(index), df, 'full_window')

    assert comparison is not None
    assert comparison['start_date'] == index[0]
    assert comparison['strategy_curve'].index[0] == index[0]
    assert comparison['benchmark_curve'].iloc[0] == pytest.approx(10_000.0)
    assert comparison['benchmark_curve'].iloc[-1] == pytest.approx(13_000.0)


def test_bh_comparison_since_first_trade_slices_both_curves():
    index = pd.date_range('2024-01-01', periods=3, freq='D')
    df = pd.DataFrame({'close': [100.0, 120.0, 130.0]}, index=index)
    comparison = _build_bh_comparison_window(_make_results(index), df, 'since_first_trade')

    assert comparison is not None
    assert comparison['start_date'] == index[1]
    assert list(comparison['strategy_curve'].index) == list(index[1:])
    assert comparison['benchmark_curve'].iloc[0] == pytest.approx(comparison['strategy_curve'].iloc[0])

    strategy_stats = _calculate_curve_stats(comparison['strategy_curve'], 252)
    benchmark_stats = _calculate_curve_stats(comparison['benchmark_curve'], 252)

    assert strategy_stats['return_pct'] == pytest.approx(10.0)
    assert benchmark_stats['return_pct'] == pytest.approx((130.0 / 120.0 - 1.0) * 100.0)
