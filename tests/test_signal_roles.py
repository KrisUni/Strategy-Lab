"""
Tests for trigger-vs-filter signal role semantics.

Trigger specs enter only on the False→True edge of their long/short condition.
Filter specs gate entry but never cause entry on their own.
"""
import warnings

import pandas as pd
import pytest

from src.strategy import SignalGenerator, StrategyParams, ConditionOperator, TradeDirection


def _minimal_params(**overrides):
    """StrategyParams with all indicators off except what the caller enables."""
    defaults = dict(
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
        # exits off too
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
    )
    defaults.update(overrides)
    return StrategyParams(**defaults)


class TestTriggerEdgeDetection:
    def test_trigger_signal_requires_transition(self):
        """Trigger enters only on the False→True crossing bar.
        RSI stays below oversold for bars 1–5, crosses back above, then dips again.
        Only the two crossing bars (1 and 8) should produce entry_long=True."""
        rsi_vals = [50, 20, 20, 20, 20, 20, 50, 50, 20, 20]
        df = pd.DataFrame({'rsi': rsi_vals}, index=pd.RangeIndex(len(rsi_vals)))
        params = _minimal_params(
            rsi_enabled=True,
            rsi_oversold=30,
            rsi_overbought=70,
            trade_direction=TradeDirection.LONG_ONLY,
        )

        result = SignalGenerator(params).generate_entry_signals(df)

        # Only bars 1 and 8 are crossings (50→20 and 50→20)
        expected = [False, True, False, False, False, False, False, False, True, False]
        assert result['entry_long'].tolist() == expected

    def test_trigger_no_reentry_while_condition_persists(self):
        """After a trigger fires, no re-entry while same condition stays True."""
        rsi_vals = [50, 20, 20, 20]
        df = pd.DataFrame({'rsi': rsi_vals}, index=pd.RangeIndex(4))
        params = _minimal_params(
            rsi_enabled=True,
            rsi_oversold=30,
            rsi_overbought=70,
            trade_direction=TradeDirection.LONG_ONLY,
        )

        result = SignalGenerator(params).generate_entry_signals(df)

        # Bar 1 is the only crossing; bars 2 and 3 are suppressed
        assert result['entry_long'].tolist() == [False, True, False, False]


class TestFilterNeverEntersAlone:
    def test_filter_signal_does_not_alone_enter(self):
        """Filter-only config produces zero entries even when the filter is True."""
        df = pd.DataFrame({
            'ma_fast': [105.0, 110.0, 115.0],
            'ma_slow': [100.0, 100.0, 100.0],
        }, index=pd.RangeIndex(3))
        params = _minimal_params(
            ma_trend_enabled=True,
            ma_trend_signal_mode="filter",
            trade_direction=TradeDirection.LONG_ONLY,
        )

        result = SignalGenerator(params).generate_entry_signals(df)

        assert result['entry_long'].tolist() == [False, False, False]

    def test_no_triggers_warns(self):
        """All-filter config emits a UserWarning pointing to missing triggers."""
        df = pd.DataFrame({
            'ma_fast': [105.0, 110.0, 115.0],
            'ma_slow': [100.0, 100.0, 100.0],
        }, index=pd.RangeIndex(3))
        params = _minimal_params(
            ma_trend_enabled=True,
            ma_trend_signal_mode="filter",
            trade_direction=TradeDirection.LONG_ONLY,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SignalGenerator(params).generate_entry_signals(df)

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "trigger" in str(w[0].message).lower()


class TestFilterPlusTriggerComposition:
    def test_trigger_gated_by_filter(self):
        """RSI (trigger) + MA Trend (filter): entry only when trigger fires AND filter passes."""
        # RSI crosses oversold at bars 1 and 3.
        # MA Trend is True at bar 1 (fast > slow) and False at bar 3 (fast < slow).
        df = pd.DataFrame({
            'rsi':     [50.0, 20.0, 50.0, 20.0, 20.0],
            'ma_fast': [105.0, 110.0, 110.0, 90.0, 88.0],
            'ma_slow': [100.0, 100.0, 100.0, 100.0, 100.0],
        }, index=pd.RangeIndex(5))
        params = _minimal_params(
            rsi_enabled=True,
            rsi_oversold=30,
            rsi_overbought=70,
            ma_trend_enabled=True,
            ma_trend_signal_mode="filter",
            entry_operator=ConditionOperator.AND,
            trade_direction=TradeDirection.LONG_ONLY,
        )

        result = SignalGenerator(params).generate_entry_signals(df)

        # Bar 1: RSI trigger fires + MA filter True  → enter
        # Bar 3: RSI trigger fires + MA filter False → blocked
        assert result['entry_long'].tolist() == [False, True, False, False, False]

    def test_filter_and_operator_applied_between_multiple_filters(self):
        """With two filters AND one trigger, AND entry_operator requires both filters True."""
        # PAMRP as trigger (signal_mode_param default = trigger)
        # MA Trend + BBWP as filters
        # trigger fires at bar 1 (pamrp crosses below long_threshold)
        # MA filter True at bar 1, BBWP filter True at bar 1 → AND passes → entry
        # trigger fires at bar 3, MA True but BBWP False → AND fails → no entry
        df = pd.DataFrame({
            'pamrp_entry': [50.0, 10.0, 50.0, 10.0, 10.0],
            'ma_fast':     [105.0, 110.0, 110.0, 110.0, 110.0],
            'ma_slow':     [100.0, 100.0, 100.0, 100.0, 100.0],
            'bbwp':        [30.0, 30.0, 70.0, 70.0, 70.0],
            'bbwp_sma':    [30.0, 30.0, 70.0, 70.0, 70.0],
        }, index=pd.RangeIndex(5))
        params = _minimal_params(
            pamrp_enabled=True,
            pamrp_entry_long=20,
            pamrp_entry_short=80,
            ma_trend_enabled=True,
            ma_trend_signal_mode="filter",
            bbwp_enabled=True,
            bbwp_signal_mode="filter",
            bbwp_threshold_long=50,
            bbwp_ma_filter="disabled",
            entry_operator=ConditionOperator.AND,
            trade_direction=TradeDirection.LONG_ONLY,
        )

        result = SignalGenerator(params).generate_entry_signals(df)

        # Bar 1: PAMRP trigger fires (50→10), MA True, BBWP True (30<50) → enter
        # Bar 3: PAMRP trigger fires (50→10), MA True, BBWP False (70>50) → blocked by AND
        assert result['entry_long'].tolist() == [False, True, False, False, False]


class TestSignalModeParamOverride:
    def test_signal_mode_param_filter_override(self):
        """Setting pamrp_signal_mode='filter' makes PAMRP a state filter.
        With RSI as the only trigger, entry fires on RSI crossing while
        PAMRP state is True (below threshold) — not on PAMRP transition."""
        # PAMRP stays below threshold (always True as filter)
        # RSI crosses oversold at bars 1 and 3
        df = pd.DataFrame({
            'pamrp_entry': [10.0, 10.0, 10.0, 10.0, 10.0],
            'rsi':         [50.0, 20.0, 50.0, 20.0, 20.0],
        }, index=pd.RangeIndex(5))
        params = _minimal_params(
            pamrp_enabled=True,
            pamrp_signal_mode="filter",  # PAMRP acts as regime filter
            pamrp_entry_long=20,
            pamrp_entry_short=80,
            rsi_enabled=True,
            rsi_oversold=30,
            rsi_overbought=70,
            entry_operator=ConditionOperator.AND,
            trade_direction=TradeDirection.LONG_ONLY,
        )

        result = SignalGenerator(params).generate_entry_signals(df)

        # Bar 1: RSI trigger fires, PAMRP filter True (10 < 20) → enter
        # Bar 3: RSI trigger fires, PAMRP filter True (10 < 20) → enter
        # Bar 4: RSI stays below 30 (no new crossing), no trigger → no entry
        assert result['entry_long'].tolist() == [False, True, False, True, False]
