"""
Tests: entry fires when signal exit is already True.
"""
import pandas as pd
import pytest

from src.backtest import BacktestEngine
from src.strategy import (
    StrategyParams, TradeDirection, EntryConflictMode, EntryExitConflictMode,
)


class _NoopSignalGen:
    """Stub: returns df unchanged so tests can inject signal columns directly."""
    def generate_all_signals(self, df):
        return df


def _make_engine(mode="skip", exit_operator="or", **kw):
    params = StrategyParams(
        entry_exit_conflict_mode=mode,
        exit_operator=exit_operator,
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
        **kw,
    )
    engine = BacktestEngine(params, commission_pct=0.0, slippage_pct=0.0)
    engine.signal_gen = _NoopSignalGen()
    return engine


def _make_df(n, entry_long, exit_long_signal,
             entry_short=None, exit_short_signal=None,
             extra_cols=None, price=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "open":   [price] * n,
        "high":   [price * 1.01] * n,
        "low":    [price * 0.99] * n,
        "close":  [price] * n,
        "volume": [1_000] * n,
    }, index=idx)
    df["entry_long"]        = entry_long
    df["entry_short"]       = entry_short       if entry_short       is not None else [False] * n
    df["exit_long_signal"]  = exit_long_signal
    df["exit_short_signal"] = exit_short_signal if exit_short_signal is not None else [False] * n
    if extra_cols:
        for col, vals in extra_cols.items():
            df[col] = vals
    return df


# ── SKIP mode ─────────────────────────────────────────────────────────────────

class TestSkipMode:

    def test_skip_long_entry_when_exit_signal_active_same_bar(self):
        df = _make_df(3, [True, False, False], [True, False, False])
        assert len(_make_engine("skip").run(df).trades) == 0

    def test_skip_short_entry_when_exit_signal_active_same_bar(self):
        df = _make_df(3, [False]*3, [False]*3,
                      entry_short=[True, False, False],
                      exit_short_signal=[True, False, False])
        assert len(_make_engine("skip").run(df).trades) == 0

    def test_skip_allows_entry_when_exit_inactive(self):
        df = _make_df(4, [True, False, False, False], [False, False, True, False])
        result = _make_engine("skip").run(df)
        assert len(result.trades) == 1

    def test_skip_allows_entry_next_bar_when_exit_clears(self):
        # Bar 0: entry+exit both True → skip.
        # Bar 1: entry still True, exit False → enter at bar 2 open.
        df = _make_df(5, [True, True, False, False, False],
                      [True, False, False, True, False])
        result = _make_engine("skip").run(df)
        assert len(result.trades) == 1
        assert result.trades[0].entry_idx == 2

    def test_skip_does_not_block_unrelated_direction(self):
        # Long is blocked by exit, short is not → short should open.
        df = _make_df(5, [True, False, False, False, False],
                      [True, False, False, False, False],
                      entry_short=[True, False, False, False, False],
                      exit_short_signal=[False]*5)
        eng = _make_engine("skip",
                           trade_direction=TradeDirection.BOTH,
                           entry_conflict_mode=EntryConflictMode.PREFER_SHORT)
        result = eng.run(df)
        assert len(result.trades) == 1
        assert result.trades[0].direction == "short"

    def test_one_bar_round_trip_does_not_occur(self):
        df = _make_df(4, [True, False, False, False], [True, True, False, False])
        assert len(_make_engine("skip").run(df).trades) == 0

    def test_metrics_not_polluted_by_skipped_entry(self):
        df = _make_df(3, [True, False, False], [True, False, False])
        result = _make_engine("skip").run(df)
        assert result.num_trades == 0
        assert result.total_return_pct == pytest.approx(0.0, abs=1e-9)


# ── DEFER mode ────────────────────────────────────────────────────────────────

class TestDeferMode:

    def test_defer_takes_trade_when_exit_active_at_entry(self):
        """DEFER enters even when exit is True at entry bar."""
        spec_a = [True, True, True, False, True, False, False, False]
        df = _make_df(8, [True]+[False]*7, spec_a,
                      extra_cols={"exit_long_spec_a": spec_a,
                                  "exit_short_spec_a": [False]*8})
        result = _make_engine("defer").run(df)
        assert len(result.trades) == 1
        assert result.trades[0].entry_idx == 1

    def test_defer_arm_blocked_exit_cannot_fire_while_never_cleared(self):
        """Arm-blocked exit stays True always → never clears → exit at end_of_data."""
        n = 10
        spec_a = [True] * n
        df = _make_df(n, [True]+[False]*(n-1), spec_a,
                      extra_cols={"exit_long_spec_a": spec_a,
                                  "exit_short_spec_a": [False]*n})
        result = _make_engine("defer").run(df)
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "end_of_data"

    def test_defer_blocked_exit_fires_after_flip_false_then_true(self):
        """Arm-blocked spec must flip False then back True before it can fire.
        Bars: 0=entry signal, 1=enter, 2=exit True(blocked), 3=exit False(clear),
              4=exit True(armed) → exit at bar 5 open."""
        spec_a = [True, True, True, False, True, False, False]
        df = _make_df(7, [True]+[False]*6, spec_a,
                      extra_cols={"exit_long_spec_a": spec_a,
                                  "exit_short_spec_a": [False]*7})
        result = _make_engine("defer").run(df)
        assert len(result.trades) == 1
        assert result.trades[0].exit_idx == 5
        assert result.trades[0].exit_reason == "signal"

    def test_defer_non_blocked_spec_fires_normally_under_or(self):
        """DEFER + OR: spec_a arm-blocked, spec_b not blocked → spec_b fires."""
        spec_a = [True,  True,  True,  True,  True,  True]
        spec_b = [False, False, True,  True,  True,  True]
        combined = [a or b for a, b in zip(spec_a, spec_b)]
        df = _make_df(6, [True]+[False]*5, combined,
                      extra_cols={"exit_long_spec_a": spec_a,
                                  "exit_short_spec_a": [False]*6,
                                  "exit_long_spec_b": spec_b,
                                  "exit_short_spec_b": [False]*6})
        result = _make_engine("defer", exit_operator="or").run(df)
        # spec_b False at entry (not blocked), fires bar 2 → exit at bar 3 open
        assert len(result.trades) == 1
        assert result.trades[0].exit_idx == 3
        assert result.trades[0].exit_reason == "signal"

    def test_defer_and_operator_blocks_until_arm_cleared(self):
        """DEFER + AND: spec_a arm-blocked; AND requires both → blocked until spec_a clears."""
        spec_a = [True,  True,  True,  True,  False, True,  True,  True]
        spec_b = [False, False, True,  True,  True,  True,  True,  True]
        combined = [a and b for a, b in zip(spec_a, spec_b)]
        df = _make_df(8, [True]+[False]*7, combined,
                      extra_cols={"exit_long_spec_a": spec_a,
                                  "exit_short_spec_a": [False]*8,
                                  "exit_long_spec_b": spec_b,
                                  "exit_short_spec_b": [False]*8})
        result = _make_engine("defer", exit_operator="and").run(df)
        # spec_a clears at bar 4, re-fires bar 5; both True from bar 5
        # → exit check uses prev_row=bar5 at bar 6 open
        assert len(result.trades) == 1
        assert result.trades[0].exit_idx == 6
        assert result.trades[0].exit_reason == "signal"

    def test_defer_stop_exits_unaffected_by_arm_block(self):
        """DEFER: arm-block only gates signal exits; stop-loss fires normally."""
        n = 6
        price = 100.0
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "open":   [price] * n,
            "high":   [price * 1.01] * n,
            "low":    [price * 0.99, price * 0.99, 85.0] + [price * 0.99] * 3,
            "close":  [price] * n,
            "volume": [1_000] * n,
        }, index=idx)
        spec_a = [True] * n
        df["entry_long"]        = [True] + [False] * (n - 1)
        df["entry_short"]       = [False] * n
        df["exit_long_signal"]  = spec_a
        df["exit_short_signal"] = [False] * n
        df["exit_long_spec_a"]  = spec_a
        df["exit_short_spec_a"] = [False] * n

        eng = BacktestEngine(
            StrategyParams(
                entry_exit_conflict_mode="defer",
                stop_loss_enabled=True,
                stop_loss_pct_long=10.0,
                take_profit_enabled=False,
                trailing_stop_enabled=False,
                time_exit_enabled=False,
                allow_same_bar_exit=True,
                pamrp_enabled=False, bbwp_enabled=False, adx_enabled=False,
                ma_trend_enabled=False, rsi_enabled=False, volume_enabled=False,
                supertrend_enabled=False, vwap_enabled=False, macd_enabled=False,
                stoch_rsi_entry_enabled=False, pamrp_exit_enabled=False,
                ma_exit_enabled=False, bbwp_exit_enabled=False,
                stoch_rsi_exit_enabled=False, rsi_exit_enabled=False,
                atr_trailing_enabled=False,
            ),
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        eng.signal_gen = _NoopSignalGen()
        result = eng.run(df)
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "stop_loss"


# ── mode comparison ───────────────────────────────────────────────────────────

class TestModeComparison:

    def test_skip_and_defer_differ_on_overlap_fixture(self):
        """SKIP → 0 trades; DEFER → 1 trade on always-active exit fixture."""
        spec_a = [True] * 5
        df = _make_df(5, [True]+[False]*4, spec_a,
                      extra_cols={"exit_long_spec_a": spec_a,
                                  "exit_short_spec_a": [False]*5})
        skip_result = _make_engine("skip").run(df)
        defer_result = _make_engine("defer").run(df)
        assert len(skip_result.trades) == 0
        assert len(defer_result.trades) == 1

    def test_clean_strategies_identical_across_modes(self):
        """No overlap → SKIP and DEFER produce same trade."""
        spec_a = [False, False, False, True, False, False]
        df = _make_df(6, [True]+[False]*5, spec_a,
                      extra_cols={"exit_long_spec_a": spec_a,
                                  "exit_short_spec_a": [False]*6})
        skip_r  = _make_engine("skip").run(df)
        defer_r = _make_engine("defer").run(df)
        assert len(skip_r.trades) == len(defer_r.trades) == 1
        assert skip_r.trades[0].entry_idx == defer_r.trades[0].entry_idx
        assert skip_r.trades[0].exit_idx  == defer_r.trades[0].exit_idx

    def test_default_mode_is_skip(self):
        p = StrategyParams()
        assert p.entry_exit_conflict_mode == EntryExitConflictMode.SKIP

    def test_from_dict_coerces_string_to_enum(self):
        p = StrategyParams.from_dict({"entry_exit_conflict_mode": "defer"})
        assert p.entry_exit_conflict_mode == EntryExitConflictMode.DEFER

    def test_to_dict_serializes_enum_to_string(self):
        p = StrategyParams(entry_exit_conflict_mode=EntryExitConflictMode.DEFER)
        assert p.to_dict()["entry_exit_conflict_mode"] == "defer"
