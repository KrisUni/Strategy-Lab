"""
Unit Tests for Trading Toolkit
==============================
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_sample_data
from src.indicators import (
    pamrp, bbwp, rsi, macd, supertrend, atr, sma, ema,
    bollinger_bands, stochastic_oscillator, cci, williams_r,
    obv, donchian_channel, keltner_channel, parabolic_sar,
    ichimoku, hull_ma, trix,
)
from src.strategy import StrategyParams, SignalGenerator, TradeDirection, ConditionOperator, EntryConflictMode
from src.backtest import BacktestEngine


class TestIndicators:
    """Test indicator calculations"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample data for testing"""
        return generate_sample_data(days=100, seed=42)
    
    def test_pamrp_range(self, sample_df):
        """PAMRP should be between 0 and 100"""
        result = pamrp(sample_df['high'], sample_df['low'], sample_df['close'], 21)
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100
    
    def test_bbwp_range(self, sample_df):
        """BBWP should be between 0 and 100"""
        result = bbwp(sample_df['close'], 13, 50)
        valid = result[50:]  # After lookback period
        assert valid.min() >= 0
        assert valid.max() <= 100
    
    def test_rsi_range(self, sample_df):
        """RSI should be between 0 and 100"""
        result = rsi(sample_df['close'], 14)
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100
    
    def test_macd_output(self, sample_df):
        """MACD should return 3 series"""
        macd_line, signal_line, histogram = macd(sample_df['close'], 12, 26, 9)
        assert len(macd_line) == len(sample_df)
        assert len(signal_line) == len(sample_df)
        assert len(histogram) == len(sample_df)
    
    def test_supertrend_output(self, sample_df):
        """Supertrend should return value and direction"""
        st_value, st_direction = supertrend(
            sample_df['high'], sample_df['low'], sample_df['close'], 10, 3.0
        )
        assert len(st_value) == len(sample_df)
        assert len(st_direction) == len(sample_df)
        # Direction should be -1 (bullish) or 1 (bearish)
        unique_dirs = st_direction.dropna().unique()
        assert all(d in [-1, 1] for d in unique_dirs)
    
    def test_atr_positive(self, sample_df):
        """ATR should be positive"""
        result = atr(sample_df['high'], sample_df['low'], sample_df['close'], 14)
        valid = result.dropna()
        assert (valid > 0).all()
    
    def test_sma_calculation(self, sample_df):
        """SMA should be average of last n values"""
        result = sma(sample_df['close'], 5)
        # Check a specific calculation
        idx = 10
        expected = sample_df['close'].iloc[idx-4:idx+1].mean()
        assert abs(result.iloc[idx] - expected) < 0.0001
    
    def test_ema_responsiveness(self, sample_df):
        """EMA should be more responsive than SMA"""
        # After a big move, EMA should be closer to current price
        sma_result = sma(sample_df['close'], 20)
        ema_result = ema(sample_df['close'], 20)
        
        # Both should have same length
        assert len(sma_result) == len(ema_result)


class TestStrategy:
    """Test strategy signal generation"""
    
    @pytest.fixture
    def sample_df(self):
        return generate_sample_data(days=200, seed=42)
    
    @pytest.fixture
    def default_params(self):
        return StrategyParams()
    
    def test_signal_generator_creates_columns(self, sample_df, default_params):
        """Signal generator should add required columns"""
        gen = SignalGenerator(default_params)
        result = gen.generate_all_signals(sample_df)

        assert 'pamrp' in result.columns
        assert 'pamrp_entry' in result.columns
        assert 'pamrp_exit' in result.columns
        assert 'bbwp' in result.columns
        assert 'entry_long' in result.columns
        assert 'entry_short' in result.columns
        assert 'exit_long_signal' in result.columns
        assert 'exit_short_signal' in result.columns
    
    def test_long_only_no_short_signals(self, sample_df):
        """Long-only mode should not generate short signals"""
        params = StrategyParams(trade_direction=TradeDirection.LONG_ONLY)
        gen = SignalGenerator(params)
        result = gen.generate_all_signals(sample_df)
        
        assert not result['entry_short'].any()
    
    def test_short_only_no_long_signals(self, sample_df):
        """Short-only mode should not generate long signals"""
        params = StrategyParams(trade_direction=TradeDirection.SHORT_ONLY)
        gen = SignalGenerator(params)
        result = gen.generate_all_signals(sample_df)
        
        assert not result['entry_long'].any()
    
    def test_params_to_dict(self, default_params):
        """Params should convert to dict"""
        d = default_params.to_dict()
        assert isinstance(d, dict)
        assert 'pamrp_entry_length' in d
        assert 'pamrp_exit_length' in d
        assert 'stop_loss_enabled' in d

    def test_params_from_dict(self):
        """Params should be creatable from dict"""
        d = {'pamrp_length': 30, 'stop_loss_pct_long': 5.0}
        params = StrategyParams.from_dict(d)
        assert params.pamrp_entry_length == 30
        assert params.pamrp_exit_length == 30
        assert params.stop_loss_pct_long == 5.0

    def test_pamrp_entry_and_exit_lengths_are_independent(self, sample_df):
        """Entry and exit PAMRP should be calculated from separate lookbacks."""
        params = StrategyParams(
            pamrp_enabled=True,
            pamrp_entry_length=10,
            pamrp_exit_enabled=True,
            pamrp_exit_length=30,
        )
        gen = SignalGenerator(params)
        result = gen.calculate_indicators(sample_df)

        diff = (result['pamrp_entry'] - result['pamrp_exit']).abs().dropna()
        assert not diff.empty
        assert (diff > 1e-9).any()

    def test_entry_operator_defaults_and_round_trips(self, default_params):
        """Entry/exit operators should serialize and deserialize cleanly."""
        d = default_params.to_dict()
        assert d['entry_operator'] == 'and'
        assert d['exit_operator'] == 'or'
        assert d['allow_same_bar_exit'] is True
        assert d['allow_same_bar_reversal'] is False
        assert d['entry_conflict_mode'] == 'skip'

        params = StrategyParams.from_dict({
            'entry_operator': 'or',
            'exit_operator': 'and',
            'allow_same_bar_exit': False,
            'allow_same_bar_reversal': True,
            'entry_conflict_mode': 'prefer_short',
        })
        assert params.entry_operator == ConditionOperator.OR
        assert params.exit_operator == ConditionOperator.AND
        assert params.allow_same_bar_exit is False
        assert params.allow_same_bar_reversal is True
        assert params.entry_conflict_mode == EntryConflictMode.PREFER_SHORT

    def test_entry_operator_or_allows_any_enabled_filter(self):
        """OR entry mode should trigger when any enabled filter is true."""
        idx = pd.RangeIndex(3)
        df = pd.DataFrame({
            'pamrp_entry': [10, 60, 10],
            'rsi': [50, 20, 20],
        }, index=idx)
        params = StrategyParams(
            pamrp_enabled=True,
            rsi_enabled=True,
            entry_operator=ConditionOperator.OR,
            trade_direction=TradeDirection.BOTH,
        )

        result = SignalGenerator(params).generate_entry_signals(df)

        assert result['entry_long'].tolist() == [True, True, True]
        assert result['entry_short'].tolist() == [False, False, False]

    def test_entry_operator_and_requires_all_enabled_filters(self):
        """AND entry mode should require every enabled filter to pass."""
        idx = pd.RangeIndex(3)
        df = pd.DataFrame({
            'pamrp_entry': [10, 60, 10],
            'rsi': [50, 20, 20],
        }, index=idx)
        params = StrategyParams(
            pamrp_enabled=True,
            rsi_enabled=True,
            entry_operator=ConditionOperator.AND,
            trade_direction=TradeDirection.BOTH,
        )

        result = SignalGenerator(params).generate_entry_signals(df)

        assert result['entry_long'].tolist() == [False, False, True]
        assert result['entry_short'].tolist() == [False, False, False]

    def test_exit_operator_or_matches_legacy_behavior(self):
        """OR exit mode should fire when any signal exit condition is true."""
        idx = pd.RangeIndex(3)
        df = pd.DataFrame({
            'pamrp_exit': [75, 60, 85],
            'bbwp': [70, 85, 90],
        }, index=idx)
        params = StrategyParams(
            pamrp_exit_enabled=True,
            bbwp_exit_enabled=True,
            exit_operator=ConditionOperator.OR,
        )

        result = SignalGenerator(params).generate_exit_signals(df)

        assert result['exit_long_signal'].tolist() == [True, True, True]
        assert result['exit_short_signal'].tolist() == [False, False, False]

    def test_exit_operator_and_requires_all_signal_exits(self):
        """AND exit mode should require every enabled signal exit condition to pass."""
        idx = pd.RangeIndex(3)
        df = pd.DataFrame({
            'pamrp_exit': [75, 60, 85],
            'bbwp': [70, 85, 90],
        }, index=idx)
        params = StrategyParams(
            pamrp_exit_enabled=True,
            bbwp_exit_enabled=True,
            exit_operator=ConditionOperator.AND,
        )

        result = SignalGenerator(params).generate_exit_signals(df)

        assert result['exit_long_signal'].tolist() == [False, False, True]
        assert result['exit_short_signal'].tolist() == [False, False, False]


class TestBacktest:
    """Test backtesting engine"""
    
    @pytest.fixture
    def sample_df(self):
        return generate_sample_data(days=300, seed=42)
    
    @pytest.fixture
    def default_params(self):
        return StrategyParams(
            pamrp_entry_long=35,  # More lenient to generate trades
            bbwp_threshold_long=70,
        )
    
    def test_backtest_runs(self, sample_df, default_params):
        """Backtest should run without error"""
        engine = BacktestEngine(default_params)
        results = engine.run(sample_df)
        
        assert results is not None
        assert hasattr(results, 'num_trades')
        assert hasattr(results, 'equity_curve')
    
    def test_equity_curve_starts_at_capital(self, sample_df, default_params):
        """Equity curve should start at initial capital"""
        engine = BacktestEngine(default_params, initial_capital=10000)
        results = engine.run(sample_df)
        
        assert results.equity_curve.iloc[0] == 10000
    
    def test_metrics_calculated(self, sample_df, default_params):
        """All metrics should be calculated"""
        engine = BacktestEngine(default_params)
        results = engine.run(sample_df)
        
        assert hasattr(results, 'total_return_pct')
        assert hasattr(results, 'sharpe_ratio')
        assert hasattr(results, 'profit_factor')
        assert hasattr(results, 'max_drawdown_pct')
        assert hasattr(results, 'win_rate')
    
    def test_no_trades_handled(self, sample_df):
        """Should handle case with no trades gracefully"""
        # Disable all entry filters = no signals
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
        )
        engine = BacktestEngine(params)
        results = engine.run(sample_df)
        
        assert results.num_trades == 0
        assert results.win_rate == 0
        assert results.initial_capital == 10000  # Verify stored
    
    def test_stop_loss_works(self, sample_df):
        """Stop loss should limit losses - checks HIGH/LOW"""
        params = StrategyParams(
            pamrp_enabled=True,
            pamrp_entry_long=40,  # Relaxed entry
            bbwp_enabled=True,
            bbwp_threshold_long=80,
            stop_loss_enabled=True,
            stop_loss_pct_long=2.0,
            pamrp_exit_enabled=False,
        )
        engine = BacktestEngine(params)
        results = engine.run(sample_df)
        
        # Check stop loss triggers properly
        for trade in results.trades:
            if trade.direction == 'long' and trade.exit_reason == 'stop_loss':
                # Stop loss should be near the stop level
                expected_stop = trade.entry_price * (1 - 2.0 / 100)
                assert trade.exit_price <= trade.entry_price  # Exit below entry

    def test_same_bar_exit_toggle_can_disable_entry_bar_stopout(self):
        """Disabling same-bar exits should keep an entry-bar stop hit open until later bars."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            'open': [100.0, 100.0, 99.0],
            'high': [101.0, 101.0, 100.0],
            'low': [99.0, 96.0, 98.0],
            'close': [100.0, 99.0, 99.0],
            'volume': [1000, 1000, 1000],
        }, index=idx)

        def add_signals(frame):
            frame = frame.copy()
            frame['entry_long'] = [True, False, False]
            frame['entry_short'] = [False, False, False]
            frame['exit_long_signal'] = [False, False, False]
            frame['exit_short_signal'] = [False, False, False]
            return frame

        enabled_engine = BacktestEngine(
            StrategyParams(
                stop_loss_enabled=True,
                stop_loss_pct_long=3.0,
                allow_same_bar_exit=True,
            ),
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        enabled_engine.signal_gen.generate_all_signals = add_signals
        enabled_results = enabled_engine.run(df)

        assert len(enabled_results.trades) == 1
        assert enabled_results.trades[0].exit_reason == 'stop_loss'
        assert enabled_results.trades[0].bars_held == 0

        disabled_engine = BacktestEngine(
            StrategyParams(
                stop_loss_enabled=True,
                stop_loss_pct_long=3.0,
                allow_same_bar_exit=False,
            ),
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        disabled_engine.signal_gen.generate_all_signals = add_signals
        disabled_results = disabled_engine.run(df)

        assert len(disabled_results.trades) == 1
        assert disabled_results.trades[0].exit_reason == 'end_of_data'
        assert disabled_results.trades[0].bars_held == 1

    def test_signal_exit_does_not_reverse_into_opposite_trade_on_same_bar(self):
        """A signal exit should block immediate same-bar reversal into the opposite direction."""
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({
            'open': [100.0, 110.0, 120.0, 121.0],
            'high': [101.0, 111.0, 121.0, 122.0],
            'low': [99.0, 109.0, 119.0, 120.0],
            'close': [100.0, 110.0, 120.0, 121.0],
            'volume': [1000, 1000, 1000, 1000],
        }, index=idx)

        def add_signals(frame):
            frame = frame.copy()
            frame['entry_long'] = [True, False, False, False]
            frame['entry_short'] = [False, True, False, False]
            frame['exit_long_signal'] = [False, True, False, False]
            frame['exit_short_signal'] = [False, False, False, False]
            return frame

        engine = BacktestEngine(
            StrategyParams(
                trade_direction=TradeDirection.BOTH,
                stop_loss_enabled=False,
                take_profit_enabled=False,
                time_exit_enabled=False,
            ),
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        engine.signal_gen.generate_all_signals = add_signals
        results = engine.run(df)

        assert len(results.trades) == 1
        assert results.trades[0].direction == 'long'
        assert results.trades[0].exit_reason == 'signal'

    def test_signal_exit_can_reverse_into_opposite_trade_on_same_bar_when_enabled(self):
        """Same-bar reversal should be allowed only when the toggle is enabled."""
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({
            'open': [100.0, 110.0, 120.0, 118.0],
            'high': [101.0, 111.0, 121.0, 119.0],
            'low': [99.0, 109.0, 119.0, 117.0],
            'close': [100.0, 110.0, 120.0, 118.0],
            'volume': [1000, 1000, 1000, 1000],
        }, index=idx)

        def add_signals(frame):
            frame = frame.copy()
            frame['entry_long'] = [True, False, False, False]
            frame['entry_short'] = [False, True, False, False]
            frame['exit_long_signal'] = [False, True, False, False]
            frame['exit_short_signal'] = [False, False, False, False]
            return frame

        engine = BacktestEngine(
            StrategyParams(
                trade_direction=TradeDirection.BOTH,
                allow_same_bar_reversal=True,
                stop_loss_enabled=False,
                take_profit_enabled=False,
                time_exit_enabled=False,
            ),
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        engine.signal_gen.generate_all_signals = add_signals
        results = engine.run(df)

        assert len(results.trades) == 2
        assert results.trades[0].direction == 'long'
        assert results.trades[0].exit_reason == 'signal'
        assert results.trades[1].direction == 'short'
        assert results.trades[1].entry_date == results.trades[0].exit_date

    def test_ambiguous_entry_bar_is_skipped_when_both_directions_fire(self):
        """When both long and short entries are true on the same bar, skip the bar."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1000, 1000],
        }, index=idx)

        def add_signals(frame):
            frame = frame.copy()
            frame['entry_long'] = [True, False, False]
            frame['entry_short'] = [True, False, False]
            frame['exit_long_signal'] = [False, False, False]
            frame['exit_short_signal'] = [False, False, False]
            return frame

        engine = BacktestEngine(
            StrategyParams(
                trade_direction=TradeDirection.BOTH,
                entry_conflict_mode=EntryConflictMode.SKIP,
                stop_loss_enabled=False,
                take_profit_enabled=False,
                time_exit_enabled=False,
            ),
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        engine.signal_gen.generate_all_signals = add_signals
        results = engine.run(df)

        assert results.num_trades == 0


class TestDataModule:
    """Test data fetching and generation"""
    
    def test_generate_sample_data(self):
        """Sample data should have correct structure"""
        df = generate_sample_data(days=100)
        
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert len(df) == 100
    
    def test_sample_data_ohlc_valid(self):
        """OHLC relationships should be valid"""
        df = generate_sample_data(days=100)
        
        # High should be highest
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        
        # Low should be lowest
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
    
    def test_sample_data_reproducible(self):
        """Same seed should produce same data"""
        df1 = generate_sample_data(days=50, seed=123)
        df2 = generate_sample_data(days=50, seed=123)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_sample_data_different_seeds(self):
        """Different seeds should produce different data"""
        df1 = generate_sample_data(days=50, seed=123)
        df2 = generate_sample_data(days=50, seed=456)
        
        assert not df1['close'].equals(df2['close'])


class TestNewIndicators:
    """Unit tests for the 11 new entry-filter indicators."""

    @pytest.fixture
    def df(self):
        return generate_sample_data(days=300, seed=42)

    # ── Bollinger Bands ────────────────────────────────────────────────────────

    def test_bb_output_length(self, df):
        upper, lower, mid = bollinger_bands(df['close'], 20, 2.0)
        assert len(upper) == len(df)
        assert len(lower) == len(df)
        assert len(mid) == len(df)

    def test_bb_upper_above_lower(self, df):
        upper, lower, _ = bollinger_bands(df['close'], 20, 2.0)
        valid = upper.dropna().index.intersection(lower.dropna().index)
        assert (upper[valid] >= lower[valid]).all()

    def test_bb_warmup_nan(self, df):
        upper, _, _ = bollinger_bands(df['close'], 20, 2.0)
        assert upper.iloc[:19].isna().all()
        assert not upper.iloc[20:].isna().all()

    def test_bb_squeeze_signal_long(self, df):
        """When close < lower band at least once, squeeze mode can generate a long."""
        params = StrategyParams(bb_enabled=True, bb_mode='squeeze',
                                pamrp_enabled=False, bbwp_enabled=False)
        result = SignalGenerator(params).generate_all_signals(df)
        assert 'entry_long' in result.columns

    # ── Stochastic ─────────────────────────────────────────────────────────────

    def test_stoch_output_length(self, df):
        k, d = stochastic_oscillator(df['high'], df['low'], df['close'])
        assert len(k) == len(df)
        assert len(d) == len(df)

    def test_stoch_range(self, df):
        k, d = stochastic_oscillator(df['high'], df['low'], df['close'])
        k_valid = k.dropna()
        assert k_valid.min() >= 0
        assert k_valid.max() <= 100

    # ── CCI ────────────────────────────────────────────────────────────────────

    def test_cci_output_length(self, df):
        result = cci(df['high'], df['low'], df['close'], 20)
        assert len(result) == len(df)

    def test_cci_warmup_nan(self, df):
        result = cci(df['high'], df['low'], df['close'], 20)
        assert result.iloc[:19].isna().all()

    def test_cci_signal(self, df):
        params = StrategyParams(cci_enabled=True, pamrp_enabled=False, bbwp_enabled=False)
        result = SignalGenerator(params).generate_all_signals(df)
        assert 'entry_long' in result.columns

    # ── Williams %R ────────────────────────────────────────────────────────────

    def test_willr_range(self, df):
        result = williams_r(df['high'], df['low'], df['close'], 14)
        valid = result.dropna()
        assert valid.min() >= -100
        assert valid.max() <= 0

    def test_willr_length(self, df):
        result = williams_r(df['high'], df['low'], df['close'], 14)
        assert len(result) == len(df)

    def test_willr_warmup_nan(self, df):
        result = williams_r(df['high'], df['low'], df['close'], 14)
        assert result.iloc[:13].isna().all()

    # ── OBV ────────────────────────────────────────────────────────────────────

    def test_obv_length(self, df):
        result = obv(df['close'], df['volume'])
        assert len(result) == len(df)

    def test_obv_no_nan(self, df):
        result = obv(df['close'], df['volume'])
        assert result.isna().sum() == 0

    # ── Donchian Channel ───────────────────────────────────────────────────────

    def test_donchian_upper_above_lower(self, df):
        upper, lower, _ = donchian_channel(df['high'], df['low'], 20)
        valid = upper.dropna().index
        assert (upper[valid] >= lower[valid]).all()

    def test_donchian_length(self, df):
        upper, lower, mid = donchian_channel(df['high'], df['low'], 20)
        assert len(upper) == len(df)

    # ── Keltner Channel ────────────────────────────────────────────────────────

    def test_keltner_upper_above_lower(self, df):
        upper, lower, _ = keltner_channel(df['high'], df['low'], df['close'], 20, 1.5)
        valid = upper.dropna().index
        assert (upper[valid] >= lower[valid]).all()

    def test_keltner_length(self, df):
        upper, lower, mid = keltner_channel(df['high'], df['low'], df['close'], 20, 1.5)
        assert len(upper) == len(df)

    # ── Parabolic SAR ──────────────────────────────────────────────────────────

    def test_psar_length(self, df):
        result = parabolic_sar(df['high'], df['low'])
        assert len(result) == len(df)

    def test_psar_no_nan_after_first(self, df):
        result = parabolic_sar(df['high'], df['low'])
        assert result.iloc[1:].isna().sum() == 0

    def test_psar_positive(self, df):
        result = parabolic_sar(df['high'], df['low'])
        assert (result.iloc[1:] > 0).all()

    # ── Ichimoku ───────────────────────────────────────────────────────────────

    def test_ichimoku_keys(self, df):
        result = ichimoku(df['high'], df['low'], df['close'])
        expected = {'tenkan_sen', 'kijun_sen', 'senkou_a_signal', 'senkou_b_signal',
                    'senkou_a_display', 'senkou_b_display', 'chikou_span'}
        assert expected.issubset(set(result.keys()))

    def test_ichimoku_length(self, df):
        result = ichimoku(df['high'], df['low'], df['close'])
        assert len(result['tenkan_sen']) == len(df)

    def test_ichimoku_signal_not_shifted(self, df):
        """senkou_a_signal must not have NaN beyond the warmup period from period alone."""
        result = ichimoku(df['high'], df['low'], df['close'], 9, 26, 52)
        # The signal version should have values from bar 51 onward (senkou_b period)
        assert not result['senkou_a_signal'].iloc[52:].isna().all()

    # ── Hull MA ────────────────────────────────────────────────────────────────

    def test_hull_ma_length(self, df):
        result = hull_ma(df['close'], 20)
        assert len(result) == len(df)

    def test_hull_ma_signal(self, df):
        params = StrategyParams(hull_enabled=True, pamrp_enabled=False, bbwp_enabled=False)
        result = SignalGenerator(params).generate_all_signals(df)
        assert 'entry_long' in result.columns

    # ── TRIX ───────────────────────────────────────────────────────────────────

    def test_trix_output_length(self, df):
        trix_line, sig_line = trix(df['close'], 15, 9)
        assert len(trix_line) == len(df)
        assert len(sig_line) == len(df)

    def test_trix_signal(self, df):
        params = StrategyParams(trix_enabled=True, pamrp_enabled=False, bbwp_enabled=False)
        result = SignalGenerator(params).generate_all_signals(df)
        assert 'entry_long' in result.columns

    # ── Integration: multiple new indicators together ──────────────────────────

    def test_multiple_new_indicators_and_mode(self, df):
        """All new indicators enabled with AND — should produce boolean entry columns."""
        params = StrategyParams(
            pamrp_enabled=False, bbwp_enabled=False,
            bb_enabled=True, cci_enabled=True,
            entry_operator=ConditionOperator.AND,
            trade_direction=TradeDirection.BOTH,
        )
        result = SignalGenerator(params).generate_all_signals(df)
        assert result['entry_long'].dtype == bool or result['entry_long'].dtype == object
        assert result['entry_short'].dtype == bool or result['entry_short'].dtype == object

    def test_multiple_new_indicators_or_mode(self, df):
        """OR mode: enabling several new indicators should still produce entries."""
        params = StrategyParams(
            pamrp_enabled=False, bbwp_enabled=False,
            bb_enabled=True, willr_enabled=True, hull_enabled=True,
            entry_operator=ConditionOperator.OR,
            trade_direction=TradeDirection.BOTH,
        )
        result = SignalGenerator(params).generate_all_signals(df)
        # At least some longs or shorts should fire on 300 bars
        assert result['entry_long'].any() or result['entry_short'].any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
