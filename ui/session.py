"""
ui/session.py
=============
Session-state initialisation for Strategy Lab.

Call init_session_state() once at the top of app.py, after page config.
get_default_params() is kept here so it is the single source of truth
for every strategy parameter key and its initial value.
"""

import streamlit as st
from typing import Any, Dict

from src.backtest import DEFAULT_COMMISSION_PCT, DEFAULT_SLIPPAGE_PCT
from ui.state_migration import migrate_legacy_pamrp_params, migrate_legacy_pamrp_pins


def get_default_params() -> Dict[str, Any]:
    return {
        'trade_direction': 'Long Only',
        'entry_operator': 'and',
        'exit_operator': 'or',
        'allow_same_bar_exit': True,
        'allow_same_bar_reversal': False,
        'entry_conflict_mode': 'skip',
        'position_size_pct': 100.0, 'use_kelly': False, 'kelly_fraction': 0.5,
        'pamrp_enabled': False, 'pamrp_entry_length': 21, 'pamrp_entry_long': 20, 'pamrp_entry_short': 80,
        'pamrp_exit_length': 21, 'pamrp_exit_long': 70, 'pamrp_exit_short': 30,
        'bbwp_enabled': False, 'bbwp_length': 13, 'bbwp_lookback': 252, 'bbwp_sma_length': 5,
        'bbwp_threshold_long': 50, 'bbwp_threshold_short': 50, 'bbwp_ma_filter': 'disabled',
        'adx_enabled': False, 'adx_length': 14, 'adx_smoothing': 14, 'adx_threshold': 20,
        'ma_trend_enabled': False, 'ma_fast_length': 50, 'ma_slow_length': 200, 'ma_type': 'sma',
        'rsi_enabled': False, 'rsi_length': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
        'volume_enabled': False, 'volume_ma_length': 20, 'volume_multiplier': 1.0,
        'supertrend_enabled': False, 'supertrend_period': 10, 'supertrend_multiplier': 3.0,
        'vwap_enabled': False,
        'macd_enabled': False, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'stop_loss_enabled': False, 'stop_loss_pct_long': 3.0, 'stop_loss_pct_short': 3.0,
        'take_profit_enabled': False, 'take_profit_pct_long': 5.0, 'take_profit_pct_short': 5.0,
        'trailing_stop_enabled': False, 'trailing_stop_pct': 2.0,
        'atr_trailing_enabled': False, 'atr_length': 14, 'atr_multiplier': 2.0,
        'pamrp_exit_enabled': False,
        'stoch_rsi_exit_enabled': False, 'stoch_rsi_length': 14, 'stoch_rsi_k': 3, 'stoch_rsi_d': 3,
        'stoch_rsi_overbought': 80, 'stoch_rsi_oversold': 20,
        'time_exit_enabled': False, 'time_exit_bars': 20,
        'ma_exit_enabled': False, 'ma_exit_fast': 10, 'ma_exit_slow': 20,
        'bbwp_exit_enabled': False, 'bbwp_exit_threshold': 80,
        # ── New entry filters ──────────────────────────────────────────────────
        'bb_enabled': False, 'bb_length': 20, 'bb_mult': 2.0, 'bb_mode': 'squeeze',
        'stoch_entry_enabled': False, 'stoch_entry_k_period': 14, 'stoch_entry_d_period': 3,
        'stoch_entry_slowing': 3, 'stoch_entry_oversold': 20, 'stoch_entry_overbought': 80,
        'cci_enabled': False, 'cci_length': 20, 'cci_oversold': -100, 'cci_overbought': 100,
        'willr_enabled': False, 'willr_length': 14, 'willr_oversold': -80, 'willr_overbought': -20,
        'obv_enabled': False, 'obv_ma_length': 20,
        'donchian_enabled': False, 'donchian_length': 20,
        'keltner_enabled': False, 'keltner_length': 20, 'keltner_mult': 1.5,
        'psar_enabled': False, 'psar_af_start': 0.02, 'psar_af_step': 0.02, 'psar_af_max': 0.2,
        'ichi_enabled': False, 'ichi_tenkan': 9, 'ichi_kijun': 26, 'ichi_senkou_b': 52,
        'hull_enabled': False, 'hull_length': 20,
        'trix_enabled': False, 'trix_length': 15, 'trix_signal': 9,
        # ── Visual indicators (display-only, not strategy filters) ────────────
        'hpdr_enabled': False,
        'hpdr_lookback': 252,
        'rsi_div_enabled': False,
        'rsi_div_length': 14,
        'rsi_div_pivot_left': 5,
        'rsi_div_pivot_right': 5,
    }


def init_session_state() -> None:
    """
    Initialise every session-state key that the app depends on.
    Safe to call on every rerun — existing values are never overwritten.
    """
    defaults = [
        ('params', get_default_params()),
        ('df', None),
        ('multi_df', {}),
        ('backtest_results', None),
        ('optimization_results', None),
        ('capital', 10000),
        ('commission', DEFAULT_COMMISSION_PCT),
        ('slippage', DEFAULT_SLIPPAGE_PCT),
        ('pinned_params', set()),
    ]
    for key, default in defaults:
        if key not in st.session_state:
            st.session_state[key] = default

    st.session_state.params = migrate_legacy_pamrp_params(st.session_state.params)
    st.session_state.pinned_params = migrate_legacy_pamrp_pins(
        st.session_state.get('pinned_params', set())
    )

    # Forward-fill any new keys missing from an older session state
    for k, v in get_default_params().items():
        if k not in st.session_state.params:
            st.session_state.params[k] = v
