"""
ui/helpers.py
=============
Pure helper functions and constants shared across sidebar and tab modules.

  - params_to_strategy()        — Dict → StrategyParams
  - get_active_filters_display() — list of enabled indicator labels
  - calculate_beta_alpha()       — beta / alpha / correlation vs benchmark
  - PARAM_TO_WIDGET_KEY          — sidebar widget-key lookup
  - apply_best_params_callback() — Streamlit on_click callback for "Apply Best Params"
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any

from src.strategy import StrategyParams, TradeDirection, ConditionOperator, EntryConflictMode
from ui.state_migration import migrate_legacy_pamrp_params


# ─────────────────────────────────────────────────────────────────────────────
# Widget-key mapping (param name → short Streamlit widget key)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_TO_WIDGET_KEY: Dict[str, str] = {
    'trade_direction': 'tdir',
    'entry_operator': 'eop', 'exit_operator': 'xop',
    'allow_same_bar_exit': 'same_bar_exit',
    'allow_same_bar_reversal': 'same_bar_reversal',
    'entry_conflict_mode': 'ecm',
    'pamrp_enabled': 'pe',
    'pamrp_entry_ma_length': 'ple_ma', 'pamrp_entry_lookback': 'ple_lb', 'pamrp_entry_ma_type': 'ple_mt',
    'pamrp_entry_long': 'pel', 'pamrp_entry_short': 'pes',
    'pamrp_exit_ma_length': 'pxl_ma', 'pamrp_exit_lookback': 'pxl_lb', 'pamrp_exit_ma_type': 'pxl_mt',
    'pamrp_exit_long': 'pxl_exit', 'pamrp_exit_short': 'pxs_exit',
    'bbwp_enabled': 'be', 'bbwp_length': 'bl', 'bbwp_lookback': 'blb', 'bbwp_sma_length': 'bsma',
    'bbwp_threshold_long': 'btl', 'bbwp_threshold_short': 'bts', 'bbwp_ma_filter': 'bmf',
    'adx_enabled': 'ae', 'adx_length': 'al', 'adx_threshold': 'at',
    'ma_trend_enabled': 'mae', 'ma_type': 'mat', 'ma_fast_length': 'maf', 'ma_slow_length': 'mas',
    'rsi_enabled': 're', 'rsi_length': 'rl', 'rsi_oversold': 'ros', 'rsi_overbought': 'rob',
    'volume_enabled': 've', 'volume_ma_length': 'vml', 'volume_multiplier': 'vm',
    'supertrend_enabled': 'ste', 'supertrend_period': 'stp', 'supertrend_multiplier': 'stm',
    'vwap_enabled': 'vwe', 'macd_enabled': 'mce', 'macd_fast': 'mcf', 'macd_slow': 'mcs', 'macd_signal': 'mcsi',
    'stop_loss_enabled': 'sle', 'stop_loss_pct_long': 'sll', 'stop_loss_pct_short': 'sls',
    'take_profit_enabled': 'tpe', 'take_profit_pct_long': 'tpl', 'take_profit_pct_short': 'tps',
    'trailing_stop_enabled': 'tse', 'trailing_stop_pct': 'tsp',
    'atr_trailing_enabled': 'ate', 'atr_length': 'atl', 'atr_multiplier': 'atm',
    'pamrp_exit_enabled': 'pxe',
    'stoch_rsi_exit_enabled': 'sre', 'stoch_rsi_length': 'srl', 'stoch_rsi_k': 'srk', 'stoch_rsi_d': 'srd',
    'stoch_rsi_overbought': 'srob', 'stoch_rsi_oversold': 'sros',
    'time_exit_enabled': 'txe', 'time_exit_bars': 'txb',
    'ma_exit_enabled': 'mxe', 'ma_exit_fast': 'mxf', 'ma_exit_slow': 'mxs',
    'bbwp_exit_enabled': 'bxe', 'bbwp_exit_threshold': 'bxt',
}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy param conversion
# ─────────────────────────────────────────────────────────────────────────────

def params_to_strategy(p: Dict[str, Any]) -> StrategyParams:
    """Convert the flat session-state params dict into a StrategyParams object."""
    return StrategyParams.from_dict(p)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_active_filters_display(params: Dict[str, Any]):
    """Return a list of human-readable labels for every enabled indicator / exit."""
    entry = {
        'pamrp_enabled': 'PAMRP', 'bbwp_enabled': 'BBWP', 'adx_enabled': 'ADX',
        'ma_trend_enabled': 'MA Trend', 'rsi_enabled': 'RSI', 'volume_enabled': 'Volume',
        'supertrend_enabled': 'Supertrend', 'vwap_enabled': 'VWAP', 'macd_enabled': 'MACD',
    }
    exits = {
        'stop_loss_enabled': 'Stop Loss', 'take_profit_enabled': 'Take Profit',
        'trailing_stop_enabled': 'Trailing Stop', 'atr_trailing_enabled': 'ATR Trail',
        'pamrp_exit_enabled': 'PAMRP Exit', 'stoch_rsi_exit_enabled': 'Stoch RSI Exit',
        'time_exit_enabled': 'Time Exit', 'ma_exit_enabled': 'MA Exit', 'bbwp_exit_enabled': 'BBWP Exit',
    }
    return [l for k, l in {**entry, **exits}.items() if params.get(k, False)]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark analytics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_beta_alpha(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """Compute beta, annualised alpha, and Pearson correlation vs a benchmark."""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return {'beta': 0, 'alpha': 0, 'correlation': 0}
    aligned.columns = ['strategy', 'benchmark']
    cov = aligned.cov().iloc[0, 1]
    var = aligned['benchmark'].var()
    beta = cov / var if var > 0 else 0
    alpha = (aligned['strategy'].mean() - beta * aligned['benchmark'].mean()) * 252 * 100
    return {'beta': beta, 'alpha': alpha, 'correlation': aligned.corr().iloc[0, 1]}


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit callback
# ─────────────────────────────────────────────────────────────────────────────

def apply_best_params_callback() -> None:
    """
    on_click callback for the "Apply Best Params" button in the Optimize tab.
    Writes optimised parameter values back into session state and widget keys
    so the sidebar reflects the new values on the next rerun.
    """
    res = st.session_state.get('optimization_results')
    if not res:
        return
    bp = migrate_legacy_pamrp_params(res.best_params)
    for k, v in bp.items():
        if isinstance(v, TradeDirection) or k == 'trade_direction':
            continue
        if k in st.session_state.params:
            st.session_state.params[k] = v
        wk = PARAM_TO_WIDGET_KEY.get(k)
        if wk:
            st.session_state[wk] = v
    if 'trade_direction_str' in bp:
        dm = {'long_only': 'Long Only', 'short_only': 'Short Only', 'both': 'Both'}
        trade_direction = dm.get(bp['trade_direction_str'], 'Long Only')
        st.session_state.params['trade_direction'] = trade_direction
        st.session_state.tdir = trade_direction
    st.session_state.capital = res.initial_capital
    st.session_state.commission = res.commission_pct
    st.session_state.slippage = getattr(res, 'slippage_pct', st.session_state.get('slippage', 0.0))
    st.session_state.backtest_results = None
    st.session_state._apply_success = True
