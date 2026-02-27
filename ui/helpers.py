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

from src.strategy import StrategyParams, TradeDirection


# ─────────────────────────────────────────────────────────────────────────────
# Widget-key mapping (param name → short Streamlit widget key)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_TO_WIDGET_KEY: Dict[str, str] = {
    'pamrp_enabled': 'pe', 'pamrp_length': 'pl', 'pamrp_entry_long': 'pel', 'pamrp_entry_short': 'pes',
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
    """Convert the flat session-state params dict into a typed StrategyParams object."""
    direction_map = {
        'Long Only': TradeDirection.LONG_ONLY,
        'Short Only': TradeDirection.SHORT_ONLY,
        'Both': TradeDirection.BOTH,
    }
    return StrategyParams(
        trade_direction=direction_map.get(p.get('trade_direction', 'Long Only'), TradeDirection.LONG_ONLY),
        position_size_pct=p.get('position_size_pct', 100.0),
        use_kelly=p.get('use_kelly', False), kelly_fraction=p.get('kelly_fraction', 0.5),
        pamrp_enabled=p.get('pamrp_enabled', True), pamrp_length=p.get('pamrp_length', 21),
        pamrp_entry_long=p.get('pamrp_entry_long', 20), pamrp_entry_short=p.get('pamrp_entry_short', 80),
        pamrp_exit_long=p.get('pamrp_exit_long', 70), pamrp_exit_short=p.get('pamrp_exit_short', 30),
        bbwp_enabled=p.get('bbwp_enabled', True), bbwp_length=p.get('bbwp_length', 13),
        bbwp_lookback=p.get('bbwp_lookback', 252), bbwp_sma_length=p.get('bbwp_sma_length', 5),
        bbwp_threshold_long=p.get('bbwp_threshold_long', 50), bbwp_threshold_short=p.get('bbwp_threshold_short', 50),
        bbwp_ma_filter=p.get('bbwp_ma_filter', 'disabled'),
        adx_enabled=p.get('adx_enabled', False), adx_length=p.get('adx_length', 14),
        adx_smoothing=p.get('adx_smoothing', 14), adx_threshold=p.get('adx_threshold', 20),
        ma_trend_enabled=p.get('ma_trend_enabled', False), ma_fast_length=p.get('ma_fast_length', 50),
        ma_slow_length=p.get('ma_slow_length', 200), ma_type=p.get('ma_type', 'sma'),
        rsi_enabled=p.get('rsi_enabled', False), rsi_length=p.get('rsi_length', 14),
        rsi_oversold=p.get('rsi_oversold', 30), rsi_overbought=p.get('rsi_overbought', 70),
        volume_enabled=p.get('volume_enabled', False), volume_ma_length=p.get('volume_ma_length', 20),
        volume_multiplier=p.get('volume_multiplier', 1.0),
        supertrend_enabled=p.get('supertrend_enabled', False), supertrend_period=p.get('supertrend_period', 10),
        supertrend_multiplier=p.get('supertrend_multiplier', 3.0),
        vwap_enabled=p.get('vwap_enabled', False),
        macd_enabled=p.get('macd_enabled', False), macd_fast=p.get('macd_fast', 12),
        macd_slow=p.get('macd_slow', 26), macd_signal=p.get('macd_signal', 9),
        stop_loss_enabled=p.get('stop_loss_enabled', True),
        stop_loss_pct_long=p.get('stop_loss_pct_long', 3.0), stop_loss_pct_short=p.get('stop_loss_pct_short', 3.0),
        take_profit_enabled=p.get('take_profit_enabled', False),
        take_profit_pct_long=p.get('take_profit_pct_long', 5.0), take_profit_pct_short=p.get('take_profit_pct_short', 5.0),
        trailing_stop_enabled=p.get('trailing_stop_enabled', False), trailing_stop_pct=p.get('trailing_stop_pct', 2.0),
        atr_trailing_enabled=p.get('atr_trailing_enabled', False), atr_length=p.get('atr_length', 14),
        atr_multiplier=p.get('atr_multiplier', 2.0),
        pamrp_exit_enabled=p.get('pamrp_exit_enabled', True),
        stoch_rsi_exit_enabled=p.get('stoch_rsi_exit_enabled', False),
        stoch_rsi_length=p.get('stoch_rsi_length', 14), stoch_rsi_k=p.get('stoch_rsi_k', 3),
        stoch_rsi_d=p.get('stoch_rsi_d', 3), stoch_rsi_overbought=p.get('stoch_rsi_overbought', 80),
        stoch_rsi_oversold=p.get('stoch_rsi_oversold', 20),
        time_exit_enabled=p.get('time_exit_enabled', False),
        time_exit_bars_long=p.get('time_exit_bars', 20), time_exit_bars_short=p.get('time_exit_bars', 20),
        ma_exit_enabled=p.get('ma_exit_enabled', False), ma_exit_fast=p.get('ma_exit_fast', 10),
        ma_exit_slow=p.get('ma_exit_slow', 20),
        bbwp_exit_enabled=p.get('bbwp_exit_enabled', False), bbwp_exit_threshold=p.get('bbwp_exit_threshold', 80),
    )


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
    bp = res.best_params
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
        st.session_state.params['trade_direction'] = dm.get(bp['trade_direction_str'], 'Long Only')
    st.session_state.capital = res.initial_capital
    st.session_state.commission = res.commission_pct
    st.session_state.backtest_results = None
    st.session_state._apply_success = True
