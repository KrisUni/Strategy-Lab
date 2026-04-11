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
    'pamrp_enabled': 'pe', 'pamrp_entry_length': 'ple', 'pamrp_entry_long': 'pel', 'pamrp_entry_short': 'pes',
    'pamrp_exit_length': 'pxl_len', 'pamrp_exit_long': 'pxl_exit', 'pamrp_exit_short': 'pxs_exit',
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
    # New entry filters
    'bb_enabled': 'bbe', 'bb_length': 'bbl', 'bb_mult': 'bbm', 'bb_mode': 'bbmd',
    'stoch_entry_enabled': 'stke', 'stoch_entry_k_period': 'stkk', 'stoch_entry_d_period': 'stkd',
    'stoch_entry_slowing': 'stks', 'stoch_entry_oversold': 'stkos', 'stoch_entry_overbought': 'stkob',
    'cci_enabled': 'ccie', 'cci_length': 'ccil', 'cci_oversold': 'ccios', 'cci_overbought': 'cciob',
    'willr_enabled': 'wre', 'willr_length': 'wrl', 'willr_oversold': 'wros', 'willr_overbought': 'wrob',
    'obv_enabled': 'obve', 'obv_ma_length': 'obvm',
    'donchian_enabled': 'dche', 'donchian_length': 'dchl',
    'keltner_enabled': 'klte', 'keltner_length': 'kltl', 'keltner_mult': 'kltm',
    'psar_enabled': 'psre', 'psar_af_start': 'psrs', 'psar_af_step': 'psrst', 'psar_af_max': 'psrm',
    'ichi_enabled': 'iche', 'ichi_tenkan': 'icht', 'ichi_kijun': 'ichk', 'ichi_senkou_b': 'ichsb',
    'hull_enabled': 'hule', 'hull_length': 'hull',
    'trix_enabled': 'trxe', 'trix_length': 'trxl', 'trix_signal': 'trxs',
}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy param conversion
# ─────────────────────────────────────────────────────────────────────────────

def params_to_strategy(p: Dict[str, Any]) -> StrategyParams:
    """Convert the flat session-state params dict into a typed StrategyParams object."""
    p = migrate_legacy_pamrp_params(p)
    direction_map = {
        'Long Only': TradeDirection.LONG_ONLY,
        'Short Only': TradeDirection.SHORT_ONLY,
        'Both': TradeDirection.BOTH,
    }

    def parse_operator(value: Any, default: ConditionOperator) -> ConditionOperator:
        if isinstance(value, ConditionOperator):
            return value
        if isinstance(value, str):
            try:
                return ConditionOperator(value.lower())
            except ValueError:
                return default
        return default

    def parse_entry_conflict_mode(value: Any, default: EntryConflictMode) -> EntryConflictMode:
        if isinstance(value, EntryConflictMode):
            return value
        if isinstance(value, str):
            try:
                return EntryConflictMode(value.lower())
            except ValueError:
                return default
        return default

    return StrategyParams(
        trade_direction=direction_map.get(p.get('trade_direction', 'Long Only'), TradeDirection.LONG_ONLY),
        entry_operator=parse_operator(p.get('entry_operator', 'and'), ConditionOperator.AND),
        exit_operator=parse_operator(p.get('exit_operator', 'or'), ConditionOperator.OR),
        allow_same_bar_exit=p.get('allow_same_bar_exit', True),
        allow_same_bar_reversal=p.get('allow_same_bar_reversal', False),
        entry_conflict_mode=parse_entry_conflict_mode(p.get('entry_conflict_mode', 'skip'), EntryConflictMode.SKIP),
        position_size_pct=p.get('position_size_pct', 100.0),
        use_kelly=p.get('use_kelly', False), kelly_fraction=p.get('kelly_fraction', 0.5),
        pamrp_enabled=p.get('pamrp_enabled', True),
        pamrp_entry_length=p.get('pamrp_entry_length', 21),
        pamrp_entry_long=p.get('pamrp_entry_long', 20), pamrp_entry_short=p.get('pamrp_entry_short', 80),
        pamrp_exit_length=p.get('pamrp_exit_length', 21),
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
        bb_enabled=p.get('bb_enabled', False), bb_length=p.get('bb_length', 20),
        bb_mult=p.get('bb_mult', 2.0), bb_mode=p.get('bb_mode', 'squeeze'),
        stoch_entry_enabled=p.get('stoch_entry_enabled', False),
        stoch_entry_k_period=p.get('stoch_entry_k_period', 14),
        stoch_entry_d_period=p.get('stoch_entry_d_period', 3),
        stoch_entry_slowing=p.get('stoch_entry_slowing', 3),
        stoch_entry_oversold=p.get('stoch_entry_oversold', 20),
        stoch_entry_overbought=p.get('stoch_entry_overbought', 80),
        cci_enabled=p.get('cci_enabled', False), cci_length=p.get('cci_length', 20),
        cci_oversold=p.get('cci_oversold', -100), cci_overbought=p.get('cci_overbought', 100),
        willr_enabled=p.get('willr_enabled', False), willr_length=p.get('willr_length', 14),
        willr_oversold=p.get('willr_oversold', -80), willr_overbought=p.get('willr_overbought', -20),
        obv_enabled=p.get('obv_enabled', False), obv_ma_length=p.get('obv_ma_length', 20),
        donchian_enabled=p.get('donchian_enabled', False), donchian_length=p.get('donchian_length', 20),
        keltner_enabled=p.get('keltner_enabled', False), keltner_length=p.get('keltner_length', 20),
        keltner_mult=p.get('keltner_mult', 1.5),
        psar_enabled=p.get('psar_enabled', False), psar_af_start=p.get('psar_af_start', 0.02),
        psar_af_step=p.get('psar_af_step', 0.02), psar_af_max=p.get('psar_af_max', 0.2),
        ichi_enabled=p.get('ichi_enabled', False), ichi_tenkan=p.get('ichi_tenkan', 9),
        ichi_kijun=p.get('ichi_kijun', 26), ichi_senkou_b=p.get('ichi_senkou_b', 52),
        hull_enabled=p.get('hull_enabled', False), hull_length=p.get('hull_length', 20),
        trix_enabled=p.get('trix_enabled', False), trix_length=p.get('trix_length', 15),
        trix_signal=p.get('trix_signal', 9),
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
        'bb_enabled': 'Bollinger Bands', 'stoch_entry_enabled': 'Stochastic',
        'cci_enabled': 'CCI', 'willr_enabled': 'Williams %R', 'obv_enabled': 'OBV',
        'donchian_enabled': 'Donchian', 'keltner_enabled': 'Keltner',
        'psar_enabled': 'Parabolic SAR', 'ichi_enabled': 'Ichimoku',
        'hull_enabled': 'Hull MA', 'trix_enabled': 'TRIX',
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
