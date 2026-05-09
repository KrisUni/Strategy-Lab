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
from src.indicators.registry import INDICATOR_REGISTRY
from ui.state_migration import migrate_legacy_pamrp_params


# Strategy-level params keep their hand-coded widget keys (not registry-driven).
_STRATEGY_WIDGET_KEYS: Dict[str, str] = {
    'entry_operator':         'eop',
    'exit_operator':          'xop',
    'allow_same_bar_exit':    'same_bar_exit',
    'allow_same_bar_reversal':'same_bar_reversal',
    'entry_conflict_mode':    'ecm',
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
    return [
        spec.name
        for spec in INDICATOR_REGISTRY
        if params.get(spec.enable_param, False)
    ]


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
        # Strategy-level params keep hand-coded widget keys; indicator params use widget_{name}
        wk = _STRATEGY_WIDGET_KEYS.get(k, f"widget_{k}")
        if wk in st.session_state:
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
