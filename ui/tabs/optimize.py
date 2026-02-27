"""
ui/tabs/optimize.py
===================
Renders the Optimize tab content (tabs[1]).
"""

import pandas as pd
import streamlit as st

from src.optimize import optimize_strategy
from src.strategy import TradeDirection
from ui.helpers import get_active_filters_display, apply_best_params_callback
from ui.charts import (
    create_walkforward_chart,
    create_stitched_equity_chart,
    create_optimization_chart,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pinnable parameter registry
# ─────────────────────────────────────────────────────────────────────────────

_PINNABLE = {
    'pamrp_enabled': [('pamrp_length', 'Length'), ('pamrp_entry_long', 'Entry Long'),
        ('pamrp_entry_short', 'Entry Short'), ('pamrp_exit_long', 'Exit Long'), ('pamrp_exit_short', 'Exit Short')],
    'bbwp_enabled': [('bbwp_length', 'Length'), ('bbwp_lookback', 'Lookback'), ('bbwp_sma_length', 'SMA Length'),
        ('bbwp_ma_filter', 'MA Filter'), ('bbwp_threshold_long', 'Thresh Long'), ('bbwp_threshold_short', 'Thresh Short')],
    'adx_enabled': [('adx_length', 'Length'), ('adx_smoothing', 'Smoothing'), ('adx_threshold', 'Threshold')],
    'ma_trend_enabled': [('ma_type', 'Type'), ('ma_fast_length', 'Fast'), ('ma_slow_length', 'Slow')],
    'rsi_enabled': [('rsi_length', 'Length'), ('rsi_oversold', 'Oversold'), ('rsi_overbought', 'Overbought')],
    'volume_enabled': [('volume_ma_length', 'MA Length'), ('volume_multiplier', 'Multiplier')],
    'supertrend_enabled': [('supertrend_period', 'Period'), ('supertrend_multiplier', 'Multiplier')],
    'macd_enabled': [('macd_fast', 'Fast'), ('macd_slow', 'Slow'), ('macd_signal', 'Signal')],
    'stop_loss_enabled': [('stop_loss_pct_long', '% Long'), ('stop_loss_pct_short', '% Short')],
    'take_profit_enabled': [('take_profit_pct_long', '% Long'), ('take_profit_pct_short', '% Short')],
    'trailing_stop_enabled': [('trailing_stop_pct', 'Trail %')],
    'atr_trailing_enabled': [('atr_length', 'Length'), ('atr_multiplier', 'Multiplier')],
    'time_exit_enabled': [('time_exit_bars', 'Max Bars')],
    'ma_exit_enabled': [('ma_exit_fast', 'Fast'), ('ma_exit_slow', 'Slow')],
    'bbwp_exit_enabled': [('bbwp_exit_threshold', 'Threshold')],
}

_INDICATOR_LABELS = {
    'pamrp_enabled': 'PAMRP', 'bbwp_enabled': 'BBWP', 'adx_enabled': 'ADX',
    'ma_trend_enabled': 'MA Trend', 'rsi_enabled': 'RSI', 'volume_enabled': 'Volume',
    'supertrend_enabled': 'Supertrend', 'macd_enabled': 'MACD',
    'stop_loss_enabled': 'Stop Loss', 'take_profit_enabled': 'Take Profit',
    'trailing_stop_enabled': 'Trailing Stop', 'atr_trailing_enabled': 'ATR Trail',
    'time_exit_enabled': 'Time Exit', 'ma_exit_enabled': 'MA Exit', 'bbwp_exit_enabled': 'BBWP Exit',
}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_optimize_tab() -> None:
    st.markdown("### 🎯 Optimization")

    c1, c2, c3, c4 = st.columns(4)
    opt_metric = c1.selectbox("Metric", ["profit_factor", "sharpe_ratio", "total_return_pct", "sortino_ratio"])
    opt_dir = c2.selectbox("Dir", ["long_only", "short_only", "both"])
    opt_trials = c3.slider("Trials", 50, 1500, 500)
    opt_min = c4.slider("Min Trades", 5, 30, 10)

    c1, c2, c3 = st.columns(3)
    train_pct = c1.slider("Train %", 50, 90, 70)
    use_wf = c2.toggle("Walk-Forward", True)
    n_folds = c3.slider("Folds", 3, 10, 5) if use_wf else 5

    window_type = 'rolling'
    train_window_bars = None
    if use_wf:
        c1, c2, c3 = st.columns(3)
        window_type = c1.selectbox("Window Type", ["rolling", "anchored"], index=0,
            help="Rolling: fixed-size sliding window. Anchored: expanding from bar 0.")
        if window_type == 'rolling' and st.session_state.df is not None:
            default_bars = int(len(st.session_state.df) * train_pct / 100)
            train_window_bars = c2.slider("Train Window (bars)", min_value=50,
                max_value=max(51, len(st.session_state.df) - 20),
                value=min(default_bars, max(50, len(st.session_state.df) - 20)))
        elif window_type == 'rolling':
            c2.caption("Load data to configure window size.")

    _render_pin_expander()

    active_display = get_active_filters_display(st.session_state.params)
    if active_display:
        st.caption(f"🔍 Active: **{', '.join(active_display)}**")

    if st.button("🎯 Optimize", type="primary", use_container_width=True):
        if st.session_state.df is None:
            st.warning("Load data first!")
        else:
            ef = {k: v for k, v in st.session_state.params.items() if k.endswith('_enabled')}
            pinned_dict = {k: st.session_state.params[k]
                           for k in st.session_state.pinned_params
                           if k in st.session_state.params}
            with st.spinner("Optimizing..."):
                res = optimize_strategy(
                    df=st.session_state.df.copy(), enabled_filters=ef,
                    metric=opt_metric, n_trials=opt_trials, min_trades=opt_min,
                    initial_capital=st.session_state.capital, commission_pct=st.session_state.commission,
                    trade_direction=opt_dir, train_pct=train_pct / 100, use_walkforward=use_wf,
                    n_folds=n_folds, window_type=window_type, train_window_bars=train_window_bars,
                    show_progress=False, pinned_params=pinned_dict if pinned_dict else None)
                st.session_state.optimization_results = res
                st.success("✅ Done!")

    _render_results()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_pin_expander() -> None:
    enabled_pinnable = [
        (ind_key, _INDICATOR_LABELS.get(ind_key, ind_key), params_list)
        for ind_key, params_list in _PINNABLE.items()
        if st.session_state.params.get(ind_key, False)
    ]
    pinned_set: set = st.session_state.pinned_params

    if not enabled_pinnable:
        return

    with st.expander("🔒 Pin Parameters (hold fixed during optimization)", expanded=False):
        st.caption("Checked parameters are fixed at their current sidebar values and excluded from the search space.")
        if st.button("Clear all pins", key="_clear_pins"):
            st.session_state.pinned_params = set()
            st.rerun()
        for ind_key, ind_label, param_list in enabled_pinnable:
            st.markdown(f"**{ind_label}**")
            cols = st.columns(min(len(param_list), 3))
            for idx, (pname, plabel) in enumerate(param_list):
                current_val = st.session_state.params.get(pname, '—')
                col = cols[idx % len(cols)]
                checked = col.checkbox(f"{plabel}  `{current_val}`", value=(pname in pinned_set), key=f"_pin_{pname}")
                if checked:
                    pinned_set.add(pname)
                else:
                    pinned_set.discard(pname)

    st.session_state.pinned_params = pinned_set

    if pinned_set:
        pinned_names = ', '.join(
            next((pl for pk, pl in p_list if pk == k), k)
            for ind_key, _, p_list in enabled_pinnable
            for k in pinned_set if any(pk == k for pk, _ in p_list)
        )
        st.caption(f"🔒 Pinned: **{pinned_names}**")


def _render_results() -> None:
    if not st.session_state.optimization_results:
        return

    res = st.session_state.optimization_results
    ml = res.metric.replace('_', ' ').title()

    if res.warnings:
        with st.expander(f"⚠️ {len(res.warnings)} Robustness Warning(s)", expanded=True):
            for w in res.warnings:
                st.warning(w)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Train ({ml})", f"{res.train_value:.4f}")
    c2.metric(f"OOS ({ml})", f"{res.test_value:.4f}")
    if res.train_value != 0 and res.efficiency_ratio != 0.0:
        if res.efficiency_ratio < 0.5:
            c3.metric("Efficiency (OOS/IS)", f"{res.efficiency_ratio:.2f}", delta="⚠ Overfit risk", delta_color="inverse")
        else:
            c3.metric("Efficiency (OOS/IS)", f"{res.efficiency_ratio:.2f}", delta="✓ Acceptable", delta_color="normal")
    else:
        c3.metric("Efficiency (OOS/IS)", "—")
    if res.train_value > 0:
        deg = ((res.train_value - res.test_value) / res.train_value) * 100
        c4.metric("IS→OOS Degradation", f"{deg:.1f}%", delta_color="normal" if deg < 30 else "inverse")

    if res.walkforward_folds:
        st.markdown("#### Walk-Forward Fold Performance")
        st.caption(f"Window: **{res.window_type}** | Folds: **{len(res.walkforward_folds)}**")
        st.plotly_chart(create_walkforward_chart(res.walkforward_folds),
                        use_container_width=True, config={'displayModeBar': False})
        if res.stitched_equity is not None and len(res.stitched_equity) > 0:
            st.markdown("#### 📈 Stitched OOS Equity Curve")
            st.caption("Each segment is a genuine out-of-sample period — the only unbiased performance view for WFO.")
            st.plotly_chart(create_stitched_equity_chart(res.stitched_equity), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Best Params")
        pp = res.pinned_params or {}
        rows = []
        for k, v in res.best_params.items():
            if k.startswith('trade_direction') or isinstance(v, TradeDirection):
                continue
            rows.append({"Parameter": f"{'🔒 ' if k in pp else ''}{k}", "Value": str(v)[:25]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=280)
        if pp:
            st.caption(f"🔒 = fixed at sidebar value ({len(pp)} pinned)")
    with c2:
        br = res.full_data_results
        st.markdown("### Full-Data Performance")
        st.caption("⚠ Includes in-sample periods. For WFO, refer to the stitched OOS chart.")
        st.dataframe(pd.DataFrame([
            {"Metric": "Return", "Value": f"{br.total_return_pct:.2f}%"},
            {"Metric": "CAGR", "Value": f"{br.cagr:.2f}%"},
            {"Metric": "Sharpe", "Value": f"{br.sharpe_ratio:.3f}"},
            {"Metric": "Max DD", "Value": f"{br.max_drawdown_pct:.2f}%"},
            {"Metric": "Trades", "Value": str(br.num_trades)},
        ]), use_container_width=True, hide_index=True)

    if not res.all_trials.empty:
        st.plotly_chart(create_optimization_chart(res.all_trials, res.metric), use_container_width=True)

    st.button("📋 Apply Best Params", on_click=apply_best_params_callback, use_container_width=True)
    if st.session_state.pop('_apply_success', False):
        st.success(f"✅ Applied! Capital: ${st.session_state.capital:,.0f} | Commission: {st.session_state.commission}%")
