"""
ui/tabs/montecarlo.py
=====================
Renders the Monte Carlo tab content (tabs[3]).
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.montecarlo import run_monte_carlo
from ui.charts import (
    create_mc_confidence_chart,
    create_mc_histogram,
    _chart_layout,
)


def render_montecarlo_tab() -> None:
    st.markdown("### 🎲 Monte Carlo Simulation")

    c1, c2, c3 = st.columns(3)
    mc_method = c1.selectbox("Method", ["Trade Shuffle", "Return Bootstrap", "Noise Injection"])
    n_sims = c2.slider("Simulations", 100, 5000, 1000)
    ruin_pct = c3.slider("Ruin Threshold %", 10, 90, 50)

    method_map = {
        "Trade Shuffle": "trade_shuffle",
        "Return Bootstrap": "return_bootstrap",
        "Noise Injection": "noise_injection",
    }
    method_key = method_map[mc_method]
    extra_kwargs = {}
    if method_key == 'noise_injection':
        extra_kwargs['noise_pct'] = st.slider("Noise %", 5.0, 50.0, 20.0, 5.0)
    if method_key == 'return_bootstrap':
        extra_kwargs['block_size'] = st.slider("Block Size", 1, 20, 5)

    if st.button("🎲 Run Monte Carlo", use_container_width=True):
        r = st.session_state.backtest_results
        if r and r.trades:
            with st.spinner(f"Running {n_sims} {mc_method} simulations..."):
                mc = run_monte_carlo(
                    trades=r.trades, equity_curve=r.equity_curve,
                    method=method_key, n_simulations=n_sims,
                    initial_capital=st.session_state.capital,
                    ruin_pct=ruin_pct, bars_per_year=r.bars_per_year,
                    **extra_kwargs,
                )
                if mc:
                    st.session_state._mc_result = mc
        else:
            st.warning("Run backtest first!")

    mc = st.session_state.get('_mc_result')
    if mc:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk of Ruin", f"{mc.risk_of_ruin:.1%}")
        c2.metric("5th Pctl", f"${mc.equity_percentiles['5%']:,.0f}")
        c3.metric("Median", f"${mc.equity_percentiles['50%']:,.0f}")
        c4.metric("95th Pctl", f"${mc.equity_percentiles['95%']:,.0f}")

        st.markdown("### 📈 Equity Confidence Bands")
        st.plotly_chart(create_mc_confidence_chart(mc), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 💰 Final Equity Distribution")
            st.plotly_chart(create_mc_histogram(mc.final_equities, xaxis_title='Final Equity $'),
                            use_container_width=True)
        with c2:
            st.markdown("### 📉 Max Drawdown Distribution")
            st.plotly_chart(create_mc_histogram(mc.max_drawdowns, xaxis_title='Max DD %'),
                            use_container_width=True)

        if mc.sharpe_distribution is not None:
            st.markdown("### 📊 Sharpe Ratio Distribution")
            st.plotly_chart(create_mc_histogram(mc.sharpe_distribution, xaxis_title='Sharpe'),
                            use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("5th Pctl DD", f"{mc.dd_percentiles['5%']:.1f}%")
        c2.metric("Median DD", f"{mc.dd_percentiles['50%']:.1f}%")
        c3.metric("95th Pctl DD", f"{mc.dd_percentiles['95%']:.1f}%")

    # ═════════════════════════════════════════════════════════════════════════
    # IN-SAMPLE PERMUTATION TEST
    # ═════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("### 🧪 In-Sample Permutation Test")
    st.caption(
        "**Null hypothesis:** your strategy has no real edge — the optimised performance "
        "is indistinguishable from what you'd get by optimising on randomised (shuffled) data. "
        "We try to **disprove** this by showing the real-data metric sits above the permutation cloud."
    )

    c1, c2, c3 = st.columns(3)
    perm_metric = c1.selectbox(
        "Test Metric",
        ["profit_factor", "sharpe_ratio", "total_return_pct", "sortino_ratio"],
        index=0,
        key="perm_metric",
    )
    n_perms = c2.slider("Permutations", 20, 500, 100, key="perm_n")
    perm_trials = c3.slider("Trials per Opt", 20, 200, 50, key="perm_trials",
        help="Optuna trials per optimization run. Lower = faster but noisier.")

    c1, c2 = st.columns(2)
    perm_dir = c1.selectbox("Direction", ["long_only", "short_only", "both"], key="perm_dir")
    perm_min = c2.slider("Min Trades", 3, 20, 5, key="perm_min_trades")

    if st.button("🧪 Run Permutation Test", use_container_width=True):
        if st.session_state.df is None:
            st.warning("Load data first!")
        else:
            from src.permutation import run_permutation_test
            from ui.helpers import get_active_filters_display

            p = st.session_state.params
            enabled_filters = {k: v for k, v in p.items() if k.endswith('_enabled') and isinstance(v, bool)}

            # Collect pinned params from session state
            pinned = {}
            pinned_set = st.session_state.get('pinned_params', set())
            if pinned_set:
                for pk in pinned_set:
                    if pk in p:
                        pinned[pk] = p[pk]

            progress_bar = st.progress(0, text="Optimizing on real data...")
            status_text = st.empty()

            def _progress(current, total):
                pct = current / total
                progress_bar.progress(pct, text=f"Permutation {current}/{total}...")

            with st.spinner("Running permutation test..."):
                perm_result = run_permutation_test(
                    df=st.session_state.df.copy(),
                    enabled_filters=enabled_filters,
                    metric=perm_metric,
                    n_permutations=n_perms,
                    n_trials=perm_trials,
                    min_trades=perm_min,
                    initial_capital=st.session_state.capital,
                    commission_pct=st.session_state.commission,
                    trade_direction=perm_dir,
                    train_pct=0.7,
                    progress_callback=_progress,
                    pinned_params=pinned if pinned else None,
                )

            progress_bar.empty()
            status_text.empty()

            if perm_result:
                st.session_state._perm_result = perm_result

    # ── Display permutation results ───────────────────────────────────────
    perm = st.session_state.get('_perm_result')
    if perm:
        _render_permutation_results(perm)


# ─────────────────────────────────────────────────────────────────────────────
# Permutation test display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_permutation_results(perm) -> None:
    """Render the full permutation test results section."""
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Real {perm.metric_name}", f"{perm.real_metric:.3f}")
    c2.metric("Median Permuted", f"{np.median(perm.permuted_metrics):.3f}")
    c3.metric("p-value", f"{perm.p_value:.3f}")
    c4.metric("Permutations", perm.n_permutations)

    # Interpretation
    if perm.p_value < 0.01:
        st.success(
            f"✅ **Strong evidence of real edge.** p = {perm.p_value:.3f}. "
            f"Your strategy's {perm.metric_name} ({perm.real_metric:.3f}) is very unlikely "
            f"to be the result of optimisation on noise."
        )
    elif perm.p_value < 0.05:
        st.success(
            f"✅ **Statistically significant.** p = {perm.p_value:.3f}. "
            f"Your strategy's edge appears real at the 5% significance level."
        )
    elif perm.p_value < 0.10:
        st.warning(
            f"⚠️ **Marginal.** p = {perm.p_value:.3f}. "
            f"Some evidence of an edge, but not statistically significant at the 5% level. "
            f"Consider more permutations or a larger dataset."
        )
    else:
        st.error(
            f"❌ **No significant edge detected.** p = {perm.p_value:.3f}. "
            f"The strategy's performance cannot be distinguished from what you'd get "
            f"by optimising on random data. The apparent edge is likely overfitting."
        )

    # ── Equity curve chart (like the screenshot) ──────────────────────────
    st.markdown("#### Permutation Equity Curves")
    st.plotly_chart(
        _create_permutation_equity_chart(perm),
        use_container_width=True,
    )

    # ── Metric distribution histogram ─────────────────────────────────────
    st.markdown(f"#### {perm.metric_name} Distribution")
    st.plotly_chart(
        _create_permutation_histogram(perm),
        use_container_width=True,
    )

    # ── Extra stats ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Real Trades", perm.real_num_trades)
    c2.metric("Avg Permuted Trades", f"{perm.avg_permuted_trades:.0f}")
    perm_std = float(np.std(perm.permuted_metrics))
    if perm_std > 0:
        z = (perm.real_metric - np.mean(perm.permuted_metrics)) / perm_std
        c3.metric("Z-score", f"{z:.2f}")
    else:
        c3.metric("Z-score", "N/A")


def _create_permutation_equity_chart(perm) -> go.Figure:
    """
    Create the permutation equity chart — gray cloud of permuted curves,
    red line for real optimized (like the screenshot).
    """
    fig = go.Figure()

    # Plot permuted equity curves (gray, semi-transparent)
    max_curves = min(len(perm.permuted_equities), 200)  # cap for rendering
    step = max(1, len(perm.permuted_equities) // max_curves)
    for i in range(0, len(perm.permuted_equities), step):
        eq = perm.permuted_equities[i]
        x = list(range(len(eq)))
        fig.add_trace(go.Scatter(
            x=x, y=eq, mode='lines',
            line=dict(color='rgba(180,180,180,0.12)', width=0.8),
            showlegend=(i == 0),
            name='Permutation Optimized' if i == 0 else None,
            hoverinfo='skip',
        ))

    # Highlight the best permuted curve (white)
    if perm.permuted_equities:
        best_idx = int(np.argmax(perm.permuted_metrics))
        best_eq = perm.permuted_equities[best_idx]
        fig.add_trace(go.Scatter(
            x=list(range(len(best_eq))), y=best_eq, mode='lines',
            line=dict(color='rgba(255,255,255,0.7)', width=1.5),
            name=f'Best Permuted ({perm.metric_name}={perm.permuted_metrics[best_idx]:.2f})',
            showlegend=True,
        ))

    # Real equity curve (red, bold)
    x_real = list(range(len(perm.real_equity)))
    fig.add_trace(go.Scatter(
        x=x_real, y=perm.real_equity, mode='lines',
        line=dict(color='#ef4444', width=2.5),
        name=f'Real Optimized ({perm.metric_name}={perm.real_metric:.2f})',
        showlegend=True,
    ))

    fig.update_layout(
        **_chart_layout(
            420,
            showlegend=True,
            legend=dict(orientation='h', y=1.12, font=dict(size=10)),
            plot_bgcolor='rgba(5,5,5,0.95)',
        ),
        xaxis_title='Bar',
        yaxis_title='Equity ($)',
    )
    return fig


def _create_permutation_histogram(perm) -> go.Figure:
    """Histogram of permuted metrics with a vertical line for real metric."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=perm.permuted_metrics,
        nbinsx=30,
        marker_color='rgba(100,116,139,0.6)',
        marker_line=dict(color='rgba(100,116,139,0.9)', width=1),
        name='Permuted',
    ))

    # Vertical line for real metric
    fig.add_vline(
        x=perm.real_metric,
        line_color='#ef4444',
        line_width=3,
        annotation_text=f"Real: {perm.real_metric:.3f}",
        annotation_position="top right",
        annotation_font_color='#ef4444',
        annotation_font_size=12,
    )

    # p-value annotation
    fig.add_annotation(
        x=0.02, y=0.95,
        xref='paper', yref='paper',
        text=f"p = {perm.p_value:.3f}",
        showarrow=False,
        font=dict(size=14, color='white'),
        bgcolor='rgba(0,0,0,0.5)',
        borderpad=4,
    )

    fig.update_layout(
        **_chart_layout(280),
        xaxis_title=perm.metric_name,
        yaxis_title='Count',
        bargap=0.05,
    )
    return fig