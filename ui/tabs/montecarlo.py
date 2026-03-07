"""
ui/tabs/montecarlo.py
=====================
Renders the Monte Carlo tab content (tabs[3]).
"""

import streamlit as st

from src.montecarlo import (
    run_monte_carlo,
    in_sample_permutation_test,
    create_permutation_chart,
)
from src.backtest import BacktestEngine
from ui.charts import (
    create_mc_confidence_chart,
    create_mc_histogram,
)


def render_montecarlo_tab() -> None:

    # ── Section 1: Monte Carlo Simulation ────────────────────────────────────
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

    # ── Section 2: In-Sample Permutation Test ─────────────────────────────────
    st.divider()
    st.markdown("### 🔬 In-Sample Permutation Test")
    st.caption(
        "Shuffles interior bars of the price series (first & last bars fixed to preserve trend), "
        "runs your strategy on each permuted series, and measures whether your edge is statistically "
        "distinguishable from random bar ordering."
    )

    col1, col2, col3 = st.columns(3)
    perm_metric = col1.selectbox(
        "Test Metric",
        ["profit_factor", "sharpe", "net_profit", "win_rate", "calmar", "expectancy", "total_return"],
        key="perm_metric",
    )
    n_perms = col2.slider(
        "Permutations", 100, 2000, 500, 100, key="n_perms",
        help="More permutations = more precise p-value. 500 is a good default.",
    )
    col3.markdown("<br>", unsafe_allow_html=True)
    run_perm = col3.button("🔬 Run Permutation Test", use_container_width=True, key="run_perm")

    if run_perm:
        r = st.session_state.get("backtest_results")
        df = st.session_state.get("df")

        if r is None or df is None:
            st.warning("Run a backtest first.")
        elif len(r.trades) < 5:
            st.warning("Need at least 5 trades for a meaningful permutation test.")
        else:
            with st.spinner(f"Running {n_perms} permutations... (this may take a moment)"):

                def _run_bt(permuted_df, params):
                    engine = BacktestEngine(
                        df=permuted_df,
                        params=params,
                        initial_capital=st.session_state.capital,
                        commission_pct=st.session_state.commission,
                    )
                    return engine.run()

                perm_result = in_sample_permutation_test(
                    df=df,
                    run_backtest_fn=_run_bt,
                    params=st.session_state.params,
                    metric=perm_metric,
                    n_permutations=n_perms,
                    real_results=r,
                    seed=42,
                )
                st.session_state._perm_result = perm_result

    perm_result = st.session_state.get("_perm_result")
    if perm_result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "p-value", f"{perm_result.p_value:.4f}",
            delta="Significant ✅" if perm_result.p_value < 0.05 else "Not significant ⚠️",
            delta_color="normal" if perm_result.p_value < 0.05 else "off",
        )
        c2.metric("Real Strategy", f"{perm_result.real_metric:.4f}")
        c3.metric("Null Median",   f"{perm_result.pct_50:.4f}")
        c4.metric("Beat Permutations", f"{perm_result.beat_pct:.1f}%")

        st.plotly_chart(create_permutation_chart(perm_result), use_container_width=True)

        with st.expander("📖 How to interpret this"):
            st.markdown("""
            **p-value** = fraction of random permutations that matched or beat your real strategy.

            | p-value | Interpretation |
            |---------|---------------|
            | < 0.01  | Very strong evidence of real edge |
            | 0.01 – 0.05 | Statistically significant edge |
            | 0.05 – 0.10 | Weak / borderline — treat with caution |
            | > 0.10  | No detectable edge over random ordering |

            **Why fix first and last bars?**
            Anchoring endpoints preserves the net price move (trend/drift).
            This prevents the test from rewarding a strategy simply for being
            long-biased in a bull market — the same overall trend applies to
            all permuted series.

            **Limitations:**
            This is an *in-sample* test only. A significant p-value confirms
            the strategy exploits temporal structure, but doesn't validate
            out-of-sample performance. Always combine with walk-forward analysis.
            """)