"""
ui/tabs/montecarlo.py
=====================
Renders the Monte Carlo tab content (tabs[3]).
"""

import streamlit as st

from src.montecarlo import run_monte_carlo
from ui.charts import (
    create_mc_confidence_chart,
    create_mc_histogram,
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
    if not mc:
        return

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
