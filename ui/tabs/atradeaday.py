"""
ui/tabs/atradeaday.py
=====================
Renders the A Trade A Day tab (tabs[8]).

Strategy: one trade per day based on the 9:30 AM 5-min candle high/low,
FVG breakout on 1-min, engulfing entry, 3:1 RR.
"""

import streamlit as st
from src.strategy.atradeaday import run_atradeaday, ATradeADayParams


def render_atradeaday_tab() -> None:

    # ── Config row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    risk_per_trade = c1.number_input(
        "Risk per trade ($)", 10, 10000,
        int(st.session_state.capital * 0.01), 10
    )
    rr_ratio = c2.number_input(
        "R:R Ratio", 1.0, 10.0, 3.0, 0.5
    )
    entry_time = c3.selectbox(
        "Entry candle", ["09:30", "09:35", "10:00"], index=0
    )
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Run", type="primary", use_container_width=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    if run:
        if st.session_state.df is None:
            st.warning("Load data first!")
        else:
            params = ATradeADayParams(
                rr_ratio=rr_ratio,
                risk_per_trade=risk_per_trade,
                entry_time=entry_time,
            )
            with st.spinner("Running A Trade A Day..."):
                results = run_atradeaday(st.session_state.df.copy(), params)
            st.session_state.atradeaday_results = results
            st.success(f"✅ {results['num_trades']} trades")

    # ── Results ───────────────────────────────────────────────────────────────
    r = st.session_state.atradeaday_results

    if r:
        c1, c2, c3 = st.columns(3)
        c1.metric("Return", f"{r['total_return_pct']:.2f}%")
        c2.metric("Win %", f"{r['win_rate']:.1f}%")
        c3.metric("Profit Factor", f"{r['profit_factor']:.2f}")

        if not r["trades"]:
            st.info("No trades generated yet — backend logic coming in next phase.")