"""
ui/tabs/atradeaday.py
=====================
Renders the A Trade A Day tab (tabs[8]).

This tab manages its OWN data fetch because the strategy requires 5-min bars.
The sidebar's selected interval is irrelevant here — we always fetch 5m data.

Layout mirrors backtest.py exactly so the UX is consistent.
"""

import streamlit as st

from src.data import fetch_yfinance
from src.strategy.atradeaday import run_atradeaday, ATradeADayParams
from ui.charts import create_equity_chart, create_price_chart_with_trades


def render_atradeaday_tab() -> None:

    # ── Info banner ───────────────────────────────────────────────────────────
    with st.expander("ℹ️ How this strategy works", expanded=False):
        st.markdown("""
        **A Trade A Day** — one setup, once a day, full rules:

        1. **Mark the levels** — At `entry_time` (default 09:30), the first 5-min candle
           closes. Its high and low are the only levels that matter for the day.
        2. **FVG breakout** — Wait for a 3-candle Fair Value Gap that breaks through
           one of those levels. The middle candle must cross the level with momentum,
           leaving a gap between the 1st and 3rd candle's wicks.
        3. **Retest** — Price pulls back into the FVG gap zone.
        4. **Engulfing entry** — A candle completely engulfs the pullback candle.
           Enter at its close.
        5. **Exit** — SL at the low/high of the first FVG candle. TP at `RR × risk`.
           One trade. Walk away.
        """)

    st.divider()

    # ── Config row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    ticker = c1.text_input(
        "Ticker",
        value=st.session_state.get("ticker", "SPY"),
        key="atad_ticker",
    )
    risk_per_trade = c2.number_input(
        "Risk / trade ($)", min_value=10, max_value=50000,
        value=100, step=10,
        key="atad_risk",
    )
    rr_ratio = c3.number_input(
        "R:R ratio", min_value=1.0, max_value=10.0,
        value=3.0, step=0.5,
        key="atad_rr",
    )
    entry_time = c4.selectbox(
        "Entry candle",
        options=["09:30", "09:35", "10:00"],
        index=0,
        key="atad_entry_time",
    )
    commission = c5.number_input(
        "Comm %", min_value=0.0, max_value=1.0,
        value=0.05, step=0.01,
        key="atad_commission",
    )

    # ── Date range row ────────────────────────────────────────────────────────
    dc1, dc2, dc3 = st.columns([2, 2, 1])

    import datetime
    default_end   = datetime.date.today()
    default_start = default_end - datetime.timedelta(days=55)  # 5m data: 60-day yfinance limit

    start_date = dc1.date_input(
        "Start date",
        value=default_start,
        key="atad_start",
    )
    end_date = dc2.date_input(
        "End date",
        value=default_end,
        key="atad_end",
    )

    with dc3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button(
            "🚀 Run",
            type="primary",
            use_container_width=True,
            key="atradeaday_run",
        )

    st.caption("⚠️ Yahoo Finance limits 5-min data to the last 60 calendar days.")

    # ── Run ───────────────────────────────────────────────────────────────────
    if run:
        try:
            with st.spinner(f"Fetching 5-min data for {ticker}..."):
                df = fetch_yfinance(
                    symbol   = ticker.upper().strip(),
                    start    = str(start_date),
                    end      = str(end_date),
                    interval = "5m",
                )

            params = ATradeADayParams(
                rr_ratio       = rr_ratio,
                risk_per_trade = risk_per_trade,
                entry_time     = entry_time,
                commission_pct = commission,
            )

            with st.spinner("Running A Trade A Day strategy..."):
                results = run_atradeaday(df, params)

            st.session_state["atradeaday_results"] = results
            st.session_state["atradeaday_df"]      = df

            if results.num_trades == 0:
                st.warning("Strategy ran but found 0 qualifying setups. Try a wider date range or different ticker.")
            else:
                st.success(f"✅ {results.num_trades} trades found.")

        except ValueError as e:
            st.error(f"Data error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    r = st.session_state.get("atradeaday_results")

    if r and r.num_trades > 0:

        # Primary metrics — identical layout to backtest tab
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Return",  f"{r.total_return_pct:.2f}%")
        c2.metric("CAGR",    f"{r.cagr:.2f}%")
        c3.metric("Sharpe",  f"{r.sharpe_ratio:.3f}")
        c4.metric("Sortino", f"{r.sortino_ratio:.3f}")
        c5.metric("Calmar",  f"{r.calmar_ratio:.3f}")
        c6.metric("Max DD",  f"{r.max_drawdown_pct:.2f}%")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades",     r.num_trades)
        c2.metric("Win %",      f"{r.win_rate:.1f}%")
        c3.metric("PF",         f"{r.profit_factor:.2f}")
        c4.metric("Expectancy", f"${r.expectancy:.2f}")
        c5.metric("Payoff",     f"{r.payoff_ratio:.2f}")
        c6.metric("Mkt Time",   f"{r.pct_time_in_market:.0f}%")

        with st.expander("📊 Detailed Metrics", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Avg Win",      f"${r.avg_winner:.2f}")
            c2.metric("Avg Loss",     f"${abs(r.avg_loser):.2f}")
            c3.metric("Avg Win %",    f"{r.avg_winner_pct:.2f}%")
            c4.metric("Avg Loss %",   f"{abs(r.avg_loser_pct):.2f}%")
            c5.metric("Max Consec L", r.max_consecutive_losses)
            c6.metric("Max Consec W", r.max_consecutive_wins)

            c1, c2 = st.columns(2)
            c1.metric("Avg Bars Held",  f"{r.avg_bars_held:.1f}")
            c2.metric("Longest DD",     f"{r.longest_drawdown_bars} bars")

        # Equity curve
        st.plotly_chart(create_equity_chart(r), use_container_width=True)

        # Price chart with trade markers
        df_chart = st.session_state.get("atradeaday_df")
        if df_chart is not None:
            st.plotly_chart(
                create_price_chart_with_trades(df_chart, r.trades),
                use_container_width=True,
            )

        # Trade log
        with st.expander("📋 Trade Log", expanded=False):
            trade_rows = []
            for t in r.trades:
                trade_rows.append({
                    "Date":       str(t.entry_date)[:16],
                    "Direction":  t.direction.upper(),
                    "Entry":      f"{t.entry_price:.4f}",
                    "Exit":       f"{t.exit_price:.4f}" if t.exit_price else "—",
                    "Reason":     t.exit_reason or "—",
                    "P&L $":      f"{t.pnl:+.2f}",
                    "P&L %":      f"{t.pnl_pct:+.2f}%",
                    "Bars":       t.bars_held,
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True)
