"""
ui/tabs/backtest.py
===================
Renders the Backtest tab content (tabs[0]).
"""

import streamlit as st

from src.backtest import BacktestEngine
from src.indicators import hpdr_bands
from ui.helpers import params_to_strategy
from ui.charts import (
    create_equity_chart,
    create_price_chart_with_trades,
    create_rsi_divergence_chart,
)


def render_backtest_tab() -> None:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    st.session_state.capital = c1.number_input("Capital $", 1000, 1000000, st.session_state.capital, 1000)
    st.session_state.commission = c2.number_input("Comm %", 0.0, 1.0, st.session_state.commission, 0.01)
    slippage = c3.number_input("Slip %", 0.0, 0.5, 0.0, 0.01)
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Run", type="primary", use_container_width=True)

    if run:
        if st.session_state.df is None:
            st.warning("Load data first!")
        else:
            with st.spinner("Running..."):
                engine = BacktestEngine(
                    params_to_strategy(st.session_state.params),
                    st.session_state.capital,
                    st.session_state.commission,
                    slippage,
                )
                st.session_state.backtest_results = engine.run(st.session_state.df.copy())
                st.success(f"✅ {st.session_state.backtest_results.num_trades} trades")

    if st.session_state.backtest_results:
        r = st.session_state.backtest_results
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Return", f"{r.total_return_pct:.2f}%")
        c2.metric("CAGR", f"{r.cagr:.2f}%")
        c3.metric("Sharpe", f"{r.sharpe_ratio:.3f}")
        c4.metric("Sortino", f"{r.sortino_ratio:.3f}")
        c5.metric("Calmar", f"{r.calmar_ratio:.3f}")
        c6.metric("Max DD", f"{r.max_drawdown_pct:.2f}%")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades", r.num_trades)
        c2.metric("Win%", f"{r.win_rate:.1f}%")
        c3.metric("PF", f"{r.profit_factor:.2f}")
        c4.metric("Expectancy", f"${r.expectancy:.2f}")
        c5.metric("Payoff", f"{r.payoff_ratio:.2f}")
        c6.metric("Mkt Time", f"{r.pct_time_in_market:.0f}%")
        with st.expander("📊 Detailed Metrics", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Avg Win", f"${r.avg_winner:.2f}")
            c2.metric("Avg Loss", f"${abs(r.avg_loser):.2f}")
            c3.metric("Avg Win %", f"{r.avg_winner_pct:.2f}%")
            c4.metric("Avg Loss %", f"{abs(r.avg_loser_pct):.2f}%")
            c5.metric("Max Consec L", r.max_consecutive_losses)
            c6.metric("Max Consec W", r.max_consecutive_wins)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Bars", f"{r.avg_bars_held:.1f}")
            c2.metric("Longest DD", f"{r.longest_drawdown_bars} bars")
            c3.metric("Avg MAE", f"{r.avg_mae:.2f}%")
            c4.metric("Avg MFE", f"{r.avg_mfe:.2f}%")
        st.plotly_chart(create_equity_chart(r), use_container_width=True)

    # ── Price chart (with optional HPDR overlay) ──────────────────────────────
    if st.session_state.df is not None:
        p = st.session_state.params
        df_chart = st.session_state.df

        bands_data = None
        if p.get('hpdr_enabled'):
            try:
                bands_data = hpdr_bands(
                    df_chart['close'],
                    lookback=int(p.get('hpdr_lookback', 252)),
                    z_scores=(0.5, 1.0, 1.5, 2.0, 2.5),
                )
            except Exception as e:
                st.warning(f"HPDR bands error: {e}")

        trades_to_show = st.session_state.backtest_results.trades if st.session_state.backtest_results else None
        st.plotly_chart(
            create_price_chart_with_trades(df_chart, trades_to_show, bands=bands_data),
            use_container_width=True,
        )

        # ── RSI Hidden Divergence sub-panel ───────────────────────────────────
        if p.get('rsi_div_enabled'):
            min_bars = 2 * (p['rsi_div_pivot_left'] + p['rsi_div_pivot_right'] + p['rsi_div_length'])
            if len(df_chart) < min_bars:
                st.warning(f"RSI Divergence needs at least {min_bars} bars. Load more data.")
            else:
                try:
                    st.markdown("##### 〰️ RSI Hidden Divergence")
                    st.caption(
                        f"RSI({p['rsi_div_length']}) · "
                        f"Pivot {p['rsi_div_pivot_left']}/{p['rsi_div_pivot_right']} · "
                        f"Signal lag = {p['rsi_div_pivot_right']} bars"
                    )
                    st.plotly_chart(
                        create_rsi_divergence_chart(df_chart, p),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"RSI Divergence chart error: {e}")
