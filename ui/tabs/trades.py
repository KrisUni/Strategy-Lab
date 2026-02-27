"""
ui/tabs/trades.py
=================
Renders the Trade Log tab content (tabs[7]).
"""

import numpy as np
import pandas as pd
import streamlit as st


def render_trades_tab() -> None:
    st.markdown("### 📋 Trade Log")

    if not (st.session_state.backtest_results and st.session_state.backtest_results.trades):
        st.info("Run backtest first")
        return

    trades = st.session_state.backtest_results.trades

    c1, c2 = st.columns(2)
    dir_f = c1.selectbox("Dir", ["All", "Long", "Short"])
    res_f = c2.selectbox("Result", ["All", "Winners", "Losers"])

    flt = trades
    if dir_f != "All":
        flt = [t for t in flt if t.direction.lower() == dir_f.lower()]
    if res_f == "Winners":
        flt = [t for t in flt if t.pnl > 0]
    elif res_f == "Losers":
        flt = [t for t in flt if t.pnl <= 0]

    if not flt:
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Trades", len(flt))
    total_pnl = sum(t.pnl for t in flt)
    c2.metric("Total PnL", f"${total_pnl:,.2f}")
    c3.metric("Avg", f"${np.mean([t.pnl for t in flt]):.2f}")

    trade_df = pd.DataFrame([{
        'Entry': t.entry_date,
        'Exit': t.exit_date,
        'Dir': t.direction,
        'Entry$': round(t.entry_price, 2),
        'Exit$': round(t.exit_price, 2) if t.exit_price else None,
        'Size$': round(t.size_dollars, 0),
        'PnL$': round(t.pnl, 2),
        'PnL%': round(t.pnl_pct, 3),
        'Bars': t.bars_held,
        'MAE%': round(t.mae, 2),
        'MFE%': round(t.mfe, 2),
        'Reason': t.exit_reason,
    } for t in flt])

    st.download_button("📥 Export CSV", trade_df.to_csv(index=False), "trades.csv", use_container_width=True)
    st.dataframe(trade_df, use_container_width=True, hide_index=True)
