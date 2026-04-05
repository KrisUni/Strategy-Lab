"""
ui/tabs/compare.py
==================
Renders the Compare tab content (tabs[2]).
"""

import pandas as pd
import streamlit as st

from ui.helpers import calculate_beta_alpha


def render_compare_tab() -> None:
    st.markdown("### ⚖️ Strategy vs Buy & Hold")

    if st.session_state.df is None:
        st.info("Load data first")
        return

    if not st.session_state.backtest_results:
        st.info("Run backtest first")
        return

    strat_res = st.session_state.backtest_results

    if not strat_res.trades:
        st.warning("No trades in backtest results")
        return

    df = st.session_state.df
    ft = strat_res.trades[0]
    ep = ft.entry_price
    ed = ft.entry_date
    fp = df['close'].iloc[-1]
    bh_pct = (fp - ep) / ep * 100  # B&H is always a long position
    mask = df.index >= ed
    prices = df.loc[mask, 'close']
    peak = prices.expanding().max()
    bh_dd = ((prices - peak) / peak * 100).min()

    st.dataframe(
        pd.DataFrame([
            {'Strategy': '📊 Yours', 'Return %': f"{strat_res.total_return_pct:.2f}%",
             'CAGR': f"{strat_res.cagr:.2f}%", 'Max DD': f"{strat_res.max_drawdown_pct:.2f}%",
             'Sharpe': f"{strat_res.sharpe_ratio:.3f}"},
            {'Strategy': '📈 B&H', 'Return %': f"{bh_pct:.2f}%",
             'CAGR': '-', 'Max DD': f"{bh_dd:.2f}%", 'Sharpe': '-'},
        ]),
        use_container_width=True, hide_index=True,
    )

    diff = strat_res.total_return_pct - bh_pct
    msg = f"{'🏆 Strategy beats Buy & Hold by' if diff > 0 else '📉 B&H beats strategy by'} {abs(diff):.2f}%"
    (st.success if diff > 0 else st.warning)(msg)

    ba = calculate_beta_alpha(
        strat_res.equity_curve.pct_change().dropna(),
        df.loc[mask, 'close'].pct_change().dropna(),
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Beta", f"{ba['beta']:.3f}")
    c2.metric("Alpha (ann)", f"{ba['alpha']:.2f}%")
    c3.metric("Correlation", f"{ba['correlation']:.3f}")
