"""
ui/tabs/compare.py
==================
Renders the Compare tab content (tabs[2]).
"""

import streamlit as st

from src.backtest import BacktestEngine
from ui.helpers import params_to_strategy, calculate_beta_alpha


def render_compare_tab() -> None:
    st.markdown("### ⚖️ Strategy vs Buy & Hold")

    if st.session_state.df is None:
        st.info("Load data first")
        return

    if st.button("🔄 Compare", use_container_width=True):
        df = st.session_state.df.copy()
        strat_res = BacktestEngine(
            params_to_strategy(st.session_state.params),
            st.session_state.capital,
            st.session_state.commission,
        ).run(df)

        if strat_res.trades:
            ft = strat_res.trades[0]
            ep = ft.entry_price
            ed = ft.entry_date
            fp = df['close'].iloc[-1]
            bh_pct = ((ep - fp) / ep * 100) if ft.direction == 'short' else ((fp - ep) / ep * 100)
            mask = df.index >= ed
            prices = df.loc[mask, 'close']
            peak = prices.expanding().max()
            bh_dd = ((prices - peak) / peak * 100).min()

            st.dataframe(
                __import__('pandas').DataFrame([
                    {'Strategy': '📊 Yours', 'Return %': f"{strat_res.total_return_pct:.2f}%",
                     'CAGR': f"{strat_res.cagr:.2f}%", 'Max DD': f"{strat_res.max_drawdown_pct:.2f}%",
                     'Sharpe': f"{strat_res.sharpe_ratio:.3f}"},
                    {'Strategy': '📈 B&H', 'Return %': f"{bh_pct:.2f}%",
                     'CAGR': '-', 'Max DD': f"{bh_dd:.2f}%", 'Sharpe': '-'},
                ]),
                use_container_width=True, hide_index=True,
            )

            diff = strat_res.total_return_pct - bh_pct
            msg = f"{'🏆 Strategy beats' if diff > 0 else '📉 B&H beats strategy by'} {abs(diff):.2f}%"
            (st.success if diff > 0 else st.warning)(msg)

            ba = calculate_beta_alpha(
                strat_res.equity_curve.pct_change().dropna(),
                df.loc[mask, 'close'].pct_change().dropna(),
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Beta", f"{ba['beta']:.3f}")
            c2.metric("Alpha (ann)", f"{ba['alpha']:.2f}%")
            c3.metric("Correlation", f"{ba['correlation']:.3f}")
        else:
            st.warning("No trades generated")
