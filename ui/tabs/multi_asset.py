"""
ui/tabs/multi_asset.py
======================
Renders the Multi-Asset Portfolio tab content (tabs[6]).
"""

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from src.backtest import BacktestEngine
from src.data import fetch_yfinance
from ui.helpers import params_to_strategy
from ui.charts import create_multi_asset_chart


def render_multi_asset_tab() -> None:
    st.markdown("### 🌐 Multi-Asset Portfolio")

    symbols_input = st.text_input("Symbols (comma-separated)", "SPY, QQQ, IWM")

    if st.button("📊 Run Multi-Asset", use_container_width=True):
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        results_dict = {}
        with st.spinner(f"Testing {len(symbols)} assets..."):
            for sym in symbols:
                try:
                    end = datetime.now()
                    start = end - timedelta(days=730)
                    df = fetch_yfinance(sym, str(start.date()), str(end.date()), '1d')
                    results_dict[sym] = BacktestEngine(
                        params_to_strategy(st.session_state.params),
                        st.session_state.capital,
                        st.session_state.commission,
                    ).run(df)
                except Exception as e:
                    st.warning(f"{sym}: {str(e)[:30]}")

        if results_dict:
            st.plotly_chart(create_multi_asset_chart(results_dict), use_container_width=True)
            st.dataframe(
                pd.DataFrame([
                    {
                        'Symbol': s,
                        'Return %': f"{r.total_return_pct:.2f}%",
                        'CAGR': f"{r.cagr:.2f}%",
                        'Sharpe': f"{r.sharpe_ratio:.3f}",
                        'Max DD': f"{r.max_drawdown_pct:.2f}%",
                        'Trades': r.num_trades,
                    }
                    for s, r in results_dict.items()
                ]),
                use_container_width=True,
                hide_index=True,
            )
