"""
ui/tabs/heatmap.py
==================
Renders the Parameter Sensitivity heatmap tab (tabs[5]).

create_heatmap() is defined here (not in ui/charts.py) because it imports
BacktestEngine and params_to_strategy, which would create a circular
dependency if placed in the shared charts module.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.backtest import BacktestEngine
from ui.helpers import params_to_strategy
from ui.charts import _chart_layout


# ─────────────────────────────────────────────────────────────────────────────
# Local chart factory
# ─────────────────────────────────────────────────────────────────────────────

def create_heatmap(df, param1, param2, metric, params, capital, commission) -> go.Figure:
    ranges = {
        'pamrp_entry_long': list(range(10, 45, 5)),
        'pamrp_exit_long': list(range(50, 95, 5)),
        'bbwp_threshold_long': list(range(30, 75, 5)),
        'stop_loss_pct_long': [1, 2, 3, 4, 5, 6, 7, 8],
        'take_profit_pct_long': [2, 4, 6, 8, 10, 12, 14],
    }
    r1 = ranges.get(param1, list(range(10, 50, 5)))
    r2 = ranges.get(param2, list(range(10, 50, 5)))
    m = np.zeros((len(r2), len(r1)))
    for i, v1 in enumerate(r1):
        for j, v2 in enumerate(r2):
            tp = params.copy()
            tp[param1] = v1
            tp[param2] = v2
            try:
                m[j, i] = getattr(
                    BacktestEngine(params_to_strategy(tp), capital, commission).run(df.copy()),
                    metric, 0)
            except:
                m[j, i] = 0
    fig = go.Figure(data=go.Heatmap(z=m, x=r1, y=r2, colorscale='RdYlGn', showscale=True))
    fig.update_layout(**_chart_layout(300), xaxis_title=param1, yaxis_title=param2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_heatmap_tab() -> None:
    st.markdown("### 🔥 Parameter Sensitivity")

    param_options = [
        'pamrp_entry_long', 'pamrp_exit_long',
        'bbwp_threshold_long', 'stop_loss_pct_long', 'take_profit_pct_long',
    ]
    c1, c2, c3 = st.columns(3)
    p1 = c1.selectbox("Param X", param_options, index=0)
    p2 = c2.selectbox("Param Y", param_options, index=2)
    hm_metric = c3.selectbox("Metric", ["sharpe_ratio", "total_return_pct", "profit_factor"])

    if st.button("🔥 Generate Heatmap", use_container_width=True):
        if st.session_state.df is not None:
            with st.spinner("Calculating..."):
                st.plotly_chart(
                    create_heatmap(
                        st.session_state.df, p1, p2, hm_metric,
                        st.session_state.params,
                        st.session_state.capital,
                        st.session_state.commission,
                    ),
                    use_container_width=True,
                )
        else:
            st.warning("Load data first")
