"""
Strategy Lab - app.py
=====================
Orchestrator only.  This file is intentionally thin:

  1. Page config
  2. CSS
  3. Session-state init
  4. Sidebar
  5. Tab routing → each tab module owns its own rendering

All logic lives in ui/ and src/.
"""

import streamlit as st

from ui.styles import apply_styles
from ui.session import init_session_state
from ui.sidebar import render_sidebar
from ui.tabs.backtest import render_backtest_tab
from ui.tabs.optimize import render_optimize_tab
from ui.tabs.compare import render_compare_tab
from ui.tabs.montecarlo import render_montecarlo_tab
from ui.tabs.calendar import render_calendar_tab
from ui.tabs.heatmap import render_heatmap_tab
from ui.tabs.multi_asset import render_multi_asset_tab
from ui.tabs.trades import render_trades_tab
from ui.tabs.atradeaday import render_atradeaday_tab

# ── 1. Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategy Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. CSS ────────────────────────────────────────────────────────────────────
apply_styles()

# ── 3. Session state ──────────────────────────────────────────────────────────
init_session_state()

# ── 4. Sidebar ────────────────────────────────────────────────────────────────
render_sidebar()

# ── 5. Main content ───────────────────────────────────────────────────────────
st.markdown("# 📊 Strategy Lab")

tabs = st.tabs([
    "🔬 Backtest",
    "🎯 Optimize",
    "⚖️ Compare",
    "🎲 Monte Carlo",
    "📅 Calendar",
    "🔥 Heatmap",
    "🌐 Multi-Asset",
    "📋 Trades",
    "☝️ A Trade A Day",
])

with tabs[0]:
    render_backtest_tab()

with tabs[1]:
    render_optimize_tab()

with tabs[2]:
    render_compare_tab()

with tabs[3]:
    render_montecarlo_tab()

with tabs[4]:
    render_calendar_tab()

with tabs[5]:
    render_heatmap_tab()

with tabs[6]:
    render_multi_asset_tab()

with tabs[7]:
    render_trades_tab()

with tabs[8]:
    render_atradeaday_tab()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#64748b;font-size:0.7rem;">Strategy Lab v7.1</p>',
    unsafe_allow_html=True,
)
