"""
ui/styles.py
============
Global CSS injection for Strategy Lab.
Call apply_styles() once at app startup, immediately after set_page_config.
"""

import streamlit as st


def apply_styles() -> None:
    """Inject the application-wide CSS into the Streamlit page."""
    st.markdown("""
<style>
    :root {
        --bg-primary: #0a0e14;
        --bg-secondary: #11151c;
        --bg-tertiary: #1a1f2e;
        --border-color: #2d3548;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --gradient-primary: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    }
    .stApp { background: var(--bg-primary); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #0a0e14 100%) !important; }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem !important; background: var(--gradient-primary); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stSidebar"] h3 { font-size: 0.7rem !important; color: var(--text-secondary) !important; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.6rem !important; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border-color); }
    .main .block-container { padding: 1rem 1.5rem; max-width: 100%; }
    .main h1 { font-size: 1.4rem !important; }
    .main h3 { font-size: 1rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: var(--bg-secondary); border-radius: 6px; padding: 2px; }
    .stTabs [data-baseweb="tab"] { height: 32px; padding: 0 12px; font-size: 0.8rem; }
    .stTabs [aria-selected="true"] { background: var(--gradient-primary) !important; color: white !important; border-radius: 4px; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #12161f 0%, #0d1117 100%); border: 1px solid var(--border-color); border-radius: 6px; padding: 0.5rem 0.7rem; margin-bottom: 0.4rem; }
    [data-testid="stMetricLabel"] { font-size: 0.6rem !important; }
    [data-testid="stMetricValue"] { font-size: 1rem !important; }
    [data-testid="stHorizontalBlock"] { gap: 0.4rem; margin-bottom: 0.3rem; }
    .stButton > button { background: var(--gradient-primary); color: white; border: none; border-radius: 5px; font-size: 0.8rem; }
    .streamlit-expanderHeader { font-size: 0.8rem !important; padding: 0.4rem 0.6rem !important; }
    .divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border-color), transparent); margin: 0.6rem 0; }
    @media (max-width: 1024px) { .main .block-container { padding: 0.75rem 1rem; } [data-testid="stMetricLabel"] { font-size: 0.55rem !important; } [data-testid="stMetricValue"] { font-size: 0.85rem !important; } }
    @media (max-width: 768px) { .main .block-container { padding: 0.5rem; } [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; } }
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent !important; backdrop-filter: none !important; }
</style>
""", unsafe_allow_html=True)
