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
        --bg-primary: #070b12;
        --bg-secondary: #101722;
        --bg-tertiary: #172131;
        --surface-elevated: rgba(18, 27, 41, 0.9);
        --border-color: rgba(148, 163, 184, 0.18);
        --border-strong: rgba(148, 163, 184, 0.28);
        --text-primary: #e5edf7;
        --text-secondary: #a4b4ca;
        --text-muted: #6f8199;
        --accent-primary: #4f8cff;
        --accent-secondary: #19c1b0;
        --accent-positive: #22c55e;
        --accent-negative: #f87171;
        --gradient-primary: linear-gradient(135deg, #4f8cff 0%, #19c1b0 100%);
        --shadow-soft: 0 18px 36px rgba(2, 6, 23, 0.26);
    }
    html, body, [class*="css"] {
        color-scheme: dark;
    }
    .stApp,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top right, rgba(79, 140, 255, 0.08), transparent 24%),
            radial-gradient(circle at top left, rgba(25, 193, 176, 0.08), transparent 20%),
            linear-gradient(180deg, #070b12 0%, #09111a 52%, #070b12 100%);
        color: var(--text-primary);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b131f 0%, #08111a 100%) !important;
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    .main .block-container {
        padding: 1rem 1.5rem;
        max-width: 100%;
    }
    .main h1 {
        color: var(--text-primary) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.75rem;
    }
    .main h3,
    [data-testid="stSidebar"] h3 {
        font-size: 0.74rem !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 0.75rem !important;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }
    label,
    p,
    span,
    .stCaption,
    .stMarkdown,
    .stMarkdown div {
        color: inherit;
    }
    [data-testid="stSidebar"] label,
    .main label {
        color: var(--text-secondary) !important;
        font-weight: 500;
    }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(79, 140, 255, 0.45), transparent);
        margin: 0.7rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(16, 23, 34, 0.82);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.35rem;
        box-shadow: var(--shadow-soft);
    }
    .stTabs [data-baseweb="tab"] {
        height: 34px;
        padding: 0 0.85rem;
        font-size: 0.8rem;
        border-radius: 9px;
        color: var(--text-secondary) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(79, 140, 255, 0.08);
        color: var(--text-primary) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: 0 10px 24px rgba(79, 140, 255, 0.18);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(19, 29, 44, 0.95) 0%, rgba(10, 16, 26, 0.95) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.45rem;
        box-shadow: var(--shadow-soft);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.62rem !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1rem !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 0.45rem;
        margin-bottom: 0.35rem;
    }
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: 1px solid rgba(79, 140, 255, 0.35);
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        min-height: 2.35rem;
        box-shadow: 0 12px 24px rgba(79, 140, 255, 0.18);
        transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 14px 28px rgba(79, 140, 255, 0.22);
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(13, 19, 30, 0.96) 0%, rgba(10, 15, 24, 0.96) 100%);
        box-shadow: var(--shadow-soft);
        overflow: hidden;
    }
    .streamlit-expanderHeader {
        font-size: 0.82rem !important;
        color: var(--text-primary) !important;
        padding: 0.5rem 0.75rem !important;
    }
    .streamlit-expanderContent {
        background: transparent;
    }
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stDateInput > div > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div,
    div[data-baseweb="select"] > div,
    .stTextArea textarea {
        background: rgba(11, 18, 28, 0.88) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    .stTextArea textarea {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="select"] svg,
    .stNumberInput button,
    .stDateInput button {
        color: var(--text-secondary) !important;
    }
    div[data-baseweb="popover"] {
        background: rgba(10, 16, 26, 0.98) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow-soft) !important;
    }
    div[data-baseweb="menu"] {
        background: transparent !important;
    }
    div[data-baseweb="menu"] div[role="option"] {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="menu"] div[aria-selected="true"] {
        background: rgba(79, 140, 255, 0.14) !important;
    }
    .stAlert {
        background: rgba(15, 23, 35, 0.88);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    .js-plotly-plot .plotly .modebar {
        background: rgba(10, 15, 24, 0.7) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }
    .js-plotly-plot .plotly .modebar-btn path {
        fill: var(--text-secondary) !important;
    }
    .js-plotly-plot .plotly .modebar-btn:hover path {
        fill: var(--text-primary) !important;
    }
    @media (max-width: 1024px) {
        .main .block-container { padding: 0.75rem 1rem; }
        [data-testid="stMetricLabel"] { font-size: 0.55rem !important; }
        [data-testid="stMetricValue"] { font-size: 0.85rem !important; }
    }
    @media (max-width: 768px) {
        .main .block-container { padding: 0.5rem; }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent !important; backdrop-filter: none !important; }
</style>
""", unsafe_allow_html=True)
