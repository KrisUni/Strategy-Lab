"""
ui/sidebar.py
=============
Renders the full Strategy Lab sidebar.

render_sidebar() reads from and writes to st.session_state.params directly.
It returns nothing — all state changes go through session state so the rest
of the app can react on rerun.
"""

import streamlit as st
from datetime import datetime, timedelta

from src.data import fetch_yfinance, generate_sample_data, load_csv


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_INTERVAL_MAX_DAYS = {
    '1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60,
    '60m': 730, '90m': 60, '1h': 730,
}

INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the full sidebar. Mutates st.session_state.params in place."""
    p = st.session_state.params

    with st.sidebar:
        st.markdown("# 📊 Strategy Lab")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        _render_data_section()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Strategy")

        p['trade_direction'] = st.selectbox(
            "Direction", ["Long Only", "Short Only", "Both"],
            index=["Long Only", "Short Only", "Both"].index(p['trade_direction']))

        _render_position_sizing(p)
        _render_entry_indicators(p)
        _render_visual_indicators(p)
        _render_exit_indicators(p)

    st.session_state.params = p


# ─────────────────────────────────────────────────────────────────────────────
# Data section
# ─────────────────────────────────────────────────────────────────────────────

def _render_data_section() -> None:
    st.markdown("### 📈 Data")
    data_src = st.radio("Source", ["Yahoo Finance", "Sample", "CSV"],
                        horizontal=True, label_visibility="collapsed")

    symbol, interval, days = "SPY", "1d", 730
    uploaded_file = None

    if data_src == "Yahoo Finance":
        symbol = st.text_input("Symbol", "SPY")
        c1, c2 = st.columns(2)
        interval = c1.selectbox("Interval", INTERVALS, index=INTERVALS.index("1d"))
        max_days = _INTERVAL_MAX_DAYS.get(interval)
        if max_days:
            days = c2.slider("Days", min_value=1, max_value=max_days, value=min(max_days, 60),
                help=f"Yahoo Finance only provides the last **{max_days} calendar days** for `{interval}` data.")
            st.caption(f"⚠️ `{interval}` data limited to last **{max_days} days** by Yahoo Finance API.")
        else:
            days = c2.number_input("Days", 30, 7300, 730)

    elif data_src == "Sample":
        c1, c2 = st.columns(2)
        days = c1.number_input("Bars", 100, 5000, 500)
        sample_vol = c2.slider("Volatility", 0.005, 0.04, 0.015, 0.005,
                               help="Daily return std dev. 0.015 ≈ typical equity.")

    else:  # CSV
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"],
            help="Expected columns: date, open, high, low, close[, volume]. Date must be parseable as datetime.")

    if st.button("📥 Load", use_container_width=True):
        with st.spinner("Loading..."):
            try:
                if data_src == "Yahoo Finance":
                    end_dt = datetime.now()
                    start_dt = end_dt - timedelta(days=days)
                    df = fetch_yfinance(symbol, str(start_dt.date()), str(end_dt.date()), interval)
                    st.session_state.df = df
                    if df.attrs.get('date_range_clamped'):
                        st.warning(
                            f"⚠️ Date range clamped: `{interval}` data is limited to "
                            f"the last **{df.attrs['max_interval_days']} calendar days**. "
                            f"Loaded: **{df.attrs['actual_start']}** → **{df.attrs['actual_end']}**.")
                    else:
                        st.success(f"✅ **{len(df):,} bars** · {df.attrs['actual_start']} → {df.attrs['actual_end']}")

                elif data_src == "Sample":
                    df = generate_sample_data(days=int(days), volatility=sample_vol, seed=42)
                    st.session_state.df = df
                    st.success(f"✅ **{len(df):,} bars** synthetic · {df.attrs['actual_start']} → {df.attrs['actual_end']}")

                else:  # CSV
                    if uploaded_file is None:
                        st.warning("Please upload a CSV file first.")
                    else:
                        import tempfile, os
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        try:
                            df = load_csv(tmp_path)
                            st.session_state.df = df
                            st.success(f"✅ **{len(df):,} bars** from CSV · "
                                       f"{str(df.index[0].date())} → {str(df.index[-1].date())}")
                        finally:
                            os.unlink(tmp_path)

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────────────

def _render_position_sizing(p: dict) -> None:
    with st.expander("💰 Position Sizing", expanded=False):
        p['position_size_pct'] = st.slider("Position %", 10, 100, int(p['position_size_pct']), 10)
        p['use_kelly'] = st.toggle("Kelly Criterion", p['use_kelly'])
        if p['use_kelly']:
            p['kelly_fraction'] = st.slider("Kelly Fraction", 0.1, 1.0, p['kelly_fraction'], 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry indicators
# ─────────────────────────────────────────────────────────────────────────────

def _render_entry_indicators(p: dict) -> None:
    with st.expander("📊 PAMRP", expanded=False):
        p['pamrp_enabled'] = st.toggle("Enable", p['pamrp_enabled'], key="pe")
        if p['pamrp_enabled']:
            p['pamrp_length'] = st.slider("Length", 5, 50, p['pamrp_length'], key="pl")
            p['pamrp_entry_long'] = st.slider("Entry Long", 5, 50, p['pamrp_entry_long'], key="pel")
            p['pamrp_entry_short'] = st.slider("Entry Short", 50, 95, p['pamrp_entry_short'], key="pes")
            p['pamrp_exit_long'] = st.slider("Exit Long", 50, 100, p['pamrp_exit_long'], key="pxl")
            p['pamrp_exit_short'] = st.slider("Exit Short", 0, 50, p['pamrp_exit_short'], key="pxs")

    with st.expander("📊 BBWP", expanded=False):
        p['bbwp_enabled'] = st.toggle("Enable", p['bbwp_enabled'], key="be")
        if p['bbwp_enabled']:
            p['bbwp_length'] = st.slider("Length", 5, 30, p['bbwp_length'], key="bl")
            p['bbwp_lookback'] = st.slider("Lookback", 50, 400, p['bbwp_lookback'], key="blb")
            p['bbwp_sma_length'] = st.slider("SMA", 2, 15, p['bbwp_sma_length'], key="bsma")
            p['bbwp_threshold_long'] = st.slider("Thresh Long", 20, 80, p['bbwp_threshold_long'], key="btl")
            p['bbwp_threshold_short'] = st.slider("Thresh Short", 20, 80, p['bbwp_threshold_short'], key="bts")
            p['bbwp_ma_filter'] = st.selectbox("MA Filter", ["disabled", "decreasing", "increasing"],
                index=["disabled", "decreasing", "increasing"].index(p['bbwp_ma_filter']), key="bmf")

    with st.expander("📈 ADX", expanded=False):
        p['adx_enabled'] = st.toggle("Enable", p['adx_enabled'], key="ae")
        if p['adx_enabled']:
            p['adx_length'] = st.slider("Length", 7, 21, p['adx_length'], key="al")
            p['adx_threshold'] = st.slider("Threshold", 10, 40, p['adx_threshold'], key="at")

    with st.expander("📈 MA Trend", expanded=False):
        p['ma_trend_enabled'] = st.toggle("Enable", p['ma_trend_enabled'], key="mae")
        if p['ma_trend_enabled']:
            p['ma_type'] = st.selectbox("Type", ["sma", "ema"],
                index=["sma", "ema"].index(p['ma_type']), key="mat")
            p['ma_fast_length'] = st.slider("Fast", 10, 100, p['ma_fast_length'], key="maf")
            p['ma_slow_length'] = st.slider("Slow", 50, 400, p['ma_slow_length'], key="mas")

    with st.expander("📈 RSI", expanded=False):
        p['rsi_enabled'] = st.toggle("Enable", p['rsi_enabled'], key="re")
        if p['rsi_enabled']:
            p['rsi_length'] = st.slider("Length", 5, 21, p['rsi_length'], key="rl")
            p['rsi_oversold'] = st.slider("Oversold", 15, 45, p['rsi_oversold'], key="ros")
            p['rsi_overbought'] = st.slider("Overbought", 55, 85, p['rsi_overbought'], key="rob")

    with st.expander("📊 Volume", expanded=False):
        p['volume_enabled'] = st.toggle("Enable", p['volume_enabled'], key="ve")
        if p['volume_enabled']:
            p['volume_ma_length'] = st.slider("MA Length", 10, 50, p['volume_ma_length'], key="vml")
            p['volume_multiplier'] = st.slider("Multiplier", 0.5, 2.0, p['volume_multiplier'], 0.1, key="vm")

    with st.expander("📈 Supertrend", expanded=False):
        p['supertrend_enabled'] = st.toggle("Enable", p['supertrend_enabled'], key="ste")
        if p['supertrend_enabled']:
            p['supertrend_period'] = st.slider("Period", 5, 20, p['supertrend_period'], key="stp")
            p['supertrend_multiplier'] = st.slider("Mult", 1.0, 5.0, p['supertrend_multiplier'], 0.5, key="stm")

    with st.expander("📈 VWAP", expanded=False):
        p['vwap_enabled'] = st.toggle("Enable", p['vwap_enabled'], key="vwe")

    with st.expander("📈 MACD", expanded=False):
        p['macd_enabled'] = st.toggle("Enable", p['macd_enabled'], key="mce")
        if p['macd_enabled']:
            p['macd_fast'] = st.slider("Fast", 5, 20, p['macd_fast'], key="mcf")
            p['macd_slow'] = st.slider("Slow", 15, 40, p['macd_slow'], key="mcs")
            p['macd_signal'] = st.slider("Signal", 5, 15, p['macd_signal'], key="mcsi")


# ─────────────────────────────────────────────────────────────────────────────
# Visual indicators (display-only)
# ─────────────────────────────────────────────────────────────────────────────

def _render_visual_indicators(p: dict) -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🔭 Visual Indicators")

    with st.expander("📐 HPDR Bands", expanded=False):
        p['hpdr_enabled'] = st.toggle("Show on chart", p['hpdr_enabled'], key="hpdr_e")
        if p['hpdr_enabled']:
            p['hpdr_lookback'] = st.slider("Lookback", 30, 504, p['hpdr_lookback'], key="hpdr_lb",
                help="Rolling window for return distribution (bars). 252 ≈ 1 trading year.")
            st.caption("5 zones: teal (±0.5σ) → green → yellow → orange → red (±2.5σ)")
            st.caption("Y-axis is always locked to price range.")

    with st.expander("〰️ RSI Hidden Divergence", expanded=False):
        p['rsi_div_enabled'] = st.toggle("Show on chart", p['rsi_div_enabled'], key="rdiv_e")
        if p['rsi_div_enabled']:
            p['rsi_div_length'] = st.slider("RSI Length", 7, 21, p['rsi_div_length'], key="rdiv_l")
            c1, c2 = st.columns(2)
            p['rsi_div_pivot_left'] = c1.slider("Pivot Left", 2, 10, p['rsi_div_pivot_left'], key="rdiv_pl",
                help="Bars left of swing required to confirm a pivot.")
            p['rsi_div_pivot_right'] = c2.slider("Pivot Right", 2, 10, p['rsi_div_pivot_right'], key="rdiv_pr",
                help="Bars right required — equals signal lag.")
            st.caption(f"⚠️ Signal lag = {p['rsi_div_pivot_right']} bars")
            st.caption("🟢 Hidden Bull = Higher Low price + Lower Low RSI")
            st.caption("🔴 Hidden Bear = Lower High price + Higher High RSI")


# ─────────────────────────────────────────────────────────────────────────────
# Exit indicators
# ─────────────────────────────────────────────────────────────────────────────

def _render_exit_indicators(p: dict) -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🚪 Exits")

    with st.expander("🛑 Stop Loss", expanded=False):
        p['stop_loss_enabled'] = st.toggle("Enable", p['stop_loss_enabled'], key="sle")
        if p['stop_loss_enabled']:
            p['stop_loss_pct_long'] = st.slider("% Long", 0.5, 15.0, p['stop_loss_pct_long'], 0.5, key="sll")
            p['stop_loss_pct_short'] = st.slider("% Short", 0.5, 15.0, p['stop_loss_pct_short'], 0.5, key="sls")

    with st.expander("🎯 Take Profit", expanded=False):
        p['take_profit_enabled'] = st.toggle("Enable", p['take_profit_enabled'], key="tpe")
        if p['take_profit_enabled']:
            p['take_profit_pct_long'] = st.slider("% Long", 1.0, 30.0, p['take_profit_pct_long'], 0.5, key="tpl")
            p['take_profit_pct_short'] = st.slider("% Short", 1.0, 30.0, p['take_profit_pct_short'], 0.5, key="tps")

    with st.expander("📉 Trailing Stop", expanded=False):
        p['trailing_stop_enabled'] = st.toggle("Enable", p['trailing_stop_enabled'], key="tse")
        if p['trailing_stop_enabled']:
            p['trailing_stop_pct'] = st.slider("Trail %", 0.5, 10.0, p['trailing_stop_pct'], 0.5, key="tsp")

    with st.expander("📉 ATR Trail", expanded=False):
        p['atr_trailing_enabled'] = st.toggle("Enable", p['atr_trailing_enabled'], key="ate")
        if p['atr_trailing_enabled']:
            p['atr_length'] = st.slider("Length", 7, 21, p['atr_length'], key="atl")
            p['atr_multiplier'] = st.slider("Mult", 1.0, 5.0, p['atr_multiplier'], 0.5, key="atm")

    with st.expander("📊 PAMRP Exit", expanded=False):
        p['pamrp_exit_enabled'] = st.toggle("Enable", p['pamrp_exit_enabled'], key="pxe")
        if p['pamrp_exit_enabled']:
            p['pamrp_length']     = st.slider("Length", 5, 50, p['pamrp_length'], key="pxl_len",
                help="PAMRP period — shared with entry if both enabled.")
            p['pamrp_exit_long']  = st.slider("Exit Long (overbought)", 50, 100, p['pamrp_exit_long'],  key="pxl_exit")
            p['pamrp_exit_short'] = st.slider("Exit Short (oversold)",  0,   50,  p['pamrp_exit_short'], key="pxs_exit")

    with st.expander("📈 Stoch RSI Exit", expanded=False):
        p['stoch_rsi_exit_enabled'] = st.toggle("Enable", p['stoch_rsi_exit_enabled'], key="sre")
        if p['stoch_rsi_exit_enabled']:
            p['stoch_rsi_length'] = st.slider("Length", 7, 21, p['stoch_rsi_length'], key="srl")
            p['stoch_rsi_k'] = st.slider("K", 2, 5, p['stoch_rsi_k'], key="srk")
            p['stoch_rsi_d'] = st.slider("D", 2, 5, p['stoch_rsi_d'], key="srd")
            p['stoch_rsi_overbought'] = st.slider("OB", 60, 90, p['stoch_rsi_overbought'], key="srob")
            p['stoch_rsi_oversold'] = st.slider("OS", 10, 40, p['stoch_rsi_oversold'], key="sros")

    with st.expander("⏱️ Time Exit", expanded=False):
        p['time_exit_enabled'] = st.toggle("Enable", p['time_exit_enabled'], key="txe")
        if p['time_exit_enabled']:
            p['time_exit_bars'] = st.slider("Max Bars", 5, 100, p['time_exit_bars'], key="txb")

    with st.expander("📈 MA Exit", expanded=False):
        p['ma_exit_enabled'] = st.toggle("Enable", p['ma_exit_enabled'], key="mxe")
        if p['ma_exit_enabled']:
            p['ma_exit_fast'] = st.slider("Fast", 3, 20, p['ma_exit_fast'], key="mxf")
            p['ma_exit_slow'] = st.slider("Slow", 10, 40, p['ma_exit_slow'], key="mxs")

    with st.expander("📊 BBWP Exit", expanded=False):
        p['bbwp_exit_enabled'] = st.toggle("Enable", p['bbwp_exit_enabled'], key="bxe")
        if p['bbwp_exit_enabled']:
            p['bbwp_exit_threshold'] = st.slider("Threshold", 60, 95, p['bbwp_exit_threshold'], key="bxt")
