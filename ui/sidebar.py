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
from src.persistence import delete_strategy, list_strategies, load_strategy, save_strategy
from ui.session import get_default_params
from ui.sidebar_renderer import render_indicator_section


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_INTERVAL_MAX_DAYS = {
    '1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60,
    '60m': 730, '90m': 60, '1h': 730,
}

INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
TRADE_DIRECTION_OPTIONS = ["Long Only", "Short Only", "Both"]
ENTRY_CONFLICT_OPTIONS = ["skip", "prefer_long", "prefer_short"]


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
        _render_strategy_persistence()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Strategy")

        _sync_trade_direction_widget(p.get('trade_direction', 'Long Only'))
        p['trade_direction'] = st.selectbox(
            "Direction",
            TRADE_DIRECTION_OPTIONS,
            key="tdir",
        )
        p['entry_conflict_mode'] = st.selectbox(
            "If Long And Short Trigger Together",
            ENTRY_CONFLICT_OPTIONS,
            index=ENTRY_CONFLICT_OPTIONS.index(p.get('entry_conflict_mode', 'skip')),
            format_func=lambda value: {
                "skip": "Skip ambiguous bar",
                "prefer_long": "Prefer long",
                "prefer_short": "Prefer short",
            }[value],
            key="ecm",
            help="Controls what happens when both entry directions are true on the same bar in Both mode.",
        )

        _render_position_sizing(p)
        _render_entry_indicators(p)
        _render_visual_indicators(p)
        _render_exit_indicators(p)

    st.session_state.params = p


def _sync_trade_direction_widget(current_direction: str) -> None:
    if current_direction not in TRADE_DIRECTION_OPTIONS:
        current_direction = "Long Only"

    previous_strategy_direction = st.session_state.get('_last_sidebar_trade_direction')

    if 'tdir' not in st.session_state:
        st.session_state.tdir = current_direction
    elif (
        previous_strategy_direction is not None
        and current_direction != previous_strategy_direction
        and st.session_state.tdir == previous_strategy_direction
    ):
        st.session_state.tdir = current_direction

    st.session_state._last_sidebar_trade_direction = current_direction


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
        p['position_size_pct'] = st.slider("Position %", 10, 100, int(p['position_size_pct']), 10, key="ps_size_pct")
        p['use_kelly'] = st.toggle("Kelly Criterion", p['use_kelly'], key="ps_use_kelly")
        if p['use_kelly']:
            p['kelly_fraction'] = st.slider("Kelly Fraction", 0.1, 1.0, p['kelly_fraction'], 0.1, key="ps_kelly_frac")


# ─────────────────────────────────────────────────────────────────────────────
# Entry indicators
# ─────────────────────────────────────────────────────────────────────────────

def _render_entry_indicators(p: dict) -> None:
    p['entry_operator'] = st.radio(
        "Combine Entry Filters",
        options=["and", "or"],
        index=["and", "or"].index(p.get('entry_operator', 'and')),
        format_func=lambda value: "Match all (AND)" if value == "and" else "Match any (OR)",
        horizontal=True,
        key="eop",
    )
    render_indicator_section("entry", p)


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
    p['allow_same_bar_exit'] = st.toggle(
        "Allow Same-Bar Entry/Exit",
        value=p.get('allow_same_bar_exit', True),
        key="same_bar_exit",
        help="When enabled, stop-loss, take-profit, trailing, and ATR exits can close a trade on the same bar it opened. Disable to force the trade to survive at least until the next bar.",
    )
    p['allow_same_bar_reversal'] = st.toggle(
        "Allow Same-Bar Reversal After Exit",
        value=p.get('allow_same_bar_reversal', False),
        key="same_bar_reversal",
        help="When enabled, a signal or time exit at the current bar open may be followed by an opposite-direction entry on that same bar. Intrabar stop-type exits still wait until the next bar.",
    )
    p['exit_operator'] = st.radio(
        "Combine Signal Exits",
        options=["and", "or"],
        index=["and", "or"].index(p.get('exit_operator', 'or')),
        format_func=lambda value: "Match all (AND)" if value == "and" else "Match any (OR)",
        horizontal=True,
        key="xop",
        help="Applies to signal-based exits below. Stop-loss, take-profit, trailing, ATR, and time exits still trigger independently.",
    )

    render_indicator_section("exit", p)
    render_indicator_section("risk", p)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy persistence
# ─────────────────────────────────────────────────────────────────────────────

def _do_save_strategy(name: str) -> None:
    save_strategy(
        name,
        dict(st.session_state.params),
        {
            "capital": int(st.session_state.capital),
            "commission": float(st.session_state.commission),
            "slippage": float(st.session_state.slippage),
        },
    )
    st.toast(f"Saved '{name}'")
    st.session_state._strat_clear_name = True


def _render_strategy_persistence() -> None:
    st.markdown("### 💾 Strategies")

    saved = list_strategies()
    names = [s["name"] for s in saved]

    # Clear stale confirmation flags
    for flag in ("_strat_delete_confirm", "_strat_overwrite_confirm"):
        val = st.session_state.get(flag)
        if val is not None and val not in names:
            st.session_state.pop(flag)

    if names:
        selected = st.selectbox(
            "Saved strategies",
            names,
            key="strat_select",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns(2)

        if c1.button("Load", use_container_width=True, key="strat_load_btn"):
            loaded_params, loaded_exec = load_strategy(selected)
            merged = {**get_default_params(), **loaded_params}
            st.session_state.params = merged
            # Sync manually-keyed sidebar widgets
            st.session_state.tdir = merged.get("trade_direction", "Long Only")
            st.session_state.ecm = merged.get("entry_conflict_mode", "skip")
            st.session_state.eop = merged.get("entry_operator", "and")
            st.session_state.same_bar_exit = bool(merged.get("allow_same_bar_exit", True))
            st.session_state.same_bar_reversal = bool(merged.get("allow_same_bar_reversal", False))
            st.session_state.xop = merged.get("exit_operator", "or")
            # Sync position sizing widget keys
            st.session_state.ps_size_pct = int(merged.get("position_size_pct", 100))
            st.session_state.ps_use_kelly = bool(merged.get("use_kelly", False))
            st.session_state.ps_kelly_frac = float(merged.get("kelly_fraction", 0.5))
            # Sync visual indicator widget keys
            st.session_state.hpdr_e = bool(merged.get("hpdr_enabled", False))
            st.session_state.hpdr_lb = int(merged.get("hpdr_lookback", 252))
            st.session_state.rdiv_e = bool(merged.get("rsi_div_enabled", False))
            st.session_state.rdiv_l = int(merged.get("rsi_div_length", 14))
            st.session_state.rdiv_pl = int(merged.get("rsi_div_pivot_left", 5))
            st.session_state.rdiv_pr = int(merged.get("rsi_div_pivot_right", 5))
            # Sync indicator param widget keys (render_param_widget uses f"widget_{name}")
            for k, v in merged.items():
                st.session_state[f"widget_{k}"] = v
            # Sync execution params
            if loaded_exec:
                cap = int(loaded_exec.get("capital", st.session_state.capital))
                com = float(loaded_exec.get("commission", st.session_state.commission))
                slp = float(loaded_exec.get("slippage", st.session_state.slippage))
                st.session_state.capital = cap
                st.session_state.commission = com
                st.session_state.slippage = slp
                st.session_state.exec_capital = cap
                st.session_state.exec_commission = com
                st.session_state.exec_slippage = slp
            st.session_state.pop("_strat_delete_confirm", None)
            st.toast(f"Loaded '{selected}'")
            st.rerun()

        if c2.button("Delete", use_container_width=True, key="strat_delete_btn"):
            st.session_state._strat_delete_confirm = selected
            st.session_state.pop("_strat_overwrite_confirm", None)

        if st.session_state.get("_strat_delete_confirm") == selected:
            st.warning(f"Delete **{selected}**?")
            d1, d2 = st.columns(2)
            if d1.button("Yes, delete", key="strat_delete_yes"):
                delete_strategy(selected)
                st.session_state.pop("_strat_delete_confirm", None)
                st.session_state.pop("strat_select", None)
                st.toast(f"Deleted '{selected}'")
                st.rerun()
            if d2.button("Cancel", key="strat_delete_no"):
                st.session_state.pop("_strat_delete_confirm", None)
                st.rerun()
    else:
        st.caption("No saved strategies yet.")

    if st.session_state.pop("_strat_clear_name", False):
        st.session_state.strat_name_input = ""
    name_input = st.text_input(
        "Name",
        key="strat_name_input",
        placeholder="e.g. PAMRP+BBWP long SPY",
        label_visibility="collapsed",
    )
    if st.button("💾 Save current", use_container_width=True, key="strat_save_btn"):
        name_clean = name_input.strip()
        if not name_clean:
            st.warning("Enter a strategy name first.")
        elif name_clean in names:
            st.session_state._strat_overwrite_confirm = name_clean
            st.session_state.pop("_strat_delete_confirm", None)
        else:
            _do_save_strategy(name_clean)
            st.rerun()

    if st.session_state.get("_strat_overwrite_confirm"):
        oname = st.session_state._strat_overwrite_confirm
        st.warning(f"Overwrite **{oname}**?")
        o1, o2 = st.columns(2)
        if o1.button("Yes, overwrite", key="strat_overwrite_yes"):
            _do_save_strategy(oname)
            st.session_state.pop("_strat_overwrite_confirm", None)
            st.rerun()
        if o2.button("Cancel", key="strat_overwrite_no"):
            st.session_state.pop("_strat_overwrite_confirm", None)
            st.rerun()
