"""ui.advisor.context — assemble read-only session context for the AI advisor."""
from __future__ import annotations

import streamlit as st

from ui.helpers import get_active_filters_display
from src.mcp_server import get_research_history


def assemble_context() -> str:
    """
    Build a plain-text snapshot of current session state.

    Injected as a synthetic prefix to the first user message each API call.
    Never stored in advisor_messages. Never fabricates a metric.
    """
    parts: list[str] = []

    df = st.session_state.get("df")
    symbol = st.session_state.get("loaded_symbol")
    interval = st.session_state.get("loaded_interval")

    # ── Data ──────────────────────────────────────────────────────────────────
    if df is None:
        parts.append("=== DATA ===\nNo data loaded.")
    else:
        sym_str = (
            symbol
            or df.attrs.get("symbol")
            or df.attrs.get("ticker")
            or "unknown"
        )
        ivl_str = interval or "unknown"
        date_from = str(df.index[0].date()) if len(df) > 0 else "?"
        date_to = str(df.index[-1].date()) if len(df) > 0 else "?"
        parts.append(
            f"=== DATA ===\n"
            f"Symbol:   {sym_str}\n"
            f"Interval: {ivl_str}\n"
            f"Bars:     {len(df):,}\n"
            f"Range:    {date_from} → {date_to}"
        )

    # ── Active indicators ──────────────────────────────────────────────────────
    params = st.session_state.get("params", {})
    active = get_active_filters_display(params)
    if active:
        parts.append(
            "=== ACTIVE INDICATORS ===\n"
            + "\n".join(f"  • {ind}" for ind in active)
        )
    else:
        parts.append("=== ACTIVE INDICATORS ===\nNone enabled.")

    # ── Execution costs ────────────────────────────────────────────────────────
    capital = st.session_state.get("capital", "N/A")
    commission = st.session_state.get("commission", "N/A")
    slippage = st.session_state.get("slippage", "N/A")
    parts.append(
        f"=== EXECUTION COSTS ===\n"
        f"Capital:    {capital}\n"
        f"Commission: {commission}%\n"
        f"Slippage:   {slippage}%"
    )

    # ── Strategy params ────────────────────────────────────────────────────────
    if params:
        lines = ["=== STRATEGY PARAMS ==="]
        for k, v in params.items():
            lines.append(f"  {k}: {v}")
        parts.append("\n".join(lines))
    else:
        parts.append("=== STRATEGY PARAMS ===\nNone set.")

    # ── Backtest metrics ────────────────────────────────────────────────────────
    results = st.session_state.get("backtest_results")
    if results is None:
        parts.append("=== BACKTEST METRICS ===\nNo backtest run yet.")
    else:
        r = results
        lines = [
            "=== BACKTEST METRICS ===",
            f"  Trades:             {r.num_trades}",
            f"  Total return:       {r.total_return_pct:.2f}%",
            f"  CAGR:               {r.cagr:.2f}%",
            f"  Sharpe ratio:       {r.sharpe_ratio:.3f}",
            f"  Sortino ratio:      {r.sortino_ratio:.3f}",
            f"  Calmar ratio:       {r.calmar_ratio:.3f}",
            f"  Max drawdown:       {r.max_drawdown_pct:.2f}%",
            f"  Win rate:           {r.win_rate:.1f}%",
            f"  Profit factor:      {r.profit_factor:.3f}",
            f"  Expectancy:         {r.expectancy:.4f}",
            f"  Payoff ratio:       {r.payoff_ratio:.3f}",
            f"  % time in market:   {r.pct_time_in_market:.1f}%",
            f"  Avg winner:         {r.avg_winner_pct:.2f}%",
            f"  Avg loser:          {r.avg_loser_pct:.2f}%",
            f"  Avg bars held:      {r.avg_bars_held:.1f}",
            f"  Max consec losses:  {r.max_consecutive_losses}",
            f"  Max consec wins:    {r.max_consecutive_wins}",
            f"  Longest DD bars:    {r.longest_drawdown_bars}",
            f"  Avg MFE:            {r.avg_mfe:.2f}%",
            f"  Avg MAE:            {r.avg_mae:.2f}%",
        ]
        parts.append("\n".join(lines))

    # ── Research history ───────────────────────────────────────────────────────
    sym_for_history = symbol or (
        df.attrs.get("symbol") or df.attrs.get("ticker") if df is not None else None
    )
    # Fetch all entries to surface real total for multiple-comparisons warning;
    # display only the last 5 to keep context compact.
    full_history = get_research_history(symbol=sym_for_history, last_n=9999)
    all_entries = full_history.get("entries", [])
    total_tested = len(all_entries)
    display_entries = all_entries[:5]
    if not all_entries:
        note = full_history.get("note", "No research history for this symbol.")
        parts.append(f"=== RESEARCH HISTORY ===\nTotal strategies tested: 0\n{note}")
    else:
        lines = [
            f"=== RESEARCH HISTORY ===",
            f"Total strategies tested: {total_tested}",
            f"(showing last {len(display_entries)} of {total_tested})",
        ]
        for e in display_entries:
            verdict = e.get("verdict", "?")
            sym_e = e.get("symbol", "?")
            ivl_e = e.get("interval", "")
            thesis = e.get("thesis", "no thesis recorded")
            lines.append(f"  [{verdict}] {sym_e} {ivl_e} — {thesis}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)
