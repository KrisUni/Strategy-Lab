"""
ui/tabs/backtest.py
===================
Renders the Backtest tab content (tabs[0]).
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.backtest import BacktestEngine
from src.indicators import hpdr_bands
from src.strategy import SignalGenerator
from ui.helpers import params_to_strategy, calculate_beta_alpha
from ui.charts import (
    create_equity_chart,
    create_price_chart_with_trades,
    create_rsi_divergence_chart,
    create_bh_comparison_chart,
    PLOTLY_CONFIG,
)

BH_WINDOW_OPTIONS = {
    'full_window': 'Since start of data',
    'since_first_trade': 'Since first strategy trade',
}


def render_backtest_tab() -> None:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    st.session_state.capital = c1.number_input("Capital $", 1000, 1000000, st.session_state.capital, 1000)
    st.session_state.commission = c2.number_input(
        "Comm / side %",
        min_value=0.0,
        value=st.session_state.commission,
        step=0.01,
        format="%.2f",
        help="Applied on entry and exit. A 0.10% commission becomes 0.20% round-trip.",
    )
    st.session_state.slippage = c3.number_input(
        "Slip / fill %",
        min_value=0.0,
        value=st.session_state.slippage,
        step=0.01,
        format="%.2f",
        help="Applied on entries and non-gap-aware exits. No hard UI cap.",
    )
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Run", type="primary", use_container_width=True)

    st.caption(
        f"Round-trip commission: {st.session_state.commission * 2:.2f}% | "
        f"Slippage setting: {st.session_state.slippage:.2f}% per fill"
    )

    if run:
        if st.session_state.df is None:
            st.warning("Load data first!")
        else:
            with st.spinner("Running..."):
                engine = BacktestEngine(
                    params_to_strategy(st.session_state.params),
                    st.session_state.capital,
                    st.session_state.commission,
                    st.session_state.slippage,
                )
                st.session_state.backtest_results = engine.run(st.session_state.df.copy())
                st.success(f"✅ {st.session_state.backtest_results.num_trades} trades")

    # ── Summary metrics ───────────────────────────────────────────────────
    if st.session_state.backtest_results:
        r = st.session_state.backtest_results
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Return", f"{r.total_return_pct:.2f}%")
        c2.metric("CAGR", f"{r.cagr:.2f}%")
        c3.metric("Sharpe", f"{r.sharpe_ratio:.3f}")
        c4.metric("Sortino", f"{r.sortino_ratio:.3f}")
        c5.metric("Calmar", f"{r.calmar_ratio:.3f}")
        c6.metric("Max DD", f"{r.max_drawdown_pct:.2f}%")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades", r.num_trades)
        c2.metric("Win%", f"{r.win_rate:.1f}%")
        c3.metric("PF", f"{r.profit_factor:.2f}")
        c4.metric("Expectancy", f"${r.expectancy:.2f}")
        c5.metric("Payoff", f"{r.payoff_ratio:.2f}")
        c6.metric("Mkt Time", f"{r.pct_time_in_market:.0f}%")
        with st.expander("📊 Detailed Metrics", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Avg Win", f"${r.avg_winner:.2f}")
            c2.metric("Avg Loss", f"${abs(r.avg_loser):.2f}")
            c3.metric("Avg Win %", f"{r.avg_winner_pct:.2f}%")
            c4.metric("Avg Loss %", f"{abs(r.avg_loser_pct):.2f}%")
            c5.metric("Max Consec L", r.max_consecutive_losses)
            c6.metric("Max Consec W", r.max_consecutive_wins)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Bars", f"{r.avg_bars_held:.1f}")
            c2.metric("Longest DD", f"{r.longest_drawdown_bars} bars")
            c3.metric("Avg MAE", f"{r.avg_mae:.2f}%")
            c4.metric("Avg MFE", f"{r.avg_mfe:.2f}%")

    # ── Price chart (with optional HPDR overlay + strategy indicators) ───────
    if st.session_state.df is not None:
        p = st.session_state.params
        df_chart = st.session_state.df

        bands_data = None
        if p.get('hpdr_enabled'):
            try:
                bands_data = hpdr_bands(
                    df_chart['close'],
                    lookback=int(p.get('hpdr_lookback', 252)),
                    z_scores=(0.5, 1.0, 1.5, 2.0, 2.5),
                )
            except Exception as e:
                st.warning(f"HPDR bands error: {e}")

        # Calculate indicator values so enabled strategy indicators render on chart
        indicator_df = None
        try:
            sg = SignalGenerator(params_to_strategy(p))
            indicator_df = sg.calculate_indicators(df_chart.copy())
        except Exception:
            pass

        trades_to_show = st.session_state.backtest_results.trades if st.session_state.backtest_results else None
        st.plotly_chart(
            create_price_chart_with_trades(
                df_chart,
                trades_to_show,
                bands=bands_data,
                params=p,
                indicator_df=indicator_df,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            theme=None,
            key="backtest_price_chart",
        )

    # ── Equity & drawdown curves ──────────────────────────────────────────
    if st.session_state.backtest_results:
        st.plotly_chart(
            create_equity_chart(r),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            theme=None,
            key="backtest_equity_chart",
        )

    # ── RSI Hidden Divergence sub-panel ───────────────────────────────────
    if st.session_state.df is not None:
        p = st.session_state.params
        df_chart = st.session_state.df
        if p.get('rsi_div_enabled'):
            min_bars = 2 * (p['rsi_div_pivot_left'] + p['rsi_div_pivot_right'] + p['rsi_div_length'])
            if len(df_chart) < min_bars:
                st.warning(f"RSI Divergence needs at least {min_bars} bars. Load more data.")
            else:
                try:
                    st.markdown("##### 〰️ RSI Hidden Divergence")
                    st.caption(
                        f"RSI({p['rsi_div_length']}) · "
                        f"Pivot {p['rsi_div_pivot_left']}/{p['rsi_div_pivot_right']} · "
                        f"Signal lag = {p['rsi_div_pivot_right']} bars"
                    )
                    st.plotly_chart(
                        create_rsi_divergence_chart(df_chart, p),
                        use_container_width=True, config=PLOTLY_CONFIG,
                        theme=None, key="backtest_rsi_divergence_chart",
                    )
                except Exception as e:
                    st.warning(f"RSI Divergence chart error: {e}")

    # ── Strategy vs Buy & Hold ─────────────────────────────────────────────
    if st.session_state.backtest_results and st.session_state.df is not None:
        with st.expander("⚖️ Strategy vs Buy & Hold", expanded=False):
            _render_bh_comparison(st.session_state.backtest_results, st.session_state.df)

    # ── Trade log ─────────────────────────────────────────────────────────
    _render_trade_log()


def _calculate_curve_stats(curve: pd.Series, bars_per_year: int) -> dict[str, float]:
    """Calculate comparable performance stats for any equity-like curve."""
    if curve is None or len(curve) == 0:
        return {'return_pct': 0.0, 'cagr': 0.0, 'max_dd_pct': 0.0, 'sharpe': 0.0}

    start_val = float(curve.iloc[0])
    end_val = float(curve.iloc[-1])

    if start_val > 0:
        total_return_pct = (end_val / start_val - 1.0) * 100
    else:
        total_return_pct = 0.0

    n_bars = len(curve)
    if n_bars > 1 and start_val > 0 and end_val > 0:
        cagr = ((end_val / start_val) ** (bars_per_year / n_bars) - 1.0) * 100
    else:
        cagr = 0.0

    peak = curve.expanding().max()
    max_dd_pct = float(((curve - peak) / peak).min() * 100) if peak.max() > 0 else 0.0

    returns = curve.pct_change().dropna()
    active_returns = returns[returns != 0]
    n_total_rets = len(returns)
    n_active = len(active_returns)

    if n_total_rets > 0 and n_active > 0:
        active_bars_per_year = n_active * (bars_per_year / n_total_rets)
    else:
        active_bars_per_year = bars_per_year

    if n_active > 1 and active_returns.std() > 0:
        sharpe = float((active_returns.mean() / active_returns.std()) * np.sqrt(active_bars_per_year))
    else:
        sharpe = 0.0

    return {
        'return_pct': float(total_return_pct),
        'cagr': float(cagr),
        'max_dd_pct': max_dd_pct,
        'sharpe': sharpe,
    }


def _build_bh_comparison_window(r, df: pd.DataFrame, window_mode: str) -> dict | None:
    """Build aligned strategy and passive benchmark curves for the chosen window."""
    if df is None or df.empty or r.equity_curve is None or len(r.equity_curve) == 0:
        return None

    if window_mode == 'since_first_trade' and r.trades:
        window_label = BH_WINDOW_OPTIONS['since_first_trade']
        start_date = r.trades[0].entry_date
    else:
        window_label = BH_WINDOW_OPTIONS['full_window']
        start_date = df.index[0]

    strategy_curve = r.equity_curve.loc[r.equity_curve.index >= start_date]
    benchmark_prices = df.loc[df.index >= start_date, 'close']
    common_index = strategy_curve.index.intersection(benchmark_prices.index)

    if len(common_index) == 0:
        return None

    strategy_curve = strategy_curve.loc[common_index]
    benchmark_prices = benchmark_prices.loc[common_index]

    start_equity = float(strategy_curve.iloc[0])
    start_price = float(benchmark_prices.iloc[0])
    if start_equity <= 0 or start_price <= 0:
        return None

    benchmark_curve = benchmark_prices / start_price * start_equity

    return {
        'window_label': window_label,
        'start_date': common_index[0],
        'end_date': common_index[-1],
        'strategy_curve': strategy_curve,
        'benchmark_curve': benchmark_curve,
    }


def _render_trade_log() -> None:
    """Inline trade log — previously the standalone Trades tab."""
    if not (st.session_state.backtest_results and st.session_state.backtest_results.trades):
        return

    with st.expander("📋 Trade Log", expanded=False):
        trades = st.session_state.backtest_results.trades

        c1, c2 = st.columns(2)
        dir_f = c1.selectbox("Dir", ["All", "Long", "Short"])
        res_f = c2.selectbox("Result", ["All", "Winners", "Losers"])

        flt = trades
        if dir_f != "All":
            flt = [t for t in flt if t.direction.lower() == dir_f.lower()]
        if res_f == "Winners":
            flt = [t for t in flt if t.pnl > 0]
        elif res_f == "Losers":
            flt = [t for t in flt if t.pnl <= 0]

        if not flt:
            st.info("No trades match the current filters.")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Trades", len(flt))
        total_pnl = sum(t.pnl for t in flt)
        c2.metric("Total PnL", f"${total_pnl:,.2f}")
        c3.metric("Avg", f"${np.mean([t.pnl for t in flt]):.2f}")

        trade_df = pd.DataFrame([{
            'Entry': t.entry_date,
            'Exit': t.exit_date,
            'Dir': t.direction,
            'Entry$': round(t.entry_price, 2),
            'Exit$': round(t.exit_price, 2) if t.exit_price else None,
            'Size$': round(t.size_dollars, 0),
            'PnL$': round(t.pnl, 2),
            'PnL%': round(t.pnl_pct, 3),
            'Bars': t.bars_held,
            'MAE%': round(t.mae, 2),
            'MFE%': round(t.mfe, 2),
            'Reason': t.exit_reason,
        } for t in flt])

        st.download_button("📥 Export CSV", trade_df.to_csv(index=False), "trades.csv", use_container_width=True)
        st.dataframe(trade_df, use_container_width=True, hide_index=True)


def _render_bh_comparison(r, df: pd.DataFrame) -> None:
    """Inline Strategy vs Buy & Hold comparison panel."""
    available_modes = ['full_window']
    if r.trades:
        available_modes.append('since_first_trade')

    default_mode = st.session_state.get('bh_window_mode', available_modes[0])
    if default_mode not in available_modes:
        default_mode = available_modes[0]

    selected_mode = st.radio(
        "Benchmark Window",
        available_modes,
        format_func=lambda m: BH_WINDOW_OPTIONS[m],
        horizontal=True,
        index=available_modes.index(default_mode),
        key="bh_window_mode",
    )

    comparison = _build_bh_comparison_window(r, df, selected_mode)
    if comparison is None:
        st.warning("Unable to build the buy-and-hold comparison for the selected window.")
        return

    strategy_curve = comparison['strategy_curve']
    benchmark_curve = comparison['benchmark_curve']
    strategy_stats = _calculate_curve_stats(strategy_curve, r.bars_per_year)
    benchmark_stats = _calculate_curve_stats(benchmark_curve, r.bars_per_year)

    st.caption("Benchmark is always long-only buy & hold of the underlying asset, regardless of strategy direction.")
    st.caption(
        f"Window: {comparison['window_label']} · "
        f"{comparison['start_date'].date()} → {comparison['end_date'].date()}"
    )
    if not r.trades:
        st.caption("No trades were taken, so only the full-data comparison window is available.")

    st.dataframe(
        pd.DataFrame([
            {'Strategy': '📊 Strategy', 'Return %': f"{strategy_stats['return_pct']:.2f}%",
             'CAGR': f"{strategy_stats['cagr']:.2f}%", 'Max DD': f"{strategy_stats['max_dd_pct']:.2f}%",
             'Sharpe': f"{strategy_stats['sharpe']:.3f}"},
            {'Strategy': '📈 Buy & Hold (Long-only)', 'Return %': f"{benchmark_stats['return_pct']:.2f}%",
             'CAGR': f"{benchmark_stats['cagr']:.2f}%", 'Max DD': f"{benchmark_stats['max_dd_pct']:.2f}%",
             'Sharpe': f"{benchmark_stats['sharpe']:.3f}"},
        ]),
        use_container_width=True, hide_index=True,
    )

    diff = strategy_stats['return_pct'] - benchmark_stats['return_pct']
    msg = f"{'🏆 Strategy beats Buy & Hold by' if diff > 0 else '📉 B&H beats strategy by'} {abs(diff):.2f}%"
    (st.success if diff > 0 else st.warning)(msg)

    st.plotly_chart(
        create_bh_comparison_chart(
            strategy_curve,
            benchmark_curve,
            benchmark_name='Buy & Hold (Long-only)',
        ),
        use_container_width=True, config=PLOTLY_CONFIG,
        theme=None, key="backtest_bh_comparison_chart",
    )

    ba = calculate_beta_alpha(
        strategy_curve.pct_change().dropna(),
        benchmark_curve.pct_change().dropna(),
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Beta", f"{ba['beta']:.3f}")
    c2.metric("Alpha (ann)", f"{ba['alpha']:.2f}%")
    c3.metric("Correlation", f"{ba['correlation']:.3f}")
