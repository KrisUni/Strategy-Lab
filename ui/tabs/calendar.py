"""
ui/tabs/calendar.py
===================
Renders the Calendar Analytics tab content (tabs[4]).
"""

import streamlit as st

from src.analytics import analyze_calendar, analyze_trade_calendar
from ui.charts import (
    create_dow_chart,
    create_monthly_bar_chart,
    create_monthly_heatmap,
    create_dom_chart,
    create_hourly_chart,
    create_return_distribution_chart,
)


def render_calendar_tab() -> None:
    st.markdown("### 📅 Calendar Analytics")

    if st.session_state.df is None:
        st.info("Load data first.")
        return

    if st.button("📅 Analyze Calendar", use_container_width=True):
        with st.spinner("Analyzing..."):
            cal = analyze_calendar(st.session_state.df)
            st.session_state._calendar = cal
            if st.session_state.backtest_results and st.session_state.backtest_results.trades:
                st.session_state._trade_calendar = analyze_trade_calendar(
                    st.session_state.backtest_results.trades)

    cal = st.session_state.get('_calendar')
    if not cal:
        return

    # ── Summary stats ─────────────────────────────────────────────────────────
    ss = cal.summary_stats
    if ss:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Observations", f"{ss.get('total_observations', 0):,}")
        c2.metric("Daily Win Rate", f"{ss.get('overall_win_rate', 0):.1f}%")
        c3.metric("Best Day", ss.get('best_day', '—'), delta=f"avg {ss.get('best_day_avg', 0):+.4f}%")
        c4.metric("Best Month", ss.get('best_month', '—'), delta=f"avg {ss.get('best_month_avg', 0):+.2f}%")
        c5.metric("Ann. Return (daily avg)", f"{ss.get('annualized_return', 0):+.2f}%")
        st.markdown("---")

    # ── Consecutive streaks ───────────────────────────────────────────────────
    cons = cal.consecutive
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Max Win Streak", f"{cons.max_win_streak}d")
    c2.metric("Max Loss Streak", f"{cons.max_loss_streak}d")
    c3.metric("Avg Win Streak", f"{cons.avg_win_streak:.1f}d")
    c4.metric("Avg Loss Streak", f"{cons.avg_loss_streak:.1f}d")
    cur = cons.current_streak
    c5.metric("Current Streak", f"{'🟢' if cur >= 0 else '🔴'} {abs(cur)}d",
        delta="winning" if cur >= 0 else "losing",
        delta_color="normal" if cur >= 0 else "inverse")
    st.markdown("---")

    # ── Day-of-week ───────────────────────────────────────────────────────────
    st.markdown("#### 📆 Day-of-Week Returns")
    st.plotly_chart(create_dow_chart(cal.day_of_week_df),
                    use_container_width=True, config={'displayModeBar': False})
    with st.expander("📋 Day-of-Week Table", expanded=False):
        st.dataframe(cal.day_of_week_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── Monthly seasonality ───────────────────────────────────────────────────
    st.markdown("#### 🗓️ Monthly Seasonality")
    st.plotly_chart(create_monthly_bar_chart(cal.monthly_df),
                    use_container_width=True, config={'displayModeBar': False})
    if not cal.monthly_heatmap.empty:
        st.markdown("**Year × Month Heatmap**")
        st.plotly_chart(create_monthly_heatmap(cal.monthly_heatmap),
                        use_container_width=True, config={'displayModeBar': False})
    with st.expander("📋 Monthly Table", expanded=False):
        st.dataframe(cal.monthly_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── Day-of-month ──────────────────────────────────────────────────────────
    if not cal.day_of_month_df.empty:
        st.markdown("#### 📅 Day-of-Month Effect")
        st.plotly_chart(create_dom_chart(cal.day_of_month_df),
                        use_container_width=True, config={'displayModeBar': False})
        with st.expander("📋 Day-of-Month Table", expanded=False):
            st.dataframe(cal.day_of_month_df, use_container_width=True, hide_index=True)
        st.markdown("---")

    # ── Hourly (intraday only) ────────────────────────────────────────────────
    if cal.is_intraday and cal.hourly_df is not None:
        st.markdown("#### ⏰ Hourly Returns")
        st.plotly_chart(create_hourly_chart(cal.hourly_df),
                        use_container_width=True, config={'displayModeBar': False})
        with st.expander("📋 Hourly Table", expanded=False):
            st.dataframe(cal.hourly_df, use_container_width=True, hide_index=True)
        st.markdown("---")

    # ── Return distribution ───────────────────────────────────────────────────
    dist = cal.distribution
    if dist.bins:
        st.markdown("#### 📊 Return Distribution")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Daily", f"{dist.mean:+.4f}%")
        c2.metric("Std Dev", f"{dist.std:.4f}%")
        c3.metric("Skewness", f"{dist.skew:.3f}", help="Negative = left tail heavier")
        c4.metric("Excess Kurtosis", f"{dist.kurtosis:.3f}", help="Positive = fat tails")
        st.plotly_chart(create_return_distribution_chart(dist),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")

    # ── Trade calendar ────────────────────────────────────────────────────────
    tc = st.session_state.get('_trade_calendar')
    if tc and not tc.trades_by_day.empty:
        st.markdown("#### 🎯 Strategy Trade Calendar")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**By Entry Day**")
            st.dataframe(tc.trades_by_day, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**By Entry Month**")
            st.dataframe(tc.trades_by_month, use_container_width=True, hide_index=True)
