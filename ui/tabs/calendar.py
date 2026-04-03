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
    create_quarterly_chart,
    create_yearly_chart,
    create_rolling_dow_chart,
    create_autocorr_chart,
    create_day_hour_heatmap,
)


def _sig_badge(p: float) -> str:
    if p < 0.01:
        return " 🟢 p<0.01"
    if p < 0.05:
        return " 🟡 p<0.05"
    if p < 0.10:
        return " 🟠 p<0.10"
    return " 🔴 not significant"


def render_calendar_tab() -> None:
    st.markdown("### 📅 Calendar Analytics")
    st.caption(
        "Calendar analytics reveals whether certain times of year, month, week, or day "
        "systematically produce better or worse returns — known as seasonal effects or calendar anomalies. "
        "The critical discipline throughout: always check whether a pattern is **statistically significant**. "
        "With small samples, almost any pattern can appear purely by chance."
    )

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

    tpy = cal.trading_days_per_year
    tpy_label = "365 days/yr — crypto/FX detected" if tpy == 365 else "252 days/yr — traditional markets"

    # ── Summary stats ─────────────────────────────────────────────────────────
    ss = cal.summary_stats
    if ss:
        st.markdown("#### Overview")
        st.caption(
            f"Headline numbers across the full dataset ({cal.start_date} → {cal.end_date}, "
            f"{cal.total_bars:,} bars). **Ann. Return** scales the average daily return to a full year "
            f"using **{tpy_label}** — it assumes continuous investment with no compounding."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Observations", f"{ss.get('total_observations', 0):,}",
                  help="Number of daily return data points used in the analysis.")
        c2.metric("Daily Win Rate", f"{ss.get('overall_win_rate', 0):.1f}%",
                  help="Percentage of all calendar days that closed higher than the previous close.")
        c3.metric("Best Day of Week", ss.get('best_day', '—'),
                  delta=f"avg {ss.get('best_day_avg', 0):+.4f}%",
                  help="The day of the week with the highest average daily return.")
        c4.metric("Best Month", ss.get('best_month', '—'),
                  delta=f"avg {ss.get('best_month_avg', 0):+.2f}%",
                  help="The calendar month with the highest average monthly return.")
        c5.metric("Ann. Return", f"{ss.get('annualized_return', 0):+.2f}%",
                  help=f"Mean daily return × {tpy} ({tpy_label}).")
        st.markdown("---")

    # ── Consecutive streaks ───────────────────────────────────────────────────
    st.markdown("#### 🔢 Win / Loss Streaks")
    st.caption(
        "How many consecutive days did the market close up or down without interruption? "
        "Long streaks matter for risk management: if the historical maximum loss streak is 12 days, "
        "you need to be financially and psychologically prepared for that to happen again. "
        "Short average streaks suggest frequent reversals (mean-reversion character); long ones suggest trending behaviour."
    )
    cons = cal.consecutive
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Max Win Streak", f"{cons.max_win_streak}d",
              help="Longest consecutive run of days that closed higher.")
    c2.metric("Max Loss Streak", f"{cons.max_loss_streak}d",
              help="Longest consecutive run of days that closed lower. Use this to size your drawdown tolerance.")
    c3.metric("Avg Win Streak", f"{cons.avg_win_streak:.1f}d",
              help="Average length of a winning run before it breaks.")
    c4.metric("Avg Loss Streak", f"{cons.avg_loss_streak:.1f}d",
              help="Average length of a losing run before it breaks.")
    cur = cons.current_streak
    c5.metric("Current Streak", f"{'🟢' if cur >= 0 else '🔴'} {abs(cur)}d",
        delta="winning" if cur >= 0 else "losing",
        delta_color="normal" if cur >= 0 else "inverse",
        help="The streak ending at the last bar in the dataset.")
    st.markdown("---")

    # ── Day-of-week ───────────────────────────────────────────────────────────
    st.markdown("#### 📆 Day-of-Week Returns")
    st.caption(
        "Does the market behave differently on Mondays vs Fridays? "
        "The top panel shows the average daily return for each day of the week. "
        "The bottom panel shows the win rate — what percentage of those days closed higher — "
        "with error bars representing the 95% confidence interval on that win rate. "
        "A wide error bar means the win rate estimate is unreliable due to limited data."
    )

    kw_p = cal.kruskal_wallis_p
    with st.expander("📖 How to read significance markers", expanded=False):
        st.markdown(
            """
**The most important number is the p-value shown on each bar.**

The p-value answers: *"If there were truly no day-of-week effect, how likely is it that I'd see a result this extreme by pure chance?"*

| Marker | p-value | Meaning |
|--------|---------|---------|
| `**` | < 0.01 | Less than 1% chance this is random noise — strong evidence of a real effect |
| `*` | < 0.05 | Less than 5% chance — conventional threshold for statistical significance |
| *(none)* | ≥ 0.05 | Cannot rule out that this is random — **do not trade this pattern** |

**Kruskal-Wallis test** (shown below): a single non-parametric test that asks "does *any* day differ significantly from the others?" It's more reliable than individual t-tests because it doesn't assume returns follow a normal distribution.

**Win rate confidence interval**: a 60% win rate from 15 observations has a 95% CI of roughly 35%–80% — it looks meaningful but is statistically indistinguishable from a coin flip. The error bars on the win rate panel show exactly this uncertainty.
            """
        )
    st.caption(
        f"**Kruskal-Wallis across all days: p = {kw_p:.4f}{_sig_badge(kw_p)}** — "
        f"{'at least one day differs significantly from the others.' if kw_p < 0.05 else 'no significant difference between days overall — treat all patterns here with scepticism.'}"
    )
    st.plotly_chart(create_dow_chart(cal.day_of_week_df),
                    use_container_width=True, config={'displayModeBar': False})
    with st.expander("📋 Day-of-Week Table", expanded=False):
        st.dataframe(cal.day_of_week_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── Monthly seasonality ───────────────────────────────────────────────────
    st.markdown("#### 🗓️ Monthly Seasonality")
    st.caption(
        "Classic seasonal stories — the January Effect, 'Sell in May and go away', the year-end rally — "
        "all show up here as monthly averages. Remember: an average over many years hides enormous year-to-year variation. "
        "Check the Year × Month heatmap below to see how consistent a pattern actually is across individual years. "
        "Bars marked `*` or `**` are statistically significant; unmarked bars are probably noise."
    )
    with st.expander("📖 Background: why do monthly patterns exist?", expanded=False):
        st.markdown(
            """
Several structural forces create monthly seasonality:

- **January Effect**: tax-loss selling in December pushes prices down; new-year buying in January pushes them back up. Strongest in small-cap stocks.
- **Sell in May**: institutional managers rotate out of equities into summer. The May–October period has historically underperformed November–April in many equity markets.
- **December / year-end rally**: window dressing (fund managers buying winners to look good in annual reports), holiday optimism, low volume amplifying moves.
- **Options expiry**: the third Friday of each month sees elevated volatility as derivatives expire.

For crypto these patterns are less established and tend to shift as the asset matures. The **Year × Month heatmap** is the honest view — it shows you whether January 2020, 2021, 2022, and 2023 were all actually positive, or whether it's just an average of mixed years.
            """
        )
    st.plotly_chart(create_monthly_bar_chart(cal.monthly_df),
                    use_container_width=True, config={'displayModeBar': False})
    if not cal.monthly_heatmap.empty:
        st.markdown("**Year × Month Heatmap**")
        st.caption(
            "Each cell shows the actual return for that specific month in that specific year. "
            "Green = positive, red = negative. A pattern is only trustworthy if the same colour "
            "appears consistently in the same column across most years."
        )
        st.plotly_chart(create_monthly_heatmap(cal.monthly_heatmap),
                        use_container_width=True, config={'displayModeBar': False})
    with st.expander("📋 Monthly Table", expanded=False):
        st.dataframe(cal.monthly_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── Quarterly seasonality ─────────────────────────────────────────────────
    if not cal.quarterly_df.empty:
        st.markdown("#### 📊 Quarterly Seasonality")
        st.caption(
            "Groups months into quarters: Q1 (Jan–Mar), Q2 (Apr–Jun), Q3 (Jul–Sep), Q4 (Oct–Dec). "
            "Q4 is historically strong for equities — year-end positioning and window dressing. "
            "Q3 (summer) is often the weakest. Quarters are averages of daily returns within them, "
            "so the Sharpe and p-values here are based on individual days, not quarterly totals."
        )
        st.plotly_chart(create_quarterly_chart(cal.quarterly_df),
                        use_container_width=True, config={'displayModeBar': False})
        with st.expander("📋 Quarterly Table", expanded=False):
            st.dataframe(cal.quarterly_df, use_container_width=True, hide_index=True)
        st.markdown("---")

    # ── Year-by-year performance ──────────────────────────────────────────────
    if not cal.yearly_df.empty:
        st.markdown("#### 📈 Year-by-Year Performance")
        st.caption(
            "Each bar is the **total compounded return** for that calendar year — how much $1 "
            "invested at the start of the year turned into by year-end. "
            "The Sharpe panel shows risk-adjusted performance per year. "
            "A **declining Sharpe over recent years** is a warning sign that an edge is fading. "
            "Max drawdown in the table shows how deep the worst peak-to-trough decline was within each year."
        )
        with st.expander("📖 What to look for here", expanded=False):
            st.markdown(
                """
**Consistency**: Are most years positive, or are a few exceptional years carrying the average?

**Sharpe trend**: If Sharpe was 1.5 in 2018–2020 and is now 0.4, the market has likely adapted and the edge is smaller. Don't backtest on the good years and assume the future looks like them.

**Max drawdown by year**: Compare this to total return. A year with +30% return but -25% drawdown is a much rougher ride than +15% with -8% drawdown. Use the yearly drawdown to calibrate position sizing — your worst year's drawdown is a preview of what a bad year feels like at full size.

**Sharpe ratio**: `(average daily return / std of daily returns) × √(trading days per year)`. Above 1.0 is generally considered good; above 2.0 is exceptional. Below 0.5 suggests the return barely compensates for the risk taken.
                """
            )
        st.plotly_chart(create_yearly_chart(cal.yearly_df),
                        use_container_width=True, config={'displayModeBar': False})
        with st.expander("📋 Yearly Table", expanded=False):
            st.dataframe(cal.yearly_df, use_container_width=True, hide_index=True)
        st.markdown("---")

    # ── DOW stability (year-over-year) ────────────────────────────────────────
    if not cal.rolling_dow_df.empty:
        st.markdown("#### 🔄 Day-of-Week Edge Stability")
        st.caption(
            "This is the most important chart for validating a day-of-week pattern. "
            "Each line shows the mean return for that day of the week, measured year by year. "
            "You want to see **consistent lines across all years** — not ones that reverse or drift toward zero. "
            "A pattern visible only in older years has likely been arbitraged away."
        )
        with st.expander("📖 Why stability matters more than averages", expanded=False):
            st.markdown(
                """
The day-of-week chart above shows the average over your entire history. That average could hide the fact that Monday was strongly positive in 2015–2019 and has been slightly negative every year since 2021.

If you built a strategy around the "Monday effect" using data up to 2019, you'd be trading a pattern that no longer exists.

**What to look for:**
- Lines that stay on the same side of zero across most years → potentially real, stable edge
- Lines that cross zero multiple times → unreliable, don't trade it
- Lines converging toward zero in recent years → edge is decaying, use caution
- Lines with high variance year-to-year → the effect is real but noisy, requires large position diversification

This is why statistical significance alone isn't enough — a pattern can be significant *in-sample* over 10 years but have been dead for the last 3.
                """
            )
        st.plotly_chart(create_rolling_dow_chart(cal.rolling_dow_df),
                        use_container_width=True, config={'displayModeBar': False})
        with st.expander("📋 Year × DOW Table", expanded=False):
            st.dataframe(cal.rolling_dow_df, use_container_width=True)
        st.markdown("---")

    # ── Day-of-month ──────────────────────────────────────────────────────────
    if not cal.day_of_month_df.empty:
        st.markdown("#### 📅 Day-of-Month Effect")
        st.caption(
            "Some days of the calendar month have structural drivers. "
            "Days 1–3 often see fresh capital deployment (monthly systematic buyers). "
            "Days 28–31 often see rebalancing flows as institutions adjust portfolios at month-end. "
            "The orange line shows win rate; bars show average return. "
            "Note: not every month has a 29th, 30th, or 31st, so sample sizes drop at month-end."
        )
        st.plotly_chart(create_dom_chart(cal.day_of_month_df),
                        use_container_width=True, config={'displayModeBar': False})
        with st.expander("📋 Day-of-Month Table", expanded=False):
            st.dataframe(cal.day_of_month_df, use_container_width=True, hide_index=True)
        st.markdown("---")

    # ── Hourly (intraday only) ────────────────────────────────────────────────
    if cal.is_intraday and cal.hourly_df is not None:
        st.markdown("#### ⏰ Hourly Returns")
        st.caption(
            "Average return per hour of day across all data. "
            "Market open and close are typically the highest-volatility hours — the biggest moves happen "
            "when the most participants are active. The midday period is often quieter and less directional. "
            "For crypto, these patterns reflect global handoff times between Asian, European, and US sessions."
        )
        st.plotly_chart(create_hourly_chart(cal.hourly_df),
                        use_container_width=True, config={'displayModeBar': False})
        with st.expander("📋 Hourly Table", expanded=False):
            st.dataframe(cal.hourly_df, use_container_width=True, hide_index=True)
        st.markdown("---")

    # ── Day × Hour heatmap (intraday only) ────────────────────────────────────
    if cal.is_intraday and cal.day_hour_df is not None:
        st.markdown("#### 🗺️ Day × Hour Return Heatmap")
        st.caption(
            "Combines day-of-week and hour-of-day into a single grid. Each cell shows the average return "
            "for that specific hour on that specific day. Green = historically positive, red = negative. "
            "Look for clusters of green to identify your highest-probability entry windows, "
            "and avoid scheduling entries during consistently red cells."
        )
        st.plotly_chart(create_day_hour_heatmap(cal.day_hour_df),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")

    # ── Return distribution ───────────────────────────────────────────────────
    dist = cal.distribution
    if dist.bins:
        st.markdown("#### 📊 Return Distribution")
        st.caption(
            "How are daily returns spread out? Most days cluster near zero, but the shape of the tails "
            "— the extreme left and right — determines your real risk. "
            "Financial markets almost universally have **fatter tails than a normal distribution** predicts, "
            "meaning large losses (and gains) happen more often than statistics textbooks would suggest."
        )
        with st.expander("📖 How to interpret each metric", expanded=False):
            st.markdown(
                f"""
**Mean & Std Dev**: Average daily return and how much it typically varies around that average. Most days will fall within ±1 standard deviation of the mean.

**Skewness**: Measures lopsidedness.
- Near 0 → roughly symmetric
- Negative → more frequent small gains, but occasional large losses (the "left tail is fatter") — typical for most financial assets
- Positive → more frequent small losses, occasional large gains (rare in practice)

**Excess Kurtosis**: Measures how fat the tails are compared to a normal distribution.
- 0 → exactly normal tails
- Positive → extreme events (both gains and losses) are more common than a normal distribution would predict — this is typical of markets
- High kurtosis means standard deviation *understates* real risk

**VaR 95% ({dist.var_95:+.3f}%)**: Value at Risk at 95% confidence. On your worst 5% of days (roughly 1 in 20 trading days, or about once a month), you would have lost *at least* this much. Think of it as "a realistic bad day."

**VaR 99% ({dist.var_99:+.3f}%)**: The worst 1% of days (roughly 2–3 times per year). "A serious bad day."

**CVaR 95% ({dist.cvar_95:+.3f}%)**: Also called Expected Shortfall. When you DO have one of those worst-5% days, this is the *average* loss across all of them. CVaR is more conservative and more useful than VaR because it tells you what happens in the tail, not just the threshold. Use this for position sizing — can you survive losing CVaR% in a single day?

**Jarque-Bera p-value ({dist.jarque_bera_p:.4f})**: Tests whether returns follow a normal bell curve. p < 0.05 means they don't — fat tails or skew are present. This matters because the standard Sharpe ratio *assumes* normality. If JB fails, prefer Sortino ratio and CVaR as your risk measures.

**VaR lines on chart**: The red dashed lines mark VaR 95% and 99%. Bars to the left of these lines are your tail-risk events.
                """
            )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Daily", f"{dist.mean:+.4f}%",
                  help="Average daily return across all observations.")
        c2.metric("Std Dev", f"{dist.std:.4f}%",
                  help="Typical day-to-day variation in returns. ~68% of days fall within ±1 std dev of the mean.")
        c3.metric("Skewness", f"{dist.skew:.3f}",
                  help="Negative = left tail heavier (more/larger losses than gains). Positive = right tail heavier.")
        c4.metric("Excess Kurtosis", f"{dist.kurtosis:.3f}",
                  help="Positive = fat tails (extreme events more common than normal distribution predicts). Most markets are > 0.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VaR 95%", f"{dist.var_95:.4f}%",
                  help="On your worst 5% of days, you lose at least this much. Roughly 1 bad day per month.")
        c2.metric("VaR 99%", f"{dist.var_99:.4f}%",
                  help="On your worst 1% of days, you lose at least this much. Roughly 2–3 times per year.")
        c3.metric("CVaR 95%", f"{dist.cvar_95:.4f}%",
                  help="When you have one of your worst 5% days, this is the average loss. More useful than VaR for sizing.")
        jb_normal = dist.jarque_bera_p > 0.05
        c4.metric("Jarque-Bera p", f"{dist.jarque_bera_p:.4f}",
                  delta="likely normal" if jb_normal else "non-normal (fat tails)",
                  delta_color="normal" if jb_normal else "inverse",
                  help="p > 0.05: returns are consistent with a normal distribution. p < 0.05: fat tails or skew present — use CVaR and Sortino instead of VaR and Sharpe.")

        st.plotly_chart(create_return_distribution_chart(dist),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")

    # ── Return autocorrelation ────────────────────────────────────────────────
    ac = cal.autocorr
    if ac.lags:
        st.markdown("#### 🔗 Return Autocorrelation")
        st.caption(
            "Does knowing today's return tell you anything about tomorrow's — or next week's? "
            "Each bar shows the correlation between returns separated by that many days. "
            "Bars outside the grey confidence band are statistically significant."
        )
        with st.expander("📖 How to read this chart", expanded=False):
            st.markdown(
                f"""
**What each bar means:**

- **Positive bar at lag 1**: a gain today tends to be followed by a gain tomorrow → **momentum** character
- **Negative bar at lag 1**: a gain today tends to be followed by a loss tomorrow → **mean-reversion** character
- **Bars near zero**: no predictable relationship at that time gap
- **Bars inside the grey band** (±{ac.conf_upper:.3f}): statistically indistinguishable from zero — no exploitable pattern at that lag
- **Bars outside the grey band**: statistically significant — something is there worth investigating

**Ljung-Box test**: combines all lags into one test.
- p < 0.05: there IS detectable structure in the return sequence — returns are not random from bar to bar
- p ≥ 0.05: returns appear independent — knowing yesterday tells you nothing about today

**Practical use**: Strong positive lag-1 autocorrelation supports momentum strategies (buy recent winners). Strong negative lag-1 supports mean-reversion strategies (buy recent losers). If all bars are in the grey band and Ljung-Box p > 0.05, returns are essentially unpredictable from their own history — you need other signals.

**Important caveat**: even a statistically significant autocorrelation can be too small to profit from after transaction costs. The economic significance (the actual return you'd earn by exploiting it) matters as much as the statistical significance.
                """
            )
        lb_p = ac.ljung_box_p
        st.caption(
            f"**Ljung-Box test (lags 1–{max(ac.lags)}): Q = {ac.ljung_box_stat:.2f}, "
            f"p = {lb_p:.4f}{_sig_badge(lb_p)}** — "
            f"{'significant autocorrelation detected: returns are not independent bar-to-bar.' if lb_p < 0.05 else 'no significant autocorrelation: returns appear independent (IID).'}"
        )
        st.plotly_chart(create_autocorr_chart(ac),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")

    # ── Trade calendar ────────────────────────────────────────────────────────
    tc = st.session_state.get('_trade_calendar')
    if tc and not tc.trades_by_day.empty:
        st.markdown("#### 🎯 Strategy Trade Calendar")
        st.caption(
            "Unlike the market sections above — which analyse raw price returns — this section analyses "
            "your **strategy's actual trades**. It answers: do your wins and losses concentrate on specific "
            "days of the week or months of the year? A strategy that loses disproportionately on Mondays "
            "might be hurt by the weekend gap. High win rate in Q4 but low in Q2 might suggest adding "
            "a seasonal filter to your entry rules. The WR 95% CI column shows the confidence interval on "
            "each win rate — small trade counts will have very wide bands."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**By Entry Day of Week**")
            st.dataframe(tc.trades_by_day, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**By Entry Month**")
            st.dataframe(tc.trades_by_month, use_container_width=True, hide_index=True)
