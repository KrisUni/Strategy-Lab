"""
Calendar Analytics Module — v2.0.0
===================================
Statistically rigorous calendar seasonality analysis.

New in v2.0:
  - p-values (one-sample t-test, H0: mean=0) on every day/month/quarter mean
  - Kruskal-Wallis test across all day-of-week groups (non-parametric)
  - Wilson 95% confidence intervals on win rates
  - Annualised Sharpe and Sortino per day / month / quarter
  - VaR 95%/99% and CVaR 95% (Expected Shortfall) in return distribution
  - Jarque-Bera normality test on daily returns
  - Quarterly seasonality (Q1–Q4)
  - Year-by-year performance table (total return, Sharpe, max drawdown)
  - Year-over-year DOW stability pivot (is the edge decaying?)
  - Return autocorrelation lags 1–10 + Ljung-Box test
  - Day × Hour heatmap for intraday data
  - Auto-detect trading days per year (252 vs 365 for crypto)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

DAY_NAMES     = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTH_NAMES   = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
QUARTER_NAMES = ['Q1', 'Q2', 'Q3', 'Q4']


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DayOfWeekStats:
    day_name: str
    count: int
    mean_return_pct: float
    median_return_pct: float
    std_return_pct: float
    positive_pct: float
    win_rate_ci_low: float        # Wilson 95% CI lower bound (%)
    win_rate_ci_high: float       # Wilson 95% CI upper bound (%)
    avg_positive_pct: float
    avg_negative_pct: float
    best_pct: float
    worst_pct: float
    total_return_pct: float
    sharpe_annualised: float      # (mean / std) * sqrt(tpy)
    sortino_annualised: float     # (mean / downside_std) * sqrt(tpy)
    t_stat: float                 # one-sample t vs H0: mean=0
    p_value: float                # two-sided p-value


@dataclass
class MonthStats:
    month_name: str
    count: int
    mean_return_pct: float
    median_return_pct: float
    std_return_pct: float
    positive_pct: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    best_pct: float
    worst_pct: float
    total_return_pct: float
    sharpe_annualised: float
    sortino_annualised: float
    t_stat: float
    p_value: float


@dataclass
class ReturnDistribution:
    """Histogram + tail risk + normality test."""
    bins: List[float]
    counts: List[int]
    mean: float
    std: float
    skew: float
    kurtosis: float
    pct_positive: float
    var_95: float                 # 5th percentile (VaR at 95% confidence)
    var_99: float                 # 1st percentile (VaR at 99% confidence)
    cvar_95: float                # Expected Shortfall at 95% confidence
    jarque_bera_stat: float       # JB test statistic
    jarque_bera_p: float          # p-value: low → reject normality


@dataclass
class ConsecutiveStats:
    """Win/loss streak statistics."""
    max_win_streak: int
    max_loss_streak: int
    avg_win_streak: float
    avg_loss_streak: float
    current_streak: int           # positive = win streak, negative = loss streak


@dataclass
class AutocorrStats:
    """Return autocorrelation for lags 1–N and Ljung-Box test."""
    lags: List[int]
    acf_values: List[float]
    conf_upper: float             # +1.96 / sqrt(n)
    conf_lower: float             # -1.96 / sqrt(n)
    ljung_box_stat: float
    ljung_box_p: float            # p-value: low → significant autocorrelation


@dataclass
class CalendarAnalysis:
    day_of_week: List[DayOfWeekStats]
    day_of_week_df: pd.DataFrame
    monthly: List[MonthStats]
    monthly_df: pd.DataFrame
    monthly_heatmap: pd.DataFrame
    quarterly_df: pd.DataFrame
    yearly_df: pd.DataFrame
    rolling_dow_df: pd.DataFrame          # Year × DOW stability pivot
    day_of_month_df: pd.DataFrame
    hourly_df: Optional[pd.DataFrame]
    day_hour_df: Optional[pd.DataFrame]   # DOW × Hour heatmap (intraday)
    distribution: ReturnDistribution
    consecutive: ConsecutiveStats
    autocorr: AutocorrStats
    kruskal_wallis_p: float               # overall DOW significance
    trading_days_per_year: int            # 252 or 365
    summary_stats: Dict
    symbol: str
    start_date: str
    end_date: str
    total_bars: int
    is_intraday: bool


@dataclass
class TradeCalendarAnalysis:
    trades_by_day: pd.DataFrame
    trades_by_month: pd.DataFrame


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_intraday(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    deltas = pd.Series(df.index).diff().dropna()
    median_td = deltas.median()
    if pd.isna(median_td):
        return False
    return median_td.total_seconds() < 86_400


def _safe_month_resample(series: pd.Series) -> pd.Series:
    try:
        return series.resample('ME').last().dropna()
    except ValueError:
        return series.resample('M').last().dropna()


def _resample_to_daily(df: pd.DataFrame) -> pd.Series:
    """
    Daily close-to-close returns (%) from any-frequency OHLCV data.
    resample('D').last() → NaN on days with no bars → dropna() removes them.
    pct_change() then correctly skips gaps (weekends, holidays).
    """
    if df.empty or 'close' not in df.columns:
        return pd.Series(dtype=float)
    daily_close = df['close'].resample('D').last().dropna()
    return daily_close.pct_change().dropna() * 100


def _detect_trading_days_per_year(returns: pd.Series) -> int:
    """Return 365 if weekend data is present (crypto/FX), else 252."""
    if returns.empty:
        return 252
    return 365 if (returns.index.dayofweek >= 5).any() else 252


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion. Returns (low%, high%)."""
    if n == 0:
        return 0.0, 100.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, (center - margin) * 100), min(100.0, (center + margin) * 100)


def _sortino(mean: float, day_rets: pd.Series, tpy: int) -> float:
    downside = day_rets[day_rets < 0]
    ds_std = float(downside.std()) if len(downside) > 1 else 0.0
    return round((mean / ds_std) * np.sqrt(tpy), 3) if ds_std > 1e-9 else 0.0


def _day_stats_row(day_rets: pd.Series, tpy: int) -> dict:
    """Compute all per-bucket stats shared across DOW/quarterly helpers."""
    n = len(day_rets)
    mean = float(day_rets.mean())
    std = float(day_rets.std()) if n > 1 else 0.0
    positive = day_rets[day_rets > 0]
    negative = day_rets[day_rets <= 0]
    n_pos = len(positive)
    win_rate = n_pos / n * 100
    ci_low, ci_high = _wilson_ci(n_pos, n)
    sharpe = round((mean / std) * np.sqrt(tpy), 3) if std > 1e-9 else 0.0
    sortino = _sortino(mean, day_rets, tpy)
    if n > 1:
        t_stat, p_value = scipy_stats.ttest_1samp(day_rets.values, 0.0)
    else:
        t_stat, p_value = 0.0, 1.0
    return dict(
        count=n, mean=mean, median=float(day_rets.median()),
        std=std, win_rate=win_rate, ci_low=ci_low, ci_high=ci_high,
        avg_pos=float(positive.mean()) if n_pos > 0 else 0.0,
        avg_neg=float(negative.mean()) if len(negative) > 0 else 0.0,
        best=float(day_rets.max()), worst=float(day_rets.min()),
        total=float(day_rets.sum()),
        sharpe=sharpe, sortino=sortino,
        t_stat=round(float(t_stat), 4), p_value=round(float(p_value), 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Compute functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_day_of_week_stats(
    df: pd.DataFrame, tpy: int = 252,
) -> Tuple[List[DayOfWeekStats], pd.DataFrame, float]:
    """
    Returns (stats_list, dataframe, kruskal_wallis_p).
    Kruskal-Wallis p-value tests whether any DOW has a significantly
    different distribution from the others (non-parametric, no normality needed).
    """
    if df.empty:
        return [], pd.DataFrame(), 1.0

    returns = _resample_to_daily(df)
    if returns.empty:
        return [], pd.DataFrame(), 1.0

    dow = returns.index.dayofweek
    stats_list: List[DayOfWeekStats] = []
    rows = []
    groups = []

    for day_idx in range(7):
        day_rets = returns[dow == day_idx]
        if len(day_rets) == 0:
            continue
        if len(day_rets) >= 3:
            groups.append(day_rets.values)

        r = _day_stats_row(day_rets, tpy)
        s = DayOfWeekStats(
            day_name=DAY_NAMES[day_idx],
            count=r['count'],
            mean_return_pct=round(r['mean'], 5),
            median_return_pct=round(r['median'], 5),
            std_return_pct=round(r['std'], 5),
            positive_pct=round(r['win_rate'], 2),
            win_rate_ci_low=round(r['ci_low'], 1),
            win_rate_ci_high=round(r['ci_high'], 1),
            avg_positive_pct=round(r['avg_pos'], 5),
            avg_negative_pct=round(r['avg_neg'], 5),
            best_pct=round(r['best'], 3),
            worst_pct=round(r['worst'], 3),
            total_return_pct=round(r['total'], 3),
            sharpe_annualised=r['sharpe'],
            sortino_annualised=r['sortino'],
            t_stat=r['t_stat'],
            p_value=r['p_value'],
        )
        stats_list.append(s)
        sig = '**' if s.p_value < 0.01 else ('*' if s.p_value < 0.05 else '')
        rows.append({
            'Day':              s.day_name,
            'Observations':     s.count,
            'Avg %':            s.mean_return_pct,
            'Median %':         s.median_return_pct,
            'Std %':            s.std_return_pct,
            'Win Rate':         f"{s.positive_pct:.1f}%",
            'WR 95% CI':        f"{s.win_rate_ci_low:.0f}%–{s.win_rate_ci_high:.0f}%",
            'WR CI Low':        s.win_rate_ci_low,
            'WR CI High':       s.win_rate_ci_high,
            'Avg Win %':        s.avg_positive_pct,
            'Avg Loss %':       s.avg_negative_pct,
            'Best %':           s.best_pct,
            'Worst %':          s.worst_pct,
            'Total %':          s.total_return_pct,
            'Sharpe (ann.)':    s.sharpe_annualised,
            'Sortino (ann.)':   s.sortino_annualised,
            't-stat':           s.t_stat,
            'p-value':          s.p_value,
            'Sig':              sig,
        })

    kw_p = 1.0
    if len(groups) >= 2:
        try:
            _, kw_p = scipy_stats.kruskal(*groups)
            kw_p = round(float(kw_p), 4)
        except Exception:
            pass

    df_out = pd.DataFrame(rows) if rows else pd.DataFrame()
    return stats_list, df_out, kw_p


def compute_monthly_stats(
    df: pd.DataFrame, tpy: int = 252,
) -> Tuple[List[MonthStats], pd.DataFrame]:
    if df.empty:
        return [], pd.DataFrame()

    monthly_close = _safe_month_resample(df['close'])
    monthly_returns = monthly_close.pct_change().dropna() * 100

    if monthly_returns.empty:
        return [], pd.DataFrame()

    month_nums = monthly_returns.index.month
    # Use monthly tpy: ~12 months/year
    stats_list: List[MonthStats] = []
    rows = []

    for m in range(1, 13):
        m_rets = monthly_returns[month_nums == m]
        if len(m_rets) == 0:
            continue
        r = _day_stats_row(m_rets, 12)   # 12 months per year for annualisation
        s = MonthStats(
            month_name=MONTH_NAMES[m - 1],
            count=r['count'],
            mean_return_pct=round(r['mean'], 3),
            median_return_pct=round(r['median'], 3),
            std_return_pct=round(r['std'], 3),
            positive_pct=round(r['win_rate'], 1),
            win_rate_ci_low=round(r['ci_low'], 1),
            win_rate_ci_high=round(r['ci_high'], 1),
            best_pct=round(r['best'], 2),
            worst_pct=round(r['worst'], 2),
            total_return_pct=round(r['total'], 2),
            sharpe_annualised=r['sharpe'],
            sortino_annualised=r['sortino'],
            t_stat=r['t_stat'],
            p_value=r['p_value'],
        )
        stats_list.append(s)
        sig = '**' if s.p_value < 0.01 else ('*' if s.p_value < 0.05 else '')
        rows.append({
            'Month':            s.month_name,
            'Years':            s.count,
            'Avg %':            s.mean_return_pct,
            'Median %':         s.median_return_pct,
            'Std %':            s.std_return_pct,
            'Win Rate':         f"{s.positive_pct:.1f}%",
            'WR 95% CI':        f"{s.win_rate_ci_low:.0f}%–{s.win_rate_ci_high:.0f}%",
            'Best %':           s.best_pct,
            'Worst %':          s.worst_pct,
            'Total %':          s.total_return_pct,
            'Sharpe (ann.)':    s.sharpe_annualised,
            'Sortino (ann.)':   s.sortino_annualised,
            't-stat':           s.t_stat,
            'p-value':          s.p_value,
            'Sig':              sig,
        })

    df_out = pd.DataFrame(rows) if rows else pd.DataFrame()
    return stats_list, df_out


def compute_monthly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    monthly_close = _safe_month_resample(df['close'])
    monthly_returns = monthly_close.pct_change().dropna() * 100

    if monthly_returns.empty:
        return pd.DataFrame()

    heatmap = (
        monthly_returns
        .groupby([monthly_returns.index.year, monthly_returns.index.month])
        .first()
        .unstack(level=1)
        .round(2)
    )
    heatmap.columns = [MONTH_NAMES[m - 1] for m in heatmap.columns]
    heatmap.index.name = 'Year'
    return heatmap


def compute_quarterly_stats(df: pd.DataFrame, tpy: int = 252) -> pd.DataFrame:
    """Average daily returns grouped by calendar quarter (Q1–Q4)."""
    if df.empty:
        return pd.DataFrame()

    returns = _resample_to_daily(df)
    if returns.empty:
        return pd.DataFrame()

    quarters = returns.index.quarter
    rows = []

    for q in range(1, 5):
        q_rets = returns[quarters == q]
        if len(q_rets) == 0:
            continue
        r = _day_stats_row(q_rets, tpy)
        sig = '**' if r['p_value'] < 0.01 else ('*' if r['p_value'] < 0.05 else '')
        rows.append({
            'Quarter':          QUARTER_NAMES[q - 1],
            'Observations':     r['count'],
            'Avg %':            round(r['mean'], 4),
            'Median %':         round(r['median'], 4),
            'Std %':            round(r['std'], 4),
            'Win Rate':         f"{r['win_rate']:.1f}%",
            'WR 95% CI':        f"{r['ci_low']:.0f}%–{r['ci_high']:.0f}%",
            'Avg Win %':        round(r['avg_pos'], 4),
            'Avg Loss %':       round(r['avg_neg'], 4),
            'Best %':           round(r['best'], 3),
            'Worst %':          round(r['worst'], 3),
            'Total %':          round(r['total'], 2),
            'Sharpe (ann.)':    r['sharpe'],
            'Sortino (ann.)':   r['sortino'],
            't-stat':           r['t_stat'],
            'p-value':          r['p_value'],
            'Sig':              sig,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_yearly_stats(df: pd.DataFrame, tpy: int = 252) -> pd.DataFrame:
    """Year-by-year performance: compound return, Sharpe, max drawdown."""
    if df.empty:
        return pd.DataFrame()

    returns = _resample_to_daily(df)
    if returns.empty:
        return pd.DataFrame()

    years = sorted(returns.index.year.unique())
    rows = []

    for y in years:
        y_rets = returns[returns.index.year == y]
        if len(y_rets) < 2:
            continue
        mean = float(y_rets.mean())
        std = float(y_rets.std())
        n_pos = int((y_rets > 0).sum())
        win_rate = n_pos / len(y_rets) * 100
        compound = float(((1 + y_rets / 100).prod() - 1) * 100)
        cum = (1 + y_rets / 100).cumprod()
        mdd = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
        sharpe = round((mean / std) * np.sqrt(tpy), 3) if std > 1e-9 else 0.0
        sortino = _sortino(mean, y_rets, tpy)
        ci_low, ci_high = _wilson_ci(n_pos, len(y_rets))
        rows.append({
            'Year':             y,
            'Total Return %':   round(compound, 2),
            'Avg Daily %':      round(mean, 4),
            'Std %':            round(std, 4),
            'Win Rate':         f"{win_rate:.1f}%",
            'WR 95% CI':        f"{ci_low:.0f}%–{ci_high:.0f}%",
            'Sharpe (ann.)':    sharpe,
            'Sortino (ann.)':   sortino,
            'Max DD %':         round(mdd, 2),
            'Days':             len(y_rets),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_rolling_dow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year × DOW pivot of mean daily returns.
    Shows whether day-of-week edges are stable or decaying over time.
    Only includes DOWs that have at least one year of data.
    """
    if df.empty:
        return pd.DataFrame()

    returns = _resample_to_daily(df)
    if returns.empty:
        return pd.DataFrame()

    tmp = pd.DataFrame({
        'ret': returns.values,
        'year': returns.index.year,
        'dow': returns.index.dayofweek,
    })

    pivot = (
        tmp.groupby(['year', 'dow'])['ret']
        .mean()
        .unstack('dow')
        .round(4)
    )
    pivot.columns = [DAY_NAMES[d] for d in pivot.columns]
    pivot.index.name = 'Year'
    return pivot


def compute_day_of_month_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    returns = _resample_to_daily(df)
    if returns.empty:
        return pd.DataFrame()

    dom = returns.index.day
    rows = []

    for d in range(1, 32):
        d_rets = returns[dom == d]
        if len(d_rets) == 0:
            continue
        rows.append({
            'Day of Month': d,
            'Observations': len(d_rets),
            'Avg %':        round(float(d_rets.mean()), 4),
            'Win Rate':     f"{float((d_rets > 0).sum() / len(d_rets) * 100):.1f}%",
            'Std %':        round(float(d_rets.std()), 4) if len(d_rets) > 1 else 0.0,
            'Best %':       round(float(d_rets.max()), 3),
            'Worst %':      round(float(d_rets.min()), 3),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_hourly_stats(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Average return by hour of day (intraday data only)."""
    if not _is_intraday(df) or df.empty:
        return None

    hourly_close = df['close'].resample('1h').last().dropna()
    if len(hourly_close) < 2:
        return None

    returns = hourly_close.pct_change().dropna() * 100
    hours = returns.index.hour
    rows = []

    for h in sorted(hours.unique()):
        h_rets = returns[hours == h]
        if len(h_rets) == 0:
            continue
        rows.append({
            'Hour':         f"{h:02d}:00",
            'Observations': len(h_rets),
            'Avg %':        round(float(h_rets.mean()), 4),
            'Win Rate':     f"{float((h_rets > 0).sum() / len(h_rets) * 100):.1f}%",
            'Std %':        round(float(h_rets.std()), 4) if len(h_rets) > 1 else 0.0,
            'Best %':       round(float(h_rets.max()), 3),
            'Worst %':      round(float(h_rets.min()), 3),
        })
    return pd.DataFrame(rows) if rows else None


def compute_day_hour_heatmap(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Day-of-week × Hour mean return heatmap (intraday only).
    Returns a pivot table: rows = day names, columns = hours.
    """
    if not _is_intraday(df) or df.empty:
        return None

    hourly_close = df['close'].resample('1h').last().dropna()
    if len(hourly_close) < 2:
        return None

    returns = hourly_close.pct_change().dropna() * 100
    tmp = pd.DataFrame({
        'ret': returns.values,
        'dow': returns.index.dayofweek,
        'hour': returns.index.hour,
    })

    pivot = (
        tmp.groupby(['dow', 'hour'])['ret']
        .mean()
        .unstack('hour')
        .round(4)
    )
    pivot.index = [DAY_NAMES[d] for d in pivot.index]
    pivot.index.name = 'Day'
    return pivot


def compute_return_distribution(df: pd.DataFrame) -> ReturnDistribution:
    """
    Histogram + tail risk (VaR 95/99, CVaR 95) + Jarque-Bera normality test.
    """
    returns = _resample_to_daily(df)
    empty = ReturnDistribution([], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    if returns.empty:
        return empty

    counts, bin_edges = np.histogram(returns, bins=40)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]

    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    tail = returns[returns <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    jb_stat, jb_p = 0.0, 1.0
    if len(returns) >= 8:
        try:
            jb_stat, jb_p = scipy_stats.jarque_bera(returns.values)
        except Exception:
            pass

    return ReturnDistribution(
        bins=[round(b, 4) for b in bin_centers],
        counts=counts.tolist(),
        mean=round(float(returns.mean()), 5),
        std=round(float(returns.std()), 5),
        skew=round(float(returns.skew()), 4),
        kurtosis=round(float(returns.kurt()), 4),
        pct_positive=round(float((returns > 0).sum() / len(returns) * 100), 2),
        var_95=round(var_95, 4),
        var_99=round(var_99, 4),
        cvar_95=round(cvar_95, 4),
        jarque_bera_stat=round(float(jb_stat), 4),
        jarque_bera_p=round(float(jb_p), 4),
    )


def compute_consecutive_stats(df: pd.DataFrame) -> ConsecutiveStats:
    """Max/avg win and loss streaks from daily returns."""
    returns = _resample_to_daily(df)

    if returns.empty:
        return ConsecutiveStats(0, 0, 0.0, 0.0, 0)

    wins = (returns > 0).astype(int).values
    win_streaks: List[int] = []
    loss_streaks: List[int] = []
    cur_win = cur_loss = 0

    for w in wins:
        if w == 1:
            cur_win += 1
            if cur_loss > 0:
                loss_streaks.append(cur_loss)
                cur_loss = 0
        else:
            cur_loss += 1
            if cur_win > 0:
                win_streaks.append(cur_win)
                cur_win = 0

    if cur_win > 0:
        win_streaks.append(cur_win)
    if cur_loss > 0:
        loss_streaks.append(cur_loss)

    max_win  = max(win_streaks,  default=0)
    max_loss = max(loss_streaks, default=0)
    avg_win  = round(float(np.mean(win_streaks)),  2) if win_streaks  else 0.0
    avg_loss = round(float(np.mean(loss_streaks)), 2) if loss_streaks else 0.0

    if len(wins) == 0:
        current = 0
    elif wins[-1] == 1:
        current = int(cur_win) if cur_win > 0 else 1
    else:
        current = -int(cur_loss) if cur_loss > 0 else -1

    return ConsecutiveStats(
        max_win_streak=max_win, max_loss_streak=max_loss,
        avg_win_streak=avg_win, avg_loss_streak=avg_loss,
        current_streak=current,
    )


def compute_autocorr(df: pd.DataFrame, max_lags: int = 10) -> AutocorrStats:
    """
    ACF for lags 1..max_lags and Ljung-Box test.
    Low p-value → returns are not IID → exploitable structure exists.
    """
    empty = AutocorrStats([], [], 0.0, 0.0, 0.0, 1.0)
    returns = _resample_to_daily(df)

    if len(returns) < max_lags + 10:
        return empty

    x = returns.values.astype(float)
    n = len(x)
    x_dm = x - x.mean()

    full = np.correlate(x_dm, x_dm, 'full')
    var = full[n - 1]
    if var < 1e-12:
        return empty

    acf_full = full[n - 1:] / var
    acf = acf_full[1: max_lags + 1]

    conf = 1.96 / np.sqrt(n)

    # Ljung-Box Q statistic
    q_stat = float(n * (n + 2) * sum(
        float(acf[k]) ** 2 / max(n - k - 1, 1) for k in range(max_lags)
    ))
    lb_p = float(1.0 - scipy_stats.chi2.cdf(q_stat, df=max_lags))

    return AutocorrStats(
        lags=list(range(1, max_lags + 1)),
        acf_values=[round(float(v), 4) for v in acf],
        conf_upper=round(conf, 4),
        conf_lower=round(-conf, 4),
        ljung_box_stat=round(q_stat, 4),
        ljung_box_p=round(lb_p, 4),
    )


def _build_summary_stats(
    dow_stats: List[DayOfWeekStats],
    monthly_stats: List[MonthStats],
    returns: pd.Series,
    tpy: int,
) -> Dict:
    summary: Dict = {}

    if dow_stats:
        best_day  = max(dow_stats, key=lambda s: s.mean_return_pct)
        worst_day = min(dow_stats, key=lambda s: s.mean_return_pct)
        summary['best_day']        = best_day.day_name
        summary['best_day_avg']    = best_day.mean_return_pct
        summary['worst_day']       = worst_day.day_name
        summary['worst_day_avg']   = worst_day.mean_return_pct

    if monthly_stats:
        best_month  = max(monthly_stats, key=lambda s: s.mean_return_pct)
        worst_month = min(monthly_stats, key=lambda s: s.mean_return_pct)
        summary['best_month']      = best_month.month_name
        summary['best_month_avg']  = best_month.mean_return_pct
        summary['worst_month']     = worst_month.month_name
        summary['worst_month_avg'] = worst_month.mean_return_pct

    if not returns.empty:
        summary['overall_win_rate']    = round(float((returns > 0).sum() / len(returns) * 100), 1)
        summary['total_observations']  = len(returns)
        summary['annualized_return']   = round(float(returns.mean() * tpy), 2)
        summary['trading_days_per_year'] = tpy

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_calendar(df: pd.DataFrame, symbol: str = '') -> CalendarAnalysis:
    """
    Full calendar analysis — statistically rigorous, frequency-aware.
    Auto-detects trading days per year (252 vs 365).
    """
    empty_dist = ReturnDistribution([], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    empty_cons = ConsecutiveStats(0, 0, 0.0, 0.0, 0)
    empty_ac   = AutocorrStats([], [], 0.0, 0.0, 0.0, 1.0)

    if df.empty:
        return CalendarAnalysis(
            day_of_week=[], day_of_week_df=pd.DataFrame(),
            monthly=[], monthly_df=pd.DataFrame(),
            monthly_heatmap=pd.DataFrame(),
            quarterly_df=pd.DataFrame(), yearly_df=pd.DataFrame(),
            rolling_dow_df=pd.DataFrame(),
            day_of_month_df=pd.DataFrame(),
            hourly_df=None, day_hour_df=None,
            distribution=empty_dist, consecutive=empty_cons, autocorr=empty_ac,
            kruskal_wallis_p=1.0, trading_days_per_year=252,
            summary_stats={}, symbol=symbol, start_date='', end_date='',
            total_bars=0, is_intraday=False,
        )

    intraday = _is_intraday(df)
    returns  = _resample_to_daily(df)
    tpy      = _detect_trading_days_per_year(returns)

    dow_stats, dow_df, kw_p = compute_day_of_week_stats(df, tpy)
    month_stats, monthly_df  = compute_monthly_stats(df, tpy)
    heatmap_df   = compute_monthly_heatmap(df)
    quarterly_df = compute_quarterly_stats(df, tpy)
    yearly_df    = compute_yearly_stats(df, tpy)
    rolling_dow  = compute_rolling_dow(df)
    dom_df       = compute_day_of_month_stats(df)
    hourly_df    = compute_hourly_stats(df) if intraday else None
    day_hour_df  = compute_day_hour_heatmap(df) if intraday else None
    distribution = compute_return_distribution(df)
    consecutive  = compute_consecutive_stats(df)
    autocorr     = compute_autocorr(df)
    summary      = _build_summary_stats(dow_stats, month_stats, returns, tpy)

    return CalendarAnalysis(
        day_of_week=dow_stats,      day_of_week_df=dow_df,
        monthly=month_stats,         monthly_df=monthly_df,
        monthly_heatmap=heatmap_df,
        quarterly_df=quarterly_df,   yearly_df=yearly_df,
        rolling_dow_df=rolling_dow,
        day_of_month_df=dom_df,
        hourly_df=hourly_df,         day_hour_df=day_hour_df,
        distribution=distribution,   consecutive=consecutive,
        autocorr=autocorr,
        kruskal_wallis_p=round(float(kw_p), 4),
        trading_days_per_year=tpy,
        summary_stats=summary,
        symbol=symbol,
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
        total_bars=len(df),
        is_intraday=intraday,
    )


def analyze_trade_calendar(trades: List) -> TradeCalendarAnalysis:
    """Analyze strategy trade performance by entry day-of-week and month."""
    if not trades:
        return TradeCalendarAnalysis(
            trades_by_day=pd.DataFrame(),
            trades_by_month=pd.DataFrame(),
        )

    rows = []
    for t in trades:
        if t.entry_date is None:
            continue
        try:
            ts = pd.Timestamp(t.entry_date)
        except Exception:
            continue
        rows.append({
            'entry_date': ts,
            'dow':        ts.dayofweek,
            'month':      ts.month,
            'pnl':        getattr(t, 'pnl', 0.0),
            'pnl_pct':    getattr(t, 'pnl_pct', 0.0),
            'direction':  getattr(t, 'direction', ''),
            'winner':     1 if getattr(t, 'pnl', 0.0) > 0 else 0,
        })

    if not rows:
        return TradeCalendarAnalysis(
            trades_by_day=pd.DataFrame(),
            trades_by_month=pd.DataFrame(),
        )

    tdf = pd.DataFrame(rows)

    day_rows = []
    for d in range(7):
        sub = tdf[tdf['dow'] == d]
        if len(sub) == 0:
            continue
        n_win = int(sub['winner'].sum())
        ci_low, ci_high = _wilson_ci(n_win, len(sub))
        day_rows.append({
            'Day':          DAY_NAMES[d],
            'Trades':       len(sub),
            'Win Rate':     f"{sub['winner'].mean() * 100:.1f}%",
            'WR 95% CI':    f"{ci_low:.0f}%–{ci_high:.0f}%",
            'Avg P&L $':    round(sub['pnl'].mean(), 2),
            'Avg P&L %':    round(sub['pnl_pct'].mean(), 3),
            'Total P&L $':  round(sub['pnl'].sum(), 2),
        })

    month_rows = []
    for m in range(1, 13):
        sub = tdf[tdf['month'] == m]
        if len(sub) == 0:
            continue
        n_win = int(sub['winner'].sum())
        ci_low, ci_high = _wilson_ci(n_win, len(sub))
        month_rows.append({
            'Month':        MONTH_NAMES[m - 1],
            'Trades':       len(sub),
            'Win Rate':     f"{sub['winner'].mean() * 100:.1f}%",
            'WR 95% CI':    f"{ci_low:.0f}%–{ci_high:.0f}%",
            'Avg P&L $':    round(sub['pnl'].mean(), 2),
            'Avg P&L %':    round(sub['pnl_pct'].mean(), 3),
            'Total P&L $':  round(sub['pnl'].sum(), 2),
        })

    return TradeCalendarAnalysis(
        trades_by_day=pd.DataFrame(day_rows) if day_rows else pd.DataFrame(),
        trades_by_month=pd.DataFrame(month_rows) if month_rows else pd.DataFrame(),
    )
