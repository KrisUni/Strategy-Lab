"""
Pure performance-metric functions (single source of truth).

All functions are side-effect free and operate on plain floats / pd.Series.
"""
import numpy as np
import pandas as pd

METRIC_CAP = 999.99   # cap for unbounded ratios (no losses / no drawdown); matches profit_factor convention


def cagr_pct(start_value: float, end_value: float, n_points: int, periods_per_year: float) -> float:
    """
    Compound annual growth rate, in percent.

        CAGR = (end / start) ** (periods_per_year / (n_points - 1)) - 1

    n_points equity observations span (n_points - 1) periods.
    Returns 0.0 if n_points < 2 or start <= 0; -100.0 if end <= 0 (ruin).

    Parameters:
        start_value: Equity at the first observation.
        end_value: Equity at the last observation.
        n_points: Number of equity observations.
        periods_per_year: Observations per calendar year.

    Returns:
        CAGR in percent.
    """
    if n_points < 2 or start_value <= 0:
        return 0.0
    if end_value <= 0:
        return -100.0
    growth = (end_value / start_value) ** (periods_per_year / (n_points - 1)) - 1
    return float(growth * 100)


def annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    """
    Annualized Sharpe ratio (risk-free = 0).

        Sharpe = mean(r) / std(r, ddof=1) * sqrt(periods_per_year)

    Returns 0.0 if fewer than 2 observations or std == 0.

    Parameters:
        returns: Per-period simple returns.
        periods_per_year: Observations per calendar year.

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if not std > 0:
        return 0.0
    return float((returns.mean() / std) * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """
    Downside deviation over ALL observations (Sortino & Price definition).

        DD = sqrt( mean( min(r - target, 0)^2 ) )

    Returns 0.0 for an empty series.

    Parameters:
        returns: Per-period simple returns.
        target: Minimum acceptable return per period.

    Returns:
        Per-period downside deviation.
    """
    if len(returns) == 0:
        return 0.0
    shortfall = np.minimum(returns.to_numpy(dtype=float) - target, 0.0)
    return float(np.sqrt(np.mean(shortfall ** 2)))


def annualized_sortino(returns: pd.Series, periods_per_year: float) -> float:
    """
    Annualized Sortino ratio (target = 0).

        Sortino = mean(r) / DD * sqrt(periods_per_year)

    Returns 0.0 if fewer than 2 observations.
    If DD == 0: METRIC_CAP when mean(r) > 0, else 0.0.

    Parameters:
        returns: Per-period simple returns.
        periods_per_year: Observations per calendar year.

    Returns:
        Annualized Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0
    dd = downside_deviation(returns)
    mean = returns.mean()
    if dd == 0:
        return METRIC_CAP if mean > 0 else 0.0
    return float((mean / dd) * np.sqrt(periods_per_year))


def calmar_ratio(cagr_percent: float, max_drawdown_percent: float) -> float:
    """
    Calmar ratio. Sign follows CAGR.

        Calmar = CAGR% / |MaxDD%|

    If MaxDD == 0: METRIC_CAP when CAGR > 0, else 0.0.

    Parameters:
        cagr_percent: CAGR in percent.
        max_drawdown_percent: Max drawdown in percent (sign ignored).

    Returns:
        Calmar ratio.
    """
    if max_drawdown_percent == 0:
        return METRIC_CAP if cagr_percent > 0 else 0.0
    return float(cagr_percent / abs(max_drawdown_percent))
