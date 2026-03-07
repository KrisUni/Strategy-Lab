"""
Monte Carlo Module
==================
Three simulation methods:
1. Trade Shuffle    – permute trade execution order
2. Return Bootstrap – resample daily returns with replacement
3. Noise Injection  – add Gaussian noise to trade P&L (stress test)

Outputs:
- Equity path distribution (for confidence band charts)
- Final equity distribution
- Risk of ruin probability
- Maximum drawdown distribution
- Confidence intervals for key metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo analysis result."""
    # Method used
    method: str

    # Equity paths: shape (n_simulations, n_steps)
    equity_paths: np.ndarray

    # Final equity per simulation
    final_equities: np.ndarray

    # Max drawdown per simulation (as negative %)
    max_drawdowns: np.ndarray

    # Risk of ruin: probability of equity falling below threshold
    risk_of_ruin: float          # P(equity < ruin_level)
    ruin_level: float            # Dollar level used

    # Percentile summaries
    equity_percentiles: Dict[str, float]   # 5th, 25th, 50th, 75th, 95th
    dd_percentiles: Dict[str, float]

    # Confidence bands for equity paths at each time step
    # Each key is a percentile string, value is array of length n_steps
    equity_bands: Dict[str, np.ndarray]

    # Sharpe distribution (if method is return_bootstrap)
    sharpe_distribution: Optional[np.ndarray] = None

    n_simulations: int = 0
    initial_capital: float = 10000.0


def _equity_from_pnls(pnls: np.ndarray, initial_capital: float) -> np.ndarray:
    """Build equity curve from P&L sequence. Returns array of length len(pnls)+1."""
    equity = np.empty(len(pnls) + 1)
    equity[0] = initial_capital
    np.cumsum(pnls, out=equity[1:])
    equity[1:] += initial_capital
    return equity


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Compute maximum drawdown as a negative percentage."""
    peak = np.maximum.accumulate(equity)
    # Avoid division by zero
    safe_peak = np.where(peak > 0, peak, 1.0)
    dd_pct = (equity - peak) / safe_peak * 100
    return dd_pct.min()


def trade_shuffle(
    trades: List,
    n_simulations: int = 1000,
    initial_capital: float = 10000,
    ruin_pct: float = 50.0,
    seed: int = 42
) -> MonteCarloResult:
    """
    Monte Carlo via trade order permutation.
    
    Shuffles the sequence of trade P&L values to explore how
    sensitive the equity path is to trade ordering.
    
    This tests: "Would a different sequence of the same trades
    have caused ruin or a much worse drawdown?"
    
    Args:
        trades: List of Trade objects (must have .pnl attribute)
        n_simulations: Number of permutations to run
        initial_capital: Starting equity
        ruin_pct: Ruin defined as losing this % of capital
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)

    if n_trades < 2:
        empty = np.array([[initial_capital]])
        return MonteCarloResult(
            method='trade_shuffle', equity_paths=empty,
            final_equities=np.array([initial_capital]),
            max_drawdowns=np.array([0.0]),
            risk_of_ruin=0.0, ruin_level=initial_capital * (1 - ruin_pct / 100),
            equity_percentiles={'5%': initial_capital, '25%': initial_capital,
                                '50%': initial_capital, '75%': initial_capital,
                                '95%': initial_capital},
            dd_percentiles={'5%': 0, '50%': 0, '95%': 0},
            equity_bands={}, n_simulations=0,
            initial_capital=initial_capital,
        )

    ruin_level = initial_capital * (1 - ruin_pct / 100)

    # Pre-allocate: each path has n_trades+1 points (including start)
    paths = np.empty((n_simulations, n_trades + 1))
    finals = np.empty(n_simulations)
    dds = np.empty(n_simulations)

    for sim in range(n_simulations):
        shuffled = pnls.copy()
        rng.shuffle(shuffled)
        eq = _equity_from_pnls(shuffled, initial_capital)
        paths[sim] = eq
        finals[sim] = eq[-1]
        dds[sim] = _max_drawdown_pct(eq)

    # Risk of ruin: fraction of sims where equity ever dipped below ruin level
    ruin_count = 0
    for sim in range(n_simulations):
        if paths[sim].min() <= ruin_level:
            ruin_count += 1
    risk_of_ruin = ruin_count / n_simulations

    # Percentile bands at each time step
    bands = {}
    for pct_label, pct_val in [('5%', 5), ('25%', 25), ('50%', 50),
                                ('75%', 75), ('95%', 95)]:
        bands[pct_label] = np.percentile(paths, pct_val, axis=0)

    return MonteCarloResult(
        method='trade_shuffle',
        equity_paths=paths,
        final_equities=finals,
        max_drawdowns=dds,
        risk_of_ruin=risk_of_ruin,
        ruin_level=ruin_level,
        equity_percentiles={
            '5%': np.percentile(finals, 5),
            '25%': np.percentile(finals, 25),
            '50%': np.percentile(finals, 50),
            '75%': np.percentile(finals, 75),
            '95%': np.percentile(finals, 95),
        },
        dd_percentiles={
            '5%': np.percentile(dds, 5),
            '50%': np.percentile(dds, 50),
            '95%': np.percentile(dds, 95),
        },
        equity_bands=bands,
        n_simulations=n_simulations,
        initial_capital=initial_capital,
    )


def return_bootstrap(
    equity_curve: pd.Series,
    n_simulations: int = 1000,
    initial_capital: float = 10000,
    ruin_pct: float = 50.0,
    block_size: int = 5,
    seed: int = 42,
    bars_per_year: int = 252
) -> MonteCarloResult:
    """
    Monte Carlo via block bootstrap of returns.
    
    Resamples blocks of returns (preserving short-term autocorrelation)
    to generate synthetic equity paths of the same length.
    
    This tests: "Given the observed return distribution, what range
    of outcomes could we expect?"
    
    Args:
        equity_curve: Mark-to-market equity curve
        n_simulations: Number of bootstrap samples
        initial_capital: Starting equity
        ruin_pct: Ruin threshold as % of capital
        block_size: Block length for block bootstrap (preserves autocorrelation)
        seed: Random seed
        bars_per_year: For Sharpe calculation
    """
    rng = np.random.default_rng(seed)
    returns = equity_curve.pct_change().dropna().values
    n_returns = len(returns)

    if n_returns < block_size * 2:
        # Fall back to simple bootstrap if not enough data for blocks
        block_size = 1

    ruin_level = initial_capital * (1 - ruin_pct / 100)

    paths = np.empty((n_simulations, n_returns + 1))
    finals = np.empty(n_simulations)
    dds = np.empty(n_simulations)
    sharpes = np.empty(n_simulations)

    for sim in range(n_simulations):
        # Block bootstrap: sample blocks of consecutive returns
        sampled = np.empty(n_returns)
        idx = 0
        while idx < n_returns:
            start = rng.integers(0, max(1, n_returns - block_size + 1))
            end = min(start + block_size, n_returns)
            chunk = returns[start:end]
            take = min(len(chunk), n_returns - idx)
            sampled[idx:idx + take] = chunk[:take]
            idx += take

        # Build equity from bootstrapped returns
        eq = np.empty(n_returns + 1)
        eq[0] = initial_capital
        for j in range(n_returns):
            eq[j + 1] = eq[j] * (1 + sampled[j])

        paths[sim] = eq
        finals[sim] = eq[-1]
        dds[sim] = _max_drawdown_pct(eq)

        # Sharpe for this sim
        std = sampled.std()
        sharpes[sim] = (sampled.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0

    # Risk of ruin
    ruin_count = sum(1 for sim in range(n_simulations) if paths[sim].min() <= ruin_level)
    risk_of_ruin = ruin_count / n_simulations

    bands = {}
    for pct_label, pct_val in [('5%', 5), ('25%', 25), ('50%', 50),
                                ('75%', 75), ('95%', 95)]:
        bands[pct_label] = np.percentile(paths, pct_val, axis=0)

    return MonteCarloResult(
        method='return_bootstrap',
        equity_paths=paths,
        final_equities=finals,
        max_drawdowns=dds,
        risk_of_ruin=risk_of_ruin,
        ruin_level=ruin_level,
        equity_percentiles={
            '5%': np.percentile(finals, 5),
            '25%': np.percentile(finals, 25),
            '50%': np.percentile(finals, 50),
            '75%': np.percentile(finals, 75),
            '95%': np.percentile(finals, 95),
        },
        dd_percentiles={
            '5%': np.percentile(dds, 5),
            '50%': np.percentile(dds, 50),
            '95%': np.percentile(dds, 95),
        },
        equity_bands=bands,
        sharpe_distribution=sharpes,
        n_simulations=n_simulations,
        initial_capital=initial_capital,
    )


def noise_injection(
    trades: List,
    n_simulations: int = 1000,
    initial_capital: float = 10000,
    noise_pct: float = 20.0,
    ruin_pct: float = 50.0,
    seed: int = 42
) -> MonteCarloResult:
    """
    Monte Carlo via noise injection on trade P&L.
    
    Adds Gaussian noise proportional to each trade's P&L magnitude.
    This stress-tests: "What if our fills were slightly different,
    or market conditions caused small deviations in each trade?"
    
    Args:
        trades: List of Trade objects
        n_simulations: Number of simulations
        initial_capital: Starting equity
        noise_pct: Noise magnitude as % of each trade's |P&L|
        ruin_pct: Ruin threshold
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)

    if n_trades < 2:
        empty = np.array([[initial_capital]])
        return MonteCarloResult(
            method='noise_injection', equity_paths=empty,
            final_equities=np.array([initial_capital]),
            max_drawdowns=np.array([0.0]),
            risk_of_ruin=0.0, ruin_level=initial_capital * (1 - ruin_pct / 100),
            equity_percentiles={'5%': initial_capital, '25%': initial_capital,
                                '50%': initial_capital, '75%': initial_capital,
                                '95%': initial_capital},
            dd_percentiles={'5%': 0, '50%': 0, '95%': 0},
            equity_bands={}, n_simulations=0,
            initial_capital=initial_capital,
        )

    ruin_level = initial_capital * (1 - ruin_pct / 100)
    noise_scale = noise_pct / 100.0

    paths = np.empty((n_simulations, n_trades + 1))
    finals = np.empty(n_simulations)
    dds = np.empty(n_simulations)

    abs_pnls = np.abs(pnls)
    # Use median |pnl| as noise base for zero-pnl trades
    median_abs = np.median(abs_pnls[abs_pnls > 0]) if (abs_pnls > 0).any() else 1.0

    for sim in range(n_simulations):
        noise_base = np.where(abs_pnls > 0, abs_pnls, median_abs)
        noise = rng.normal(0, noise_base * noise_scale)
        noisy_pnls = pnls + noise
        eq = _equity_from_pnls(noisy_pnls, initial_capital)
        paths[sim] = eq
        finals[sim] = eq[-1]
        dds[sim] = _max_drawdown_pct(eq)

    ruin_count = sum(1 for sim in range(n_simulations) if paths[sim].min() <= ruin_level)
    risk_of_ruin = ruin_count / n_simulations

    bands = {}
    for pct_label, pct_val in [('5%', 5), ('25%', 25), ('50%', 50),
                                ('75%', 75), ('95%', 95)]:
        bands[pct_label] = np.percentile(paths, pct_val, axis=0)

    return MonteCarloResult(
        method='noise_injection',
        equity_paths=paths,
        final_equities=finals,
        max_drawdowns=dds,
        risk_of_ruin=risk_of_ruin,
        ruin_level=ruin_level,
        equity_percentiles={
            '5%': np.percentile(finals, 5),
            '25%': np.percentile(finals, 25),
            '50%': np.percentile(finals, 50),
            '75%': np.percentile(finals, 75),
            '95%': np.percentile(finals, 95),
        },
        dd_percentiles={
            '5%': np.percentile(dds, 5),
            '50%': np.percentile(dds, 50),
            '95%': np.percentile(dds, 95),
        },
        equity_bands=bands,
        n_simulations=n_simulations,
        initial_capital=initial_capital,
    )


def run_monte_carlo(
    trades: List,
    equity_curve: Optional[pd.Series] = None,
    method: str = 'trade_shuffle',
    n_simulations: int = 1000,
    initial_capital: float = 10000,
    ruin_pct: float = 50.0,
    noise_pct: float = 20.0,
    block_size: int = 5,
    bars_per_year: int = 252,
    seed: int = 42
) -> Optional[MonteCarloResult]:
    """
    Unified entry point for Monte Carlo simulation.
    
    Args:
        trades: List of Trade objects
        equity_curve: Mark-to-market equity (required for return_bootstrap)
        method: 'trade_shuffle', 'return_bootstrap', or 'noise_injection'
        n_simulations: Number of simulations
        initial_capital: Starting capital
        ruin_pct: % loss threshold for ruin calculation
        noise_pct: Noise level for noise_injection method
        block_size: Block size for return_bootstrap
        bars_per_year: Annualization factor
        seed: Random seed
    """
    if not trades or len(trades) < 3:
        return None

    if method == 'trade_shuffle':
        return trade_shuffle(trades, n_simulations, initial_capital, ruin_pct, seed)
    elif method == 'return_bootstrap':
        if equity_curve is None or len(equity_curve) < 20:
            return None
        return return_bootstrap(
            equity_curve, n_simulations, initial_capital,
            ruin_pct, block_size, seed, bars_per_year
        )
    elif method == 'noise_injection':
        return noise_injection(
            trades, n_simulations, initial_capital,
            noise_pct, ruin_pct, seed
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trade_shuffle', 'return_bootstrap', or 'noise_injection'.")

# ═══════════════════════════════════════════════════════════════════════════════
# IN-SAMPLE PERMUTATION TEST
# Add to: src/montecarlo/__init__.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# Design:
#   - Shuffle interior bars of the price series (rows 1 .. N-2)
#   - Keep row[0] and row[-1] fixed → preserves overall trend/drift
#   - Re-run the full strategy + backtest on each permuted series
#   - Build null distribution of a chosen metric
#   - p-value = fraction of permuted runs >= real strategy metric
#
# Why fix endpoints?
#   The raw permutation destroys trend. A bearish market strategy would look
#   great on a shuffled bull market just by luck. Anchoring first/last close
#   keeps the same net price move (same "trend") while destroying all
#   exploitable temporal structure in between.
#
# What this tests:
#   H0: The strategy's edge is indistinguishable from random bar ordering.
#   A low p-value (e.g. < 0.05) means the strategy is exploiting real
#   temporal structure, not just riding the trend.
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np
import pandas as pd


@dataclass
class PermutationTestResult:
    """
    Result of an in-sample permutation test.

    Attributes
    ----------
    real_metric : float
        Metric value on the actual (unshuffled) backtest.
    null_distribution : np.ndarray
        Metric values across all permuted runs.
    p_value : float
        Fraction of permuted runs that matched or exceeded real_metric.
        Interpretation: p < 0.05 → strategy edge is statistically significant.
    metric_name : str
        Name of the metric used (e.g. 'profit_factor', 'sharpe', 'net_profit').
    n_permutations : int
        Number of permuted series tested.
    pct_5 : float
        5th percentile of null distribution.
    pct_50 : float
        Median of null distribution.
    pct_95 : float
        95th percentile of null distribution.
    beat_pct : float
        Percentage of null runs the real strategy outperformed (= 1 - p_value).
    """
    real_metric: float
    null_distribution: np.ndarray
    p_value: float
    metric_name: str
    n_permutations: int
    pct_5: float
    pct_50: float
    pct_95: float
    beat_pct: float


def _permute_bars_fixed_endpoints(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Shuffle interior rows of an OHLCV dataframe while keeping the first
    and last rows in place.

    This preserves:
      - The starting price (open of bar 0)
      - The ending price (close of last bar)
      - The overall net price change (trend / drift)

    All OHLC relationships within each individual bar are intact because
    whole rows are shuffled, not individual columns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe indexed by datetime. Must have at least 3 rows.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        Permuted dataframe with same index as original.
    """
    n = len(df)
    if n < 3:
        return df.copy()

    # Indices of interior bars (everything except first and last)
    interior_idx = np.arange(1, n - 1)
    shuffled_interior = rng.permutation(interior_idx)

    # Build new row order: first | shuffled interior | last
    new_order = np.concatenate([[0], shuffled_interior, [n - 1]])

    permuted = df.iloc[new_order].copy()
    # Restore the original datetime index so indicators compute correctly
    permuted.index = df.index
    return permuted


def _extract_metric(results, metric_name: str) -> float:
    """
    Pull a scalar metric from a BacktestResults object.

    Supported metrics
    -----------------
    'profit_factor'  : gross_profit / gross_loss  (0 if no trades)
    'sharpe'         : annualized Sharpe ratio
    'net_profit'     : final equity - initial capital
    'win_rate'       : fraction of winning trades
    'calmar'         : CAGR / max_drawdown
    'expectancy'     : mean P&L per trade
    'total_return'   : percentage return on capital
    """
    if results is None or not results.trades:
        return 0.0

    m = metric_name.lower()

    if m == 'profit_factor':
        return getattr(results, 'profit_factor', 0.0) or 0.0

    elif m == 'sharpe':
        return getattr(results, 'sharpe_ratio', 0.0) or 0.0

    elif m == 'net_profit':
        ec = results.equity_curve
        if ec is not None and len(ec) > 1:
            return float(ec.iloc[-1] - ec.iloc[0])
        return 0.0

    elif m == 'win_rate':
        return getattr(results, 'win_rate', 0.0) or 0.0

    elif m == 'calmar':
        cagr = getattr(results, 'cagr', 0.0) or 0.0
        mdd  = abs(getattr(results, 'max_drawdown', 1.0) or 1.0)
        return cagr / mdd if mdd > 0 else 0.0

    elif m == 'expectancy':
        trades = results.trades
        if not trades:
            return 0.0
        return float(np.mean([t.pnl for t in trades]))

    elif m == 'total_return':
        ec = results.equity_curve
        if ec is not None and len(ec) > 1 and ec.iloc[0] != 0:
            return float((ec.iloc[-1] - ec.iloc[0]) / ec.iloc[0] * 100)
        return 0.0

    else:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Choose from: "
            "profit_factor, sharpe, net_profit, win_rate, calmar, expectancy, total_return"
        )


def in_sample_permutation_test(
    df: pd.DataFrame,
    run_backtest_fn: Callable,          # fn(df, params, **kwargs) -> BacktestResults
    params,                              # StrategyParams or dict passed to run_backtest_fn
    metric: str = 'profit_factor',
    n_permutations: int = 500,
    seed: int = 42,
    real_results=None,                   # Pass pre-computed results to avoid re-running
    **backtest_kwargs
) -> PermutationTestResult:
    """
    In-sample permutation test with fixed endpoints.

    Runs the strategy on N randomly permuted versions of the price series
    (first and last bars anchored) to build a null distribution, then
    computes a p-value for the real strategy's metric.

    Mathematical definition
    -----------------------
        p = #{permuted_metric >= real_metric} / n_permutations

        A p-value < 0.05 indicates the real strategy's performance is
        unlikely to arise from random bar ordering → edge is real.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV dataframe used in the original backtest.
    run_backtest_fn : Callable
        Your existing backtest runner. Signature: fn(df, params, **kwargs)
        Must return a BacktestResults object.
    params : StrategyParams
        Strategy parameters (same as used in the real backtest).
    metric : str
        Performance metric to test. Options:
        'profit_factor' | 'sharpe' | 'net_profit' |
        'win_rate' | 'calmar' | 'expectancy' | 'total_return'
    n_permutations : int
        Number of shuffled series to generate and test.
        Recommended: 500 (fast) to 2000 (high precision p-value).
    seed : int
        Random seed for reproducibility.
    real_results : BacktestResults, optional
        Pre-computed backtest on the real data. If None, it will be run.
    **backtest_kwargs
        Any additional kwargs forwarded to run_backtest_fn.

    Returns
    -------
    PermutationTestResult

    Warnings
    --------
    - n_permutations * backtest_runtime = total wall time. For complex
      strategies this can be slow. Start with n=200 to gauge speed.
    - The test is in-sample: it does NOT validate out-of-sample performance.
      Use walk-forward for that. This tests whether temporal structure matters
      at all — a necessary but not sufficient condition for a real edge.
    """
    rng = np.random.default_rng(seed)

    if len(df) < 3:
        raise ValueError("Dataframe must have at least 3 rows for permutation test.")

    # ── Real metric ──────────────────────────────────────────────────────────
    if real_results is None:
        real_results = run_backtest_fn(df, params, **backtest_kwargs)

    real_metric_val = _extract_metric(real_results, metric)

    # ── Null distribution ─────────────────────────────────────────────────────
    null_metrics = np.empty(n_permutations)

    for i in range(n_permutations):
        permuted_df = _permute_bars_fixed_endpoints(df, rng)
        try:
            perm_results = run_backtest_fn(permuted_df, params, **backtest_kwargs)
            null_metrics[i] = _extract_metric(perm_results, metric)
        except Exception:
            # If backtest fails on a permuted series (e.g. no trades), record 0
            null_metrics[i] = 0.0

    # ── p-value ───────────────────────────────────────────────────────────────
    # One-tailed: how often does a random permutation match or beat the real run?
    p_value = float(np.mean(null_metrics >= real_metric_val))

    return PermutationTestResult(
        real_metric=real_metric_val,
        null_distribution=null_metrics,
        p_value=p_value,
        metric_name=metric,
        n_permutations=n_permutations,
        pct_5=float(np.percentile(null_metrics, 5)),
        pct_50=float(np.percentile(null_metrics, 50)),
        pct_95=float(np.percentile(null_metrics, 95)),
        beat_pct=float((1 - p_value) * 100),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — add to your chart utilities
# ═══════════════════════════════════════════════════════════════════════════════

def create_permutation_chart(result: PermutationTestResult):
    """
    Histogram of the null distribution with the real metric overlaid.

    Visual interpretation:
      - The histogram = what random bar ordering produces
      - The red vertical line = your real strategy
      - If the red line is far right → strong edge
      - p-value shown in annotation
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Null distribution histogram
    fig.add_trace(go.Histogram(
        x=result.null_distribution,
        nbinsx=50,
        name='Null Distribution (Permuted)',
        marker_color='rgba(100, 149, 237, 0.6)',
        marker_line=dict(color='rgba(100, 149, 237, 1.0)', width=0.5),
    ))

    # Real metric line
    fig.add_vline(
        x=result.real_metric,
        line_color='#ef4444',
        line_width=2.5,
        annotation_text=f"Real: {result.real_metric:.3f}",
        annotation_position="top right",
        annotation_font_color='#ef4444',
    )

    # Median of null
    fig.add_vline(
        x=result.pct_50,
        line_color='#94a3b8',
        line_width=1.5,
        line_dash='dash',
        annotation_text=f"Null median: {result.pct_50:.3f}",
        annotation_position="top left",
        annotation_font_color='#94a3b8',
    )

    # Significance label
    sig_label = "✅ Significant (p < 0.05)" if result.p_value < 0.05 else "⚠️ Not significant (p ≥ 0.05)"
    sig_color = '#10b981' if result.p_value < 0.05 else '#f59e0b'

    fig.update_layout(
        title=dict(
            text=f"Permutation Test — {result.metric_name.replace('_', ' ').title()}<br>"
                 f"<span style='font-size:13px; color:{sig_color}'>"
                 f"{sig_label} &nbsp;|&nbsp; p = {result.p_value:.4f} &nbsp;|&nbsp; "
                 f"Beat {result.beat_pct:.1f}% of permutations</span>",
            font_size=16,
        ),
        xaxis_title=result.metric_name.replace('_', ' ').title(),
        yaxis_title='Count',
        paper_bgcolor='#0a0e14',
        plot_bgcolor='#11151c',
        font_color='#e2e8f0',
        showlegend=True,
        bargap=0.05,
        height=420,
    )

    return fig

