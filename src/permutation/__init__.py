"""
Permutation Test Module
=======================
In-sample permutation test for strategy validation.

Null hypothesis: the strategy has no real edge — any observed performance
is indistinguishable from what you'd get by optimizing on randomised data.

Procedure:
1. Optimize on real data → record metric (e.g. profit factor).
2. For each of N permutations:
   a. Shuffle daily returns (destroying any real signal).
   b. Reconstruct synthetic prices from shuffled returns.
   c. Run the SAME optimizer on synthetic data.
   d. Record the best metric found.
3. p-value = (# permuted metrics >= real metric) / N.

If p < 0.05, we reject the null — the strategy's edge is unlikely due to chance.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from ..backtest import DEFAULT_COMMISSION_PCT, DEFAULT_SLIPPAGE_PCT


@dataclass
class PermutationResult:
    """Result of an in-sample permutation test."""
    # Real data results
    real_metric: float
    real_equity: np.ndarray          # equity curve from real optimization

    # Permutation distribution
    permuted_metrics: np.ndarray     # shape (n_permutations,)
    permuted_equities: List[np.ndarray]  # list of equity curves

    # Statistical inference
    p_value: float                   # (count >= real) / n_permutations
    n_permutations: int
    metric_name: str

    # For display
    real_num_trades: int = 0
    avg_permuted_trades: float = 0.0


def _permute_prices(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Shuffle daily returns and reconstruct synthetic OHLCV prices.

    Preserves: return distribution, volatility, volume distribution.
    Destroys: temporal structure, autocorrelation, any real signal.
    """
    close = df['close'].values
    returns = np.diff(close) / close[:-1]

    # Shuffle returns
    rng.shuffle(returns)

    # Reconstruct close prices
    new_close = np.empty(len(close))
    new_close[0] = close[0]
    for i in range(len(returns)):
        new_close[i + 1] = new_close[i] * (1 + returns[i])

    # Scale OHLV relative to close
    ratio = new_close / close

    new_df = df.copy()
    new_df['close'] = new_close
    new_df['open'] = df['open'].values * ratio
    new_df['high'] = df['high'].values * ratio
    new_df['low'] = df['low'].values * ratio
    # Volume stays the same (shuffled returns, not volume)

    # Enforce OHLC consistency
    new_df['high'] = new_df[['open', 'high', 'close']].max(axis=1)
    new_df['low'] = new_df[['open', 'low', 'close']].min(axis=1)

    return new_df


def run_permutation_test(
    df: pd.DataFrame,
    enabled_filters: Dict[str, bool],
    metric: str = 'profit_factor',
    n_permutations: int = 100,
    n_trials: int = 50,
    min_trades: int = 5,
    initial_capital: float = 10000,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    trade_direction: str = 'long_only',
    train_pct: float = 0.7,
    seed: int = 42,
    progress_callback=None,
    pinned_params: Optional[Dict[str, Any]] = None,
) -> Optional[PermutationResult]:
    """
    Run in-sample permutation test.

    Args:
        df: OHLCV DataFrame
        enabled_filters: which indicators are active
        metric: BacktestResults attribute to test
        n_permutations: number of shuffled datasets to test
        n_trials: Optuna trials per optimization (keep low for speed)
        min_trades: minimum trades for valid optimization
        initial_capital: starting capital
        commission_pct: commission percentage
        slippage_pct: slippage percentage applied per fill
        trade_direction: 'long_only', 'short_only', 'both'
        train_pct: train/test split fraction
        seed: random seed
        progress_callback: callable(current, total) for progress updates
        pinned_params: parameters to hold fixed during optimization

    Returns:
        PermutationResult or None if insufficient data/trades.
    """
    # Import here to avoid circular dependency
    from src.optimize import optimize_strategy

    rng = np.random.default_rng(seed)

    if len(df) < 50:
        return None

    # ── Step 1: Optimize on real data ─────────────────────────────────────
    real_result = optimize_strategy(
        df=df,
        enabled_filters=enabled_filters,
        metric=metric,
        n_trials=n_trials,
        min_trades=min_trades,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        trade_direction=trade_direction,
        train_pct=train_pct,
        use_walkforward=False,
        show_progress=False,
        pinned_params=pinned_params,
    )

    real_metric_val = getattr(real_result.full_data_results, metric, 0.0)
    if real_metric_val is None or np.isnan(real_metric_val):
        real_metric_val = 0.0

    real_equity = real_result.full_data_results.equity_curve.values \
        if real_result.full_data_results.equity_curve is not None \
        else np.array([initial_capital])
    real_num_trades = real_result.full_data_results.num_trades

    # ── Step 2: Permutation loop ──────────────────────────────────────────
    permuted_metrics = np.zeros(n_permutations)
    permuted_equities = []
    permuted_trade_counts = []

    for i in range(n_permutations):
        if progress_callback:
            progress_callback(i + 1, n_permutations)

        # Generate permuted dataset
        perm_df = _permute_prices(df, rng)

        try:
            perm_result = optimize_strategy(
                df=perm_df,
                enabled_filters=enabled_filters,
                metric=metric,
                n_trials=n_trials,
                min_trades=min_trades,
                initial_capital=initial_capital,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
                trade_direction=trade_direction,
                train_pct=train_pct,
                use_walkforward=False,
                show_progress=False,
                pinned_params=pinned_params,
            )

            perm_metric = getattr(perm_result.full_data_results, metric, 0.0)
            if perm_metric is None or np.isnan(perm_metric):
                perm_metric = 0.0

            permuted_metrics[i] = perm_metric

            perm_eq = perm_result.full_data_results.equity_curve.values \
                if perm_result.full_data_results.equity_curve is not None \
                else np.array([initial_capital])
            permuted_equities.append(perm_eq)
            permuted_trade_counts.append(perm_result.full_data_results.num_trades)

        except Exception:
            permuted_metrics[i] = 0.0
            permuted_equities.append(np.array([initial_capital, initial_capital]))
            permuted_trade_counts.append(0)

    # ── Step 3: Compute p-value ───────────────────────────────────────────
    count_ge = np.sum(permuted_metrics >= real_metric_val)
    p_value = count_ge / n_permutations

    avg_perm_trades = float(np.mean(permuted_trade_counts)) if permuted_trade_counts else 0.0

    return PermutationResult(
        real_metric=real_metric_val,
        real_equity=real_equity,
        permuted_metrics=permuted_metrics,
        permuted_equities=permuted_equities,
        p_value=p_value,
        n_permutations=n_permutations,
        metric_name=metric,
        real_num_trades=real_num_trades,
        avg_permuted_trades=avg_perm_trades,
    )
