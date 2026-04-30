"""
src/mcp_server.py
=================
MCP server that exposes Strategy Lab's core functions as tools.
Claude (via Claude Desktop) connects via stdio and drives the app externally.

Transport: stdio  ← Claude Desktop reads this process's stdin/stdout
Usage:  python -m src.mcp_server
        (or: fastmcp run src/mcp_server.py)

Claude Desktop config  (~/.config/claude/claude_desktop_config.json  or
                        ~/Library/Application Support/Claude/claude_desktop_config.json):

    {
      "mcpServers": {
        "strategy-lab": {
          "command": "/absolute/path/to/venv/bin/python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/absolute/path/to/strategy-lab"
        }
      }
    }

Typical session flow:
    1. load_data("SPY", "1d", "2020-01-01")
    2. get_market_characterization()
    3. set_params({"pamrp_enabled": True, "bbwp_enabled": True, ...})
    4. run_backtest()
    5. run_optimize(metric="sharpe_ratio", n_trials=200, use_walkforward=True)
    6. run_permutation_test(n_permutations=100)
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import numpy as np
from fastmcp import FastMCP

from src.backtest import BacktestEngine, DEFAULT_COMMISSION_PCT, DEFAULT_SLIPPAGE_PCT
from src.data import fetch_yfinance
from src.indicators import adx as _adx, atr as _atr, sma as _sma
from src.optimize import optimize_strategy
from src.permutation import run_permutation_test as _run_permutation_test
from ui.helpers import params_to_strategy

# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("strategy-lab", version="1.0.0")

# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# Single dict; tools read/write across calls within the same session.
# ─────────────────────────────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "df": None,
    "symbol": None,
    "interval": None,
    "params": {},                          # flat StrategyParams-compatible dict
    "capital": 10_000.0,
    "commission": DEFAULT_COMMISSION_PCT,
    "slippage": DEFAULT_SLIPPAGE_PCT,
    "last_metrics": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sf(v: Any) -> Any:
    """JSON-safe float: round to 4dp, map nan/inf → None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _results_to_dict(r) -> Dict[str, Any]:
    """Serialize a BacktestResults dataclass to a plain dict."""
    return {
        "num_trades":         r.num_trades,
        "total_return_pct":   _sf(r.total_return_pct),
        "cagr":               _sf(r.cagr),
        "sharpe_ratio":       _sf(r.sharpe_ratio),
        "sortino_ratio":      _sf(r.sortino_ratio),
        "calmar_ratio":       _sf(r.calmar_ratio),
        "max_drawdown_pct":   _sf(r.max_drawdown_pct),
        "win_rate":           _sf(r.win_rate),
        "profit_factor":      _sf(r.profit_factor),
        "expectancy":         _sf(r.expectancy),
        "payoff_ratio":       _sf(r.payoff_ratio),
        "pct_time_in_market": _sf(r.pct_time_in_market),
    }


def _require_data() -> Optional[Dict]:
    """Return error dict if no data is loaded, else None."""
    if _state["df"] is None:
        return {"error": "No data loaded — call load_data() first."}
    return None


def _enabled_filters() -> Dict[str, bool]:
    """Extract *_enabled flags from current params."""
    return {
        k: v
        for k, v in _state["params"].items()
        if k.endswith("_enabled") and isinstance(v, bool)
    }

# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def load_data(
    symbol: str,
    interval: str = "1d",
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch OHLCV data for a symbol via yfinance and store it in server state.

    Parameters
    ----------
    symbol   : Ticker, e.g. "SPY", "BTC-USD", "TSLA".
    interval : Bar size — 1m/5m/15m/30m/1h/1d/1wk (yfinance limits apply).
    start    : ISO date string, e.g. "2020-01-01".
    end      : ISO date string. Defaults to today.

    Returns market summary: bar count, date range, annualised vol, total return.
    """
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    df = fetch_yfinance(symbol, start, end, interval)
    _state.update({"df": df, "symbol": symbol, "interval": interval})

    close = df["close"]
    returns = close.pct_change().dropna()

    return {
        "status": "ok",
        "symbol": symbol,
        "interval": interval,
        "bars": len(df),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "close_latest": _sf(close.iloc[-1]),
        "close_range": [_sf(close.min()), _sf(close.max())],
        "ann_volatility_pct": _sf(returns.std() * np.sqrt(252) * 100),
        "total_return_pct": _sf((close.iloc[-1] / close.iloc[0] - 1) * 100),
    }


@mcp.tool()
def load_data_from_file(
    file_path: str,
    symbol: str = "CUSTOM",
    interval: str = "1h",
) -> Dict[str, Any]:
    """
    Load OHLCV data from a local CSV file into server state.

    The CSV must have columns: datetime/date/time (index), open, high, low, close.
    Volume column is optional. Column names are case-insensitive.

    Parameters
    ----------
    file_path : Absolute or relative path to the CSV file.
    symbol    : Label to tag this dataset (e.g. "JOE-USD").
    interval  : Bar size label, e.g. "1h".
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / file_path

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Find the datetime column
    time_col = next(
        (c for c in df.columns if c in ("datetime", "date", "time", "timestamp", "index")),
        df.columns[0],
    )
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    # Normalise required columns
    rename = {}
    for col in df.columns:
        for std in ("open", "high", "low", "close", "volume"):
            if col == std or col.endswith(f"_{std}") or col.startswith(f"{std}_"):
                rename[col] = std
    df = df.rename(columns=rename)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        return {"status": "error", "message": f"Missing columns: {missing}"}

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    _state.update({"df": df, "symbol": symbol, "interval": interval})

    close = df["close"]
    returns = close.pct_change().dropna()

    return {
        "status": "ok",
        "symbol": symbol,
        "interval": interval,
        "bars": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "close_latest": _sf(close.iloc[-1]),
        "close_range": [_sf(close.min()), _sf(close.max())],
        "ann_volatility_pct": _sf(returns.std() * np.sqrt(252 * 24) * 100),
        "total_return_pct": _sf((close.iloc[-1] / close.iloc[0] - 1) * 100),
    }


@mcp.tool()
def set_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge a dict of StrategyParams fields into the current session state.

    Call this with any subset of StrategyParams fields — unmentioned params
    are preserved.  Also accepts "capital", "commission", "slippage" as
    execution-cost overrides.

    Example
    -------
    set_params({
        "pamrp_enabled": True,   "pamrp_entry_long": 25,
        "bbwp_enabled":  True,   "bbwp_threshold_long": 50,
        "stop_loss_enabled": True, "stop_loss_pct_long": 3.0,
        "trade_direction": "long_only",
        "capital": 50000,
    })
    """
    # Pull out execution-cost keys so they don't pollute StrategyParams
    for key in ("capital", "commission", "slippage"):
        if key in params:
            _state[key] = float(params.pop(key))

    _state["params"].update(params)

    return {
        "status": "ok",
        "execution": {
            "capital": _state["capital"],
            "commission": _state["commission"],
            "slippage": _state["slippage"],
        },
        "strategy_params": _state["params"],
    }


@mcp.tool()
def run_backtest() -> Dict[str, Any]:
    """
    Run a backtest with the current data and params.

    Data must be loaded first via load_data().
    Params are set via set_params().

    Returns a full metrics dict: return, CAGR, Sharpe, Sortino, Calmar,
    max drawdown, win rate, profit factor, expectancy, payoff ratio,
    and % time in market.
    """
    if (err := _require_data()):
        return err

    try:
        engine = BacktestEngine(
            params_to_strategy(_state["params"]),
            _state["capital"],
            _state["commission"],
            _state["slippage"],
        )
        results = engine.run(_state["df"].copy())
        metrics = _results_to_dict(results)
        _state["last_metrics"] = metrics
        return {"status": "ok", "metrics": metrics}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def run_optimize(
    metric: str = "sharpe_ratio",
    n_trials: int = 100,
    use_walkforward: bool = False,
    n_folds: int = 5,
    trade_direction: str = "long_only",
    train_pct: float = 0.7,
    window_type: str = "rolling",
) -> Dict[str, Any]:
    """
    Run Bayesian optimization (Optuna TPE) on the loaded data.

    Enabled indicators are inferred from the current params (_enabled flags).
    Best params from the LAST walk-forward fold are written back to state
    (avoids selection bias — consistent with the app's design).

    Parameters
    ----------
    metric         : Objective — "sharpe_ratio", "profit_factor", "cagr", etc.
    n_trials       : Optuna trial budget per fold.
    use_walkforward: If True, use rolling/anchored walk-forward; else simple split.
    n_folds        : Number of WFO folds (ignored if use_walkforward=False).
    trade_direction: "long_only" | "short_only" | "both".
    train_pct      : Fraction of data used for training (simple split).
    window_type    : "rolling" or "anchored" (WFO only).

    Returns
    -------
    best_value, train/test values, efficiency ratio, warnings, full-data
    metrics (IS-contaminated — informational only), and best_params written
    into state ready for run_backtest().
    """
    if (err := _require_data()):
        return err

    filters = _enabled_filters()
    if not any(filters.values()):
        return {
            "error": (
                "No indicators enabled. "
                "Call set_params({'pamrp_enabled': True, ...}) first."
            )
        }

    try:
        result = optimize_strategy(
            df=_state["df"].copy(),
            enabled_filters=filters,
            metric=metric,
            n_trials=n_trials,
            initial_capital=_state["capital"],
            commission_pct=_state["commission"],
            slippage_pct=_state["slippage"],
            trade_direction=trade_direction,
            train_pct=train_pct,
            use_walkforward=use_walkforward,
            n_folds=n_folds,
            window_type=window_type,
            show_progress=False,
        )

        # Write best params back to state so run_backtest() uses them immediately
        if result.best_params:
            _state["params"].update(result.best_params)

        return {
            "status": "ok",
            "metric": metric,
            "best_value": _sf(result.best_value),
            "train_value": _sf(result.train_value),
            "test_value": _sf(result.test_value),
            "efficiency_ratio": _sf(result.efficiency_ratio),
            "failed_trial_pct": _sf(result.failed_trial_pct),
            "warnings": result.warnings or [],
            "full_data_metrics": _results_to_dict(result.full_data_results),
            "best_params": result.best_params,
            "note": (
                "best_params written to state. "
                "Call run_backtest() to see full metrics on these params."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def run_permutation_test(
    n_permutations: int = 100,
    n_trials: int = 50,
    metric: str = "profit_factor",
    trade_direction: str = "long_only",
) -> Dict[str, Any]:
    """
    In-sample permutation test — does the strategy have a real edge?

    Procedure:
      1. Optimize on real data → record metric.
      2. Shuffle returns N times (destroying signal, preserving distribution).
      3. Re-optimize on each shuffle.
      4. p-value = fraction of permuted metrics >= real metric.

    p < 0.05 → statistically significant edge.
    p >= 0.05 → cannot distinguish from noise; suspect overfitting.

    Parameters
    ----------
    n_permutations : Number of shuffled datasets (100 = reasonable, 500 = thorough).
    n_trials       : Optuna trials per permutation (keep low for speed, e.g. 50).
    metric         : "profit_factor" | "sharpe_ratio" | "cagr" | etc.
    trade_direction: "long_only" | "short_only" | "both".
    """
    if (err := _require_data()):
        return err

    filters = _enabled_filters()

    try:
        result = _run_permutation_test(
            df=_state["df"].copy(),
            enabled_filters=filters,
            metric=metric,
            n_permutations=n_permutations,
            n_trials=n_trials,
            trade_direction=trade_direction,
            initial_capital=_state["capital"],
            commission_pct=_state["commission"],
            slippage_pct=_state["slippage"],
        )

        if result is None:
            return {"error": "Insufficient data or trades for permutation test."}

        p = result.p_value
        interpretation = (
            f"p={p:.3f} — strong evidence of real edge (p<0.01)"       if p < 0.01 else
            f"p={p:.3f} — edge is statistically significant (p<0.05)"  if p < 0.05 else
            f"p={p:.3f} — no significant edge detected; possible overfit"
        )

        return {
            "status": "ok",
            "p_value": _sf(p),
            "significant": p < 0.05,
            "real_metric": _sf(result.real_metric),
            "permuted_median": _sf(float(np.median(result.permuted_metrics))),
            "permuted_p95": _sf(float(np.percentile(result.permuted_metrics, 95))),
            "n_permutations": result.n_permutations,
            "real_num_trades": result.real_num_trades,
            "avg_permuted_trades": _sf(result.avg_permuted_trades),
            "metric": result.metric_name,
            "interpretation": interpretation,
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_market_characterization() -> Dict[str, Any]:
    """
    Characterize the loaded market data across three dimensions:

    - Volatility regime: ATR/price ratio percentile rank → low / normal / high
    - Trend: price vs SMA50/SMA200 → bullish / bearish + strength
    - ADX: current ADX value → weak (<20) / moderate (20-40) / strong (>40) trend

    No params required. Uses the loaded OHLCV data.
    Useful for deciding which strategy profile to test.
    """
    if (err := _require_data()):
        return err

    df = _state["df"]
    close, high, low = df["close"], df["high"], df["low"]
    returns = close.pct_change().dropna()

    # ── Volatility via ATR/price ratio ────────────────────────────────────────
    try:
        atr_series = _atr(high, low, close, 14)
        atr_pct    = (atr_series / close * 100).dropna()
        curr_atr   = float(atr_pct.iloc[-1])
        vol_rank   = float((atr_pct < atr_pct.iloc[-1]).mean() * 100)
        vol_regime = "low" if vol_rank < 33 else "high" if vol_rank > 66 else "normal"
    except Exception:
        curr_atr = vol_rank = None
        vol_regime = "unknown"

    # ── Trend via SMA50 / SMA200 ──────────────────────────────────────────────
    try:
        n = len(close)
        sma50  = _sma(close, min(50,  n // 4))
        sma200 = _sma(close, min(200, n // 2))
        c_last   = float(close.iloc[-1])
        s50_last = float(sma50.iloc[-1])
        s200_last= float(sma200.iloc[-1])
        trend    = "bullish" if c_last > s200_last else "bearish"
        gap_pct  = abs(c_last / s200_last - 1) * 100
        strength = "strong" if gap_pct > 5 else "moderate" if gap_pct > 2 else "weak"
    except Exception:
        s50_last = s200_last = gap_pct = None
        trend = strength = "unknown"

    # ── ADX ───────────────────────────────────────────────────────────────────
    try:
        _, _, adx_series = _adx(high, low, close, 14, 14)
        curr_adx  = float(adx_series.dropna().iloc[-1])
        adx_label = (
            "weak trend"     if curr_adx < 20 else
            "moderate trend" if curr_adx < 40 else
            "strong trend"
        )
    except Exception:
        curr_adx  = None
        adx_label = "unknown"

    return {
        "status": "ok",
        "symbol": _state["symbol"],
        "bars": len(df),
        "volatility": {
            "atr_pct_current": _sf(curr_atr),
            "percentile_rank":  _sf(vol_rank),
            "regime":           vol_regime,
            "ann_vol_pct":      _sf(float(returns.std() * np.sqrt(252) * 100)),
        },
        "trend": {
            "direction":  trend,
            "strength":   strength,
            "gap_vs_sma200_pct": _sf(gap_pct),
            "sma50":  _sf(s50_last),
            "sma200": _sf(s200_last),
        },
        "adx": {
            "value": _sf(curr_adx),
            "label": adx_label,
        },
        "buy_and_hold": {
            "total_return_pct": _sf(
                (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
            ),
            "naive_sharpe": _sf(
                float(returns.mean() / returns.std() * np.sqrt(252))
                if returns.std() > 0 else 0.0
            ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport — required for Claude Desktop