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

import ast
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import numpy as np
from fastmcp import FastMCP

from src.backtest import BacktestEngine, DEFAULT_COMMISSION_PCT, DEFAULT_SLIPPAGE_PCT
from src.data import fetch_yfinance, generate_sample_data as _generate_sample_data
from src.indicators import adx as _adx, atr as _atr, sma as _sma
import src.indicators as _ind_module
from src.indicators.registry import (
    INDICATOR_REGISTRY, IndicatorSpec, ParamSpec,
    validate_registry as _validate_registry, _PROVISIONAL_KEYS,
)
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
        "num_trades":             r.num_trades,
        "total_return_pct":       _sf(r.total_return_pct),
        "cagr":                   _sf(r.cagr),
        "sharpe_ratio":           _sf(r.sharpe_ratio),
        "sortino_ratio":          _sf(r.sortino_ratio),
        "calmar_ratio":           _sf(r.calmar_ratio),
        "max_drawdown_pct":       _sf(r.max_drawdown_pct),
        "win_rate":               _sf(r.win_rate),
        "profit_factor":          _sf(r.profit_factor),
        "expectancy":             _sf(r.expectancy),
        "payoff_ratio":           _sf(r.payoff_ratio),
        "pct_time_in_market":     _sf(r.pct_time_in_market),
        "avg_winner_pct":         _sf(r.avg_winner_pct),
        "avg_loser_pct":          _sf(r.avg_loser_pct),
        "avg_bars_held":          _sf(r.avg_bars_held),
        "max_consecutive_losses": r.max_consecutive_losses,
        "max_consecutive_wins":   r.max_consecutive_wins,
        "longest_drawdown_bars":  r.longest_drawdown_bars,
    }


def _trade_to_dict(t) -> Dict[str, Any]:
    """Serialize a Trade object to a plain dict."""
    return {
        "entry_date":  str(t.entry_date),
        "exit_date":   str(t.exit_date) if t.exit_date else None,
        "direction":   t.direction,
        "entry_price": _sf(t.entry_price),
        "exit_price":  _sf(t.exit_price) if t.exit_price else None,
        "pnl_pct":     _sf(t.pnl_pct),
        "pnl":         _sf(t.pnl),
        "bars_held":   t.bars_held,
        "exit_reason": t.exit_reason,
        "mae":         _sf(t.mae),
        "mfe":         _sf(t.mfe),
    }


def _direction_split(trades) -> Dict[str, Any]:
    """Break trade results down by long vs short direction."""
    def _stats(subset):
        if not subset:
            return {"num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_pnl_pct": 0.0}
        winners = [t for t in subset if t.pnl > 0]
        losers  = [t for t in subset if t.pnl <= 0]
        gp = sum(t.pnl for t in winners)
        gl = abs(sum(t.pnl for t in losers))
        return {
            "num_trades":    len(subset),
            "win_rate":      _sf(len(winners) / len(subset) * 100),
            "profit_factor": _sf(gp / gl if gl > 0 else (999.99 if gp > 0 else 0.0)),
            "avg_pnl_pct":   _sf(sum(t.pnl_pct for t in subset) / len(subset)),
        }
    longs  = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    return {"long": _stats(longs), "short": _stats(shorts)}


def _period_stats(trades, freq: str = "M") -> list:
    """Monthly (freq='M') or quarterly (freq='Q') performance breakdown."""
    import pandas as pd
    if not trades:
        return []
    records = [
        {"period": t.exit_date, "pnl": t.pnl, "pnl_pct": t.pnl_pct, "win": t.pnl > 0}
        for t in trades if t.exit_date
    ]
    if not records:
        return []
    df = pd.DataFrame(records)
    df["period"] = pd.to_datetime(df["period"]).dt.to_period(freq)
    out = []
    for period, grp in df.groupby("period"):
        winners = grp[grp["win"]]
        losers  = grp[~grp["win"]]
        gp = float(winners["pnl"].sum())
        gl = float(abs(losers["pnl"].sum()))
        out.append({
            "period":        str(period),
            "num_trades":    int(len(grp)),
            "win_rate":      _sf(len(winners) / len(grp) * 100),
            "profit_factor": _sf(gp / gl if gl > 0 else (999.99 if gp > 0 else 0.0)),
            "total_pnl_pct": _sf(float(grp["pnl_pct"].sum())),
        })
    return out


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
def run_backtest(
    include_trades: bool = False,
    include_periods: bool = False,
) -> Dict[str, Any]:
    """
    Run a backtest with the current data and params.

    Data must be loaded first via load_data().
    Params are set via set_params().

    Returns a full metrics dict: return, CAGR, Sharpe, Sortino, Calmar,
    max drawdown, win rate, profit factor, expectancy, payoff ratio,
    and % time in market.

    Optional flags:
    - include_trades=True  → add per-trade log and long/short direction split
    - include_periods=True → add monthly P&L breakdown
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
        out: Dict[str, Any] = {"status": "ok", "metrics": metrics}
        if include_trades:
            out["trades"] = [_trade_to_dict(t) for t in results.trades]
            out["direction_split"] = _direction_split(results.trades)
        if include_periods:
            out["monthly"] = _period_stats(results.trades, "M")
        return out
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

        fold_results = [
            {
                "fold":         f.fold_num,
                "train_period": f"{f.train_start.date()} → {f.train_end.date()}",
                "test_period":  f"{f.test_start.date()} → {f.test_end.date()}",
                "train_value":  _sf(f.train_value),
                "test_value":   _sf(f.test_value),
                "train_trades": f.train_trades,
                "test_trades":  f.test_trades,
            }
            for f in (result.walkforward_folds or [])
        ]
        return {
            "status": "ok",
            "metric": metric,
            "best_value": _sf(result.best_value),
            "train_value": _sf(result.train_value),
            "test_value": _sf(result.test_value),
            "efficiency_ratio": _sf(result.efficiency_ratio),
            "failed_trial_pct": _sf(result.failed_trial_pct),
            "warnings": result.warnings or [],
            "fold_results": fold_results,
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

@mcp.tool()
def run_cost_sweep(
    round_trip_pcts: List[float] = [0.3, 0.5, 1.0, 2.0, 3.0],
) -> Dict[str, Any]:
    """
    Run the current strategy at multiple round-trip cost levels.

    Reveals the breakeven cost point and how much margin the edge has.
    Each round-trip cost is split evenly: commission = slippage = RT/4 per leg.
    E.g. 1.0% round-trip → 0.25% commission + 0.25% slippage per leg.

    Use this to answer: 'Is the edge real, or does it only survive at unrealistic costs?'
    """
    if (err := _require_data()):
        return err

    results_out = []
    for rt in round_trip_pcts:
        per_leg = rt / 4.0
        try:
            engine = BacktestEngine(
                params_to_strategy(_state["params"]),
                _state["capital"],
                per_leg,   # commission
                per_leg,   # slippage
            )
            r = engine.run(_state["df"].copy())
            results_out.append({
                "round_trip_pct": rt,
                **_results_to_dict(r),
            })
        except Exception as exc:
            results_out.append({"round_trip_pct": rt, "error": str(exc)})

    return {"status": "ok", "cost_sweep": results_out}


@mcp.tool()
def run_sensitivity(
    metric: str = "profit_factor",
    delta_pct: float = 25.0,
) -> Dict[str, Any]:
    """
    Vary each active indicator param by ±delta_pct from its current value.

    Runs a backtest for each variation and reports how the target metric changes.
    High sensitivity = fragile (overfit). Low sensitivity = robust plateau.

    Only varies params for enabled indicators (respects *_enabled flags).
    Results sorted by sensitivity descending — most fragile params first.

    Parameters
    ----------
    metric    : metric to track — 'profit_factor', 'sharpe_ratio', 'cagr', etc.
    delta_pct : percentage change applied to each param (default ±25%)
    """
    if (err := _require_data()):
        return err

    filters = _enabled_filters()

    INDICATOR_PARAMS: Dict[str, List[str]] = {
        "pamrp_enabled":        ["pamrp_entry_ma_length", "pamrp_entry_lookback", "pamrp_entry_ma_type",
                                  "pamrp_entry_long", "pamrp_entry_short",
                                  "pamrp_exit_ma_length", "pamrp_exit_lookback", "pamrp_exit_ma_type",
                                  "pamrp_exit_long", "pamrp_exit_short"],
        "bbwp_enabled":         ["bbwp_length", "bbwp_lookback", "bbwp_sma_length",
                                  "bbwp_threshold_long", "bbwp_threshold_short"],
        "adx_enabled":          ["adx_length", "adx_smoothing", "adx_threshold"],
        "ma_trend_enabled":     ["ma_fast_length", "ma_slow_length"],
        "rsi_enabled":          ["rsi_length", "rsi_oversold", "rsi_overbought"],
        "supertrend_enabled":   ["supertrend_period", "supertrend_multiplier"],
        "stop_loss_enabled":    ["stop_loss_pct_long", "stop_loss_pct_short"],
        "take_profit_enabled":  ["take_profit_pct_long", "take_profit_pct_short"],
        "trailing_stop_enabled": ["trailing_stop_pct", "trailing_stop_activation"],
        "time_exit_enabled":    ["time_exit_bars_long", "time_exit_bars_short"],
        "atr_trailing_enabled": ["atr_length", "atr_multiplier"],
    }

    active_params: List[str] = []
    for flag, param_names in INDICATOR_PARAMS.items():
        if filters.get(flag, False):
            active_params.extend(param_names)

    if not active_params:
        return {"error": "No indicators enabled — nothing to vary."}

    # Baseline
    try:
        base_engine = BacktestEngine(
            params_to_strategy(_state["params"]),
            _state["capital"],
            _state["commission"],
            _state["slippage"],
        )
        base_r = base_engine.run(_state["df"].copy())
        base_val = getattr(base_r, metric, None)
        if base_val is None:
            return {"error": f"Unknown metric '{metric}'. Use profit_factor, sharpe_ratio, cagr, etc."}
    except Exception as exc:
        return {"error": f"Baseline backtest failed: {exc}"}

    rows = []
    for param in active_params:
        base_param_val = _state["params"].get(param)
        if base_param_val is None or base_param_val == 0:
            continue
        is_int = isinstance(base_param_val, int)
        delta = abs(base_param_val) * delta_pct / 100.0

        lo_raw = base_param_val - delta
        hi_raw = base_param_val + delta
        lo_val = max(1, int(round(lo_raw))) if is_int else round(lo_raw, 4)
        hi_val = max((lo_val + 1) if is_int else lo_val + 0.01, int(round(hi_raw)) if is_int else round(hi_raw, 4))

        row: Dict[str, Any] = {
            "param":    param,
            "base":     base_param_val,
            "low_val":  lo_val,
            "high_val": hi_val,
        }
        for label, test_val in [("low", lo_val), ("high", hi_val)]:
            test_params = dict(_state["params"])
            test_params[param] = test_val
            try:
                eng = BacktestEngine(
                    params_to_strategy(test_params),
                    _state["capital"],
                    _state["commission"],
                    _state["slippage"],
                )
                r = eng.run(_state["df"].copy())
                row[f"{label}_{metric}"] = _sf(getattr(r, metric))
            except Exception:
                row[f"{label}_{metric}"] = None

        lo_m = row.get(f"low_{metric}")
        hi_m = row.get(f"high_{metric}")
        row["sensitivity"] = _sf(abs(hi_m - lo_m)) if (lo_m is not None and hi_m is not None) else None
        rows.append(row)

    rows.sort(key=lambda r: r.get("sensitivity") or 0.0, reverse=True)
    return {
        "status":    "ok",
        "metric":    metric,
        "baseline":  _sf(base_val),
        "delta_pct": delta_pct,
        "params":    rows,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Research log
# ─────────────────────────────────────────────────────────────────────────────

_LOG_PATH = Path(__file__).parent.parent / "RESEARCH_LOG.md"


@mcp.tool()
def log_research_result(
    thesis: str,
    indicators: str,
    verdict: str,
    symbol: str = "",
    interval: str = "",
    regime: str = "",
    efficiency_ratio: Optional[float] = None,
    p_value: Optional[float] = None,
    failure_reason: str = "",
    iteration_history: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """
    Append a completed research cycle to RESEARCH_LOG.md.

    Call at the end of each full validation cycle (baseline → optimize →
    permutation test → verdict). Pulls current metrics from
    _state["last_metrics"] and date range from the loaded dataframe.

    Parameters
    ----------
    thesis           : One sentence — what edge and why it should exist.
    indicators       : Entry and exit indicators (e.g. "PAMRP + BBWP, ATR exit").
    verdict          : "VIABLE" | "MARGINAL" | "DISCARD"
    symbol           : Override symbol (defaults to last loaded symbol).
    interval         : Override interval (defaults to last loaded interval).
    regime           : Free-text regime summary from get_market_characterization.
    efficiency_ratio : OOS efficiency ratio from run_optimize (if run).
    p_value          : p-value from run_permutation_test (if run).
    failure_reason   : Why discarded/marginal (e.g. "p=0.38, trade count 12").
    iteration_history: What was changed and why, one line per step.
    notes            : Any additional notes.
    """
    sym = symbol or _state.get("symbol") or "UNKNOWN"
    ivl = interval or _state.get("interval") or "unknown"
    today = date.today().isoformat()

    metrics = _state.get("last_metrics") or {}
    df = _state.get("df")
    if df is not None and len(df) > 0:
        idx = df.index
        date_start = str(idx[0])[:10]
        date_end   = str(idx[-1])[:10]
        n_bars     = len(df)
    else:
        date_start = date_end = "unknown"
        n_bars = 0

    def _fmt(key: str, pct: bool = False) -> str:
        v = metrics.get(key)
        if v is None:
            return "—"
        return f"{v:.2f}%" if pct else f"{v:.4f}"

    eff_str  = f"{efficiency_ratio:.4f}" if efficiency_ratio is not None else "—"
    pval_str = f"{p_value:.4f}"          if p_value          is not None else "—"

    entry = (
        f"\n---\n"
        f"## {sym} {ivl} — {today} — {verdict}\n\n"
        f"**Thesis:** {thesis}\n"
        f"**Indicators:** {indicators}\n"
        f"**Regime at test time:** {regime or chr(8212)}\n"
        f"**Date range:** {date_start} → {date_end} ({n_bars} bars)\n\n"
        f"| Metric           | Value      |\n"
        f"|------------------|------------|\n"
        f"| Trades           | {metrics.get('num_trades', chr(8212))} |\n"
        f"| Total return     | {_fmt('total_return_pct', True)} |\n"
        f"| CAGR             | {_fmt('cagr', True)} |\n"
        f"| Sharpe           | {_fmt('sharpe_ratio')} |\n"
        f"| Max DD           | {_fmt('max_drawdown_pct', True)} |\n"
        f"| Profit Factor    | {_fmt('profit_factor')} |\n"
        f"| Win Rate         | {_fmt('win_rate', True)} |\n"
        f"| Expectancy       | {_fmt('expectancy')} |\n"
        f"| Efficiency Ratio | {eff_str} |\n"
        f"| p-value          | {pval_str} |\n\n"
        f"**Verdict:** {verdict}\n"
        f"**Failure reason:** {failure_reason or chr(8212)}\n"
        f"**Iteration history:** {iteration_history or chr(8212)}\n"
        f"**Notes:** {notes or chr(8212)}\n"
    )

    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(entry)
        return {
            "status": "ok",
            "logged": f"{sym} {ivl} — {today} — {verdict}",
            "path":   str(_LOG_PATH),
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_research_history(
    symbol: Optional[str] = None,
    verdict_filter: Optional[str] = None,
    last_n: int = 20,
) -> Dict[str, Any]:
    """
    Read and summarise RESEARCH_LOG.md.

    Returns past research entries filtered by symbol and/or verdict.
    Call at session start before proposing a hypothesis — lets Claude avoid
    re-testing already-disqualified ideas and surface unexplored regimes.

    Parameters
    ----------
    symbol        : Filter by symbol/interval string (case-insensitive, partial).
    verdict_filter: "VIABLE" | "MARGINAL" | "DISCARD" | None (return all).
    last_n        : Maximum entries to return, most recent first (default 20).
    """
    if not _LOG_PATH.exists():
        return {"status": "ok", "entries": [], "total": 0,
                "note": "No research log yet — this symbol has never been tested."}

    try:
        text = _LOG_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        return {"error": str(exc)}

    raw_entries = [e.strip() for e in text.split("\n---\n") if e.strip()]

    entries: List[Dict[str, Any]] = []
    for raw in raw_entries:
        lines = raw.splitlines()
        header = next((l for l in lines if l.startswith("## ")), None)
        if not header:
            continue
        parts = header[3:].split(" — ")
        if len(parts) < 3:
            continue
        sym_ivl        = parts[0].strip()
        entry_date     = parts[1].strip()
        entry_verdict  = parts[2].strip()

        if symbol and symbol.upper() not in sym_ivl.upper():
            continue
        if verdict_filter and entry_verdict.upper() != verdict_filter.upper():
            continue

        entries.append({
            "symbol_interval": sym_ivl,
            "date":            entry_date,
            "verdict":         entry_verdict,
            "raw":             raw,
        })

    entries.reverse()           # most recent first
    entries = entries[:last_n]

    return {"status": "ok", "total": len(entries), "entries": entries}

# ─────────────────────────────────────────────────────────────────────────────
# Indicator registry tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_indicators() -> Dict[str, Any]:
    """
    Returns all registered indicators — key, name, group, enabled status in
    current params, and whether provisional.
    Call this before creating an indicator to confirm it doesn't already exist.
    """
    indicators = []
    for spec in INDICATOR_REGISTRY:
        indicators.append({
            "key":         spec.key,
            "name":        spec.name,
            "group":       spec.group,
            "order":       spec.order,
            "enabled":     bool(_state["params"].get(spec.enable_param, False)),
            "provisional": spec.key in _PROVISIONAL_KEYS,
            "params": [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in spec.params
            ],
        })
    return {"indicators": indicators, "count": len(indicators)}


@mcp.tool()
def register_indicator(spec_source: str) -> Dict[str, Any]:
    """
    Validates and registers a new indicator at runtime.

    spec_source: complete Python source defining compute/signal functions and
    calling register(IndicatorSpec(...)). Do NOT include import statements —
    use pd, np, IndicatorSpec, ParamSpec, register, and any src.indicators
    compute functions (sma, ema, rsi, atr, etc.) which are pre-loaded.

    Validation (in order):
    1. Syntax check
    2. Import whitelist — pandas, numpy, scipy, src.indicators only
    3. exec in restricted namespace — must call register(IndicatorSpec(...))
    4. Duplicate key check
    5. Schema validation via validate_registry()
    6. Output column check on 100-bar synthetic data

    On success: indicator is provisional, available immediately in set_params/run_backtest.
    """
    _ALLOWED_IMPORTS = {
        "pandas", "numpy", "scipy", "scipy.stats",
        "src.indicators", "src.indicators.registry",
    }

    # Step 1: syntax check
    try:
        compiled = compile(spec_source, "<mcp_indicator>", "exec")
        tree = ast.parse(spec_source)
    except SyntaxError as e:
        return {"error": "syntax_error", "detail": str(e)}

    # Step 2: import whitelist
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if not any(name == a or name.startswith(a + ".") for a in _ALLOWED_IMPORTS):
                    return {"error": "disallowed_import", "detail": name}
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if not any(name == a or name.startswith(a + ".") for a in _ALLOWED_IMPORTS):
                return {"error": "disallowed_import", "detail": name}

    # Step 3: exec in restricted namespace
    _captured: List[Any] = []

    def _capture_register(s: Any) -> None:
        _captured.append(s)

    _ind_fns = {
        n: getattr(_ind_module, n)
        for n in dir(_ind_module)
        if not n.startswith("_")
    }

    namespace: Dict[str, Any] = {
        "__builtins__": {
            "len": len, "abs": abs, "round": round, "range": range,
            "enumerate": enumerate, "zip": zip, "list": list, "dict": dict,
            "tuple": tuple, "set": set, "float": float, "int": int,
            "bool": bool, "str": str, "None": None, "True": True, "False": False,
            "isinstance": isinstance, "getattr": getattr, "hasattr": hasattr,
            "min": min, "max": max, "sum": sum, "print": print,
        },
        "pd":           pd,
        "np":           np,
        "IndicatorSpec": IndicatorSpec,
        "ParamSpec":    ParamSpec,
        "register":     _capture_register,
        **_ind_fns,
    }

    try:
        exec(compiled, namespace)
    except Exception as e:
        return {"error": "exec_error", "detail": str(e)}

    spec = _captured[0] if _captured else namespace.get("spec")
    if spec is None:
        return {
            "error": "no_spec_produced",
            "detail": "Source must call register(IndicatorSpec(...)) or assign result to 'spec'",
        }
    if not isinstance(spec, IndicatorSpec):
        return {
            "error": "no_spec_produced",
            "detail": f"Expected IndicatorSpec, got {type(spec).__name__}",
        }

    # Step 4: duplicate key check
    if any(s.key == spec.key for s in INDICATOR_REGISTRY):
        return {"error": "duplicate_key", "detail": spec.key}

    # Step 5: schema validation (append temporarily, validate, rollback on failure)
    INDICATOR_REGISTRY.append(spec)
    try:
        _validate_registry()
    except ValueError as e:
        INDICATOR_REGISTRY.pop()
        return {"error": "schema_error", "detail": str(e)}

    # Step 6: output column check
    try:
        test_df = _generate_sample_data(days=100, seed=0)
        defaults = {p.name: p.default for p in spec.params}
        result_df = spec.compute(test_df.copy(), defaults)
        missing = [col for col in spec.outputs if col not in result_df.columns]
        if missing:
            INDICATOR_REGISTRY.pop()
            return {"error": "output_columns_missing", "detail": missing}
    except Exception as e:
        INDICATOR_REGISTRY.pop()
        return {"error": "compute_error", "detail": str(e)}

    # Registration complete — mark provisional and seed defaults into state
    _PROVISIONAL_KEYS.add(spec.key)
    for p in spec.params:
        if p.name not in _state["params"]:
            _state["params"][p.name] = p.default

    return {
        "status":      "ok",
        "key":         spec.key,
        "name":        spec.name,
        "group":       spec.group,
        "provisional": True,
        "params": [
            {"name": p.name, "type": p.type, "default": p.default}
            for p in spec.params
        ],
    }


if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport — required for Claude Desktop