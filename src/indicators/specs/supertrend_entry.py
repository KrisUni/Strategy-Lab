"""Supertrend ATR-based trend signal spec."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import supertrend as _supertrend


def compute_supertrend(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["supertrend"], df["st_direction"] = _supertrend(
        df["high"], df["low"], df["close"],
        params["supertrend_period"], params["supertrend_multiplier"],
    )
    return df


def long_signal_supertrend(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    # +1 = bullish (price above upper band), -1 = bearish
    return df["st_direction"] > 0


def short_signal_supertrend(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["st_direction"] < 0


def render_supertrend(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col) if ctx.is_subplot else {}
    bull = ctx.idf["supertrend"].where(ctx.idf["st_direction"] > 0)
    bear = ctx.idf["supertrend"].where(ctx.idf["st_direction"] < 0)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=bull, mode="lines",
        line=dict(color=ctx.palette.bullish, width=1.5),
        name="Supertrend ▲", showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=bear, mode="lines",
        line=dict(color=ctx.palette.bearish, width=1.5),
        name="Supertrend ▼", showlegend=True,
    ), **rn)


register(IndicatorSpec(
    key="supertrend_entry",
    name="Supertrend",
    group="entry",
    order=7,
    enable_param="supertrend_enabled",
    params=[
        ParamSpec("supertrend_enabled", "bool", False, optimize=False,
                  label="Supertrend enabled", order=0),
        ParamSpec("supertrend_period", "int", 10, min=2, max=50,
                  label="Period", order=1),
        ParamSpec("supertrend_multiplier", "float", 3.0, min=0.5, max=10.0, step=0.5,
                  label="Multiplier", order=2),
    ],
    compute=compute_supertrend,
    outputs=["supertrend", "st_direction"],
    long_signal=long_signal_supertrend,
    short_signal=short_signal_supertrend,
    plot=PlotSpec(
        kind="overlay",
        render=render_supertrend,
        owner_for_columns=["supertrend", "st_direction"],
    ),
))
