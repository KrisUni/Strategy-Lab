"""MA crossover exit spec — self-contained with independent compute params.

Owns its own computation params (fast, slow, ma_type). Opportunistically
reuses ma_trend's output columns when params match. Writes to its own
columns 'ma_fast_exit' and 'ma_slow_exit'.

Default: EMA(10)/EMA(20) — a fast tactical crossover on top of a slow trend
filter (e.g. SMA(50)/SMA(200) entry). The differing defaults make the common
use case work out-of-the-box without further configuration.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import ma


def compute_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("ma_trend_enabled", False)
        and params["ma_exit_fast"] == params["ma_fast_length"]
        and params["ma_exit_slow"] == params["ma_slow_length"]
        and params["ma_exit_type"] == params["ma_type"]
        and "ma_fast" in df.columns
        and "ma_slow" in df.columns
    )
    if entry_reusable:
        df["ma_fast_exit"] = df["ma_fast"]
        df["ma_slow_exit"] = df["ma_slow"]
    else:
        df["ma_fast_exit"] = ma(df["close"], params["ma_exit_fast"], params["ma_exit_type"])
        df["ma_slow_exit"] = ma(df["close"], params["ma_exit_slow"], params["ma_exit_type"])
    return df


def long_signal_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["ma_fast_exit"] < df["ma_slow_exit"]


def short_signal_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["ma_fast_exit"] > df["ma_slow_exit"]


def render_ma_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col) if ctx.is_subplot else {}
    ma_type = ctx.params.get("ma_exit_type", "ema").upper()
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["ma_fast_exit"], mode="lines",
        line=dict(color=ctx.palette.primary, width=1.2),
        name=f"Exit {ma_type}({ctx.params.get('ma_exit_fast', 10)})",
        showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["ma_slow_exit"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1.2),
        name=f"Exit {ma_type}({ctx.params.get('ma_exit_slow', 20)})",
        showlegend=True,
    ), **rn)


def contribute_ma_exit(ctx: PlotContext) -> None:
    same_config = (
        ctx.params.get("ma_exit_fast", 10)      == ctx.params.get("ma_fast_length", 50)
        and ctx.params.get("ma_exit_slow", 20)  == ctx.params.get("ma_slow_length", 200)
        and ctx.params.get("ma_exit_type", "ema") == ctx.params.get("ma_type", "sma")
    )
    if not same_config:
        render_ma_exit(ctx)


register(IndicatorSpec(
    key="ma_exit",
    name="MA Exit",
    group="exit",
    order=3,
    enable_param="ma_exit_enabled",
    params=[
        ParamSpec("ma_exit_enabled", "bool", False, optimize=False,
                  label="MA exit enabled", order=0),
        ParamSpec("ma_exit_fast", "int", 10, min=1, max=100,
                  label="Fast length", order=1),
        ParamSpec("ma_exit_slow", "int", 20, min=2, max=200,
                  label="Slow length", order=2),
        ParamSpec("ma_exit_type", "categorical", "ema",
                  choices=("sma", "ema", "wma", "rma"),
                  label="MA type", order=3),
    ],
    compute=compute_ma_exit,
    outputs=["ma_fast_exit", "ma_slow_exit"],
    long_signal=long_signal_ma_exit,
    short_signal=short_signal_ma_exit,
    reuses_outputs_from=["ma_trend"],
    plot=PlotSpec(
        kind="overlay",
        render=render_ma_exit,
        contribute=contribute_ma_exit,
    ),
))
