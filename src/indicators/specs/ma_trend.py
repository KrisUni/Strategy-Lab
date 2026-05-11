"""MA Trend structural bias filter spec."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import ma


def compute_ma_trend(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["ma_fast"] = ma(df["close"], params["ma_fast_length"], params["ma_type"])
    df["ma_slow"] = ma(df["close"], params["ma_slow_length"], params["ma_type"])
    return df


def long_signal_ma_trend(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["ma_fast"] > df["ma_slow"]


def short_signal_ma_trend(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["ma_fast"] < df["ma_slow"]


def render_ma_trend(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col) if ctx.is_subplot else {}
    ma_type = ctx.params.get("ma_type", "sma").upper()
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["ma_fast"], mode="lines",
        line=dict(color=ctx.palette.primary, width=1.2),
        name=f"{ma_type}({ctx.params.get('ma_fast_length', 50)})",
        showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["ma_slow"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1.2),
        name=f"{ma_type}({ctx.params.get('ma_slow_length', 200)})",
        showlegend=True,
    ), **rn)


register(IndicatorSpec(
    key="ma_trend",
    name="MA Trend",
    group="entry",
    order=4,
    enable_param="ma_trend_enabled",
    params=[
        ParamSpec("ma_trend_enabled", "bool", False, optimize=False,
                  label="MA Trend enabled", order=0),
        ParamSpec("ma_fast_length", "int", 50, min=1, max=200,
                  label="Fast MA length", order=1),
        ParamSpec("ma_slow_length", "int", 200, min=2, max=500,
                  label="Slow MA length", order=2),
        ParamSpec("ma_type", "categorical", "sma",
                  choices=("sma", "ema", "wma", "rma"),
                  label="MA type", order=3),
    ],
    compute=compute_ma_trend,
    outputs=["ma_fast", "ma_slow"],
    long_signal=long_signal_ma_trend,
    short_signal=short_signal_ma_trend,
    plot=PlotSpec(
        kind="overlay",
        render=render_ma_trend,
    ),
))
