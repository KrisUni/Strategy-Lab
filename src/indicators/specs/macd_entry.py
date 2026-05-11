"""MACD momentum entry spec."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import macd as _macd


def compute_macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(
        df["close"], params["macd_fast"], params["macd_slow"], params["macd_signal"],
    )
    return df


def long_signal_macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_mode"]
    if mode == "histogram":
        return df["macd_hist"] > 0
    elif mode == "crossover":
        return df["macd"] > df["macd_signal"]
    else:   # zero-line
        return df["macd"] > 0


def short_signal_macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_mode"]
    if mode == "histogram":
        return df["macd_hist"] < 0
    elif mode == "crossover":
        return df["macd"] < df["macd_signal"]
    else:   # zero-line
        return df["macd"] < 0


def render_macd(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    hist = ctx.idf["macd_hist"].fillna(0)
    bar_colors = [
        f"rgba(16,185,129,0.7)" if v >= 0 else "rgba(239,68,68,0.7)"
        for v in hist
    ]
    ctx.fig.add_trace(go.Bar(
        x=ctx.idf.index, y=hist, marker_color=bar_colors,
        name="MACD Hist", showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["macd"], mode="lines",
        line=dict(color=ctx.palette.sky, width=1.2),
        name="MACD", showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["macd_signal"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1),
        name="Signal", showlegend=True,
    ), **rn)
    ctx.fig.add_hline(y=0, line_color=ctx.palette.neutral_grid,
        row=ctx.row, col=ctx.col)
    ctx.fig.update_yaxes(title_text="MACD", title_font=dict(size=8),
        row=ctx.row, col=ctx.col)


register(IndicatorSpec(
    key="macd_entry",
    name="MACD",
    group="entry",
    order=9,
    enable_param="macd_enabled",
    params=[
        ParamSpec("macd_enabled", "bool", False, optimize=False,
                  label="MACD enabled", order=0),
        ParamSpec("macd_fast", "int", 12, min=1, max=100,
                  label="Fast", order=1),
        ParamSpec("macd_slow", "int", 26, min=2, max=200,
                  label="Slow", order=2),
        ParamSpec("macd_signal", "int", 9, min=1, max=50,
                  label="Signal", order=3),
        ParamSpec("macd_mode", "categorical", "histogram",
                  choices=("histogram", "crossover", "zero-line"),
                  label="Mode", order=4),
    ],
    compute=compute_macd,
    outputs=["macd", "macd_signal", "macd_hist"],
    long_signal=long_signal_macd,
    short_signal=short_signal_macd,
    plot=PlotSpec(
        kind="panel",
        render=render_macd,
        panel_title="MACD",
        panel_height_weight=1.5,
        owner_for_columns=["macd", "macd_signal", "macd_hist"],
    ),
))
