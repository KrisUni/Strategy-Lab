"""MACD exit spec — exit when MACD gives signal opposite to entry direction."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import macd as _macd


def compute_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "macd" not in df.columns:
        df["macd"], df["macd_signal"], df["macd_hist"] = _macd(
            df["close"], params["macd_fast"], params["macd_slow"], params["macd_signal"],
        )
    return df


def long_signal_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_exit_mode"]
    if mode == "histogram":
        return df["macd_hist"] < 0
    elif mode == "crossover":
        return df["macd"] < df["macd_signal"]
    else:   # zero-line
        return df["macd"] < 0


def short_signal_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_exit_mode"]
    if mode == "histogram":
        return df["macd_hist"] > 0
    elif mode == "crossover":
        return df["macd"] > df["macd_signal"]
    else:   # zero-line
        return df["macd"] > 0


def render_macd_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    hist = ctx.idf["macd_hist"].fillna(0)
    bar_colors = [
        "rgba(16,185,129,0.7)" if v >= 0 else "rgba(239,68,68,0.7)"
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


def contribute_macd_exit(ctx: PlotContext) -> None:
    pass  # exit mode is behavioral only — no additional visual element


register(IndicatorSpec(
    key="macd_exit",
    name="MACD Exit",
    group="exit",
    order=6,
    enable_param="macd_exit_enabled",
    params=[
        ParamSpec("macd_exit_enabled", "bool", False, optimize=False,
                  label="MACD exit enabled", order=0),
        ParamSpec("macd_exit_mode", "categorical", "histogram",
                  choices=("histogram", "crossover", "zero-line"),
                  label="Mode", order=1),
    ],
    compute=compute_macd_exit,
    outputs=[],
    long_signal=long_signal_macd_exit,
    short_signal=short_signal_macd_exit,
    reuses_outputs_from=["macd_entry"],
    plot=PlotSpec(
        kind="panel",
        render=render_macd_exit,
        panel_title="MACD",
        panel_height_weight=1.5,
        contribute=contribute_macd_exit,
    ),
))
