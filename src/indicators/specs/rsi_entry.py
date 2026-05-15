"""RSI mean-reversion entry spec."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import rsi as _rsi


def compute_rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["rsi"] = _rsi(df["close"], params["rsi_length"])
    return df


def long_signal_rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["rsi"] < params["rsi_oversold"]


def short_signal_rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["rsi"] > params["rsi_overbought"]


def render_rsi(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["rsi"], mode="lines",
        line=dict(color=ctx.palette.purple, width=1.2),
        name=f"RSI({ctx.params.get('rsi_length', 14)})", showlegend=True,
    ), **rn)
    ctx.fig.add_hline(y=ctx.params.get("rsi_oversold", 30), line_dash="dash",
        line_color=ctx.palette.os_line, row=ctx.row, col=ctx.col)
    ctx.fig.add_hline(y=ctx.params.get("rsi_overbought", 70), line_dash="dash",
        line_color=ctx.palette.ob_line, row=ctx.row, col=ctx.col)
    ctx.fig.add_hline(y=50, line_color=ctx.palette.neutral_grid, row=ctx.row, col=ctx.col)
    ctx.fig.update_yaxes(title_text="RSI", range=[0, 100],
        title_font=dict(size=8), row=ctx.row, col=ctx.col)


register(IndicatorSpec(
    key="rsi_entry",
    name="RSI",
    group="entry",
    order=5,
    enable_param="rsi_enabled",
    params=[
        ParamSpec("rsi_enabled", "bool", False, optimize=False,
                  label="RSI enabled", order=0),
        ParamSpec("rsi_length", "int", 14, min=2, max=100,
                  label="Length", order=1),
        ParamSpec("rsi_oversold", "int", 30, min=1, max=49,
                  label="Oversold", direction="long", order=2),
        ParamSpec("rsi_overbought", "int", 70, min=51, max=99,
                  label="Overbought", direction="short", order=3),
        ParamSpec("rsi_signal_mode", "categorical", "trigger",
                  choices=("trigger", "filter"),
                  label="Signal mode", order=4, optimize=False),
    ],
    compute=compute_rsi,
    outputs=["rsi"],
    signal_role="trigger",
    signal_mode_param="rsi_signal_mode",
    long_signal=long_signal_rsi,
    short_signal=short_signal_rsi,
    plot=PlotSpec(
        kind="panel",
        render=render_rsi,
        panel_title="RSI",
        panel_y_range=(0, 100),
        owner_for_columns=["rsi"],
    ),
))
