"""RSI exit spec — exit when RSI crosses back through a neutral level.

Self-contained — owns its own computation param (length) and decision params
(exit levels). Opportunistically reuses the entry's 'rsi' column when params
match. Writes to its own column 'rsi_exit'.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import rsi as _rsi


def compute_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("rsi_enabled", False)
        and params["rsi_exit_length"] == params["rsi_length"]
        and "rsi" in df.columns
    )
    if entry_reusable:
        df["rsi_exit"] = df["rsi"]
    else:
        df["rsi_exit"] = _rsi(df["close"], params["rsi_exit_length"])
    return df


def long_signal_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["rsi_exit"] > params["rsi_exit_long"]


def short_signal_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["rsi_exit"] < params["rsi_exit_short"]


def _rsi_exit_hlines(ctx: PlotContext) -> None:
    ctx.fig.add_hline(y=ctx.params.get("rsi_exit_long", 50), line_dash="dot",
        line_color=ctx.palette.exit_hline, row=ctx.row, col=ctx.col)
    ctx.fig.add_hline(y=ctx.params.get("rsi_exit_short", 50), line_dash="dot",
        line_color="rgba(249,115,22,0.7)", row=ctx.row, col=ctx.col)


def render_rsi_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["rsi_exit"], mode="lines",
        line=dict(color=ctx.palette.purple, width=1.2),
        name=f"RSI Exit({ctx.params.get('rsi_exit_length', 14)})", showlegend=True,
    ), **rn)
    _rsi_exit_hlines(ctx)
    ctx.fig.add_hline(y=50, line_color=ctx.palette.neutral_grid, row=ctx.row, col=ctx.col)
    ctx.fig.update_yaxes(title_text="RSI", range=[0, 100],
        title_font=dict(size=8), row=ctx.row, col=ctx.col)


def contribute_rsi_exit(ctx: PlotContext) -> None:
    _rsi_exit_hlines(ctx)


register(IndicatorSpec(
    key="rsi_exit",
    name="RSI Exit",
    group="exit",
    order=7,
    enable_param="rsi_exit_enabled",
    params=[
        ParamSpec("rsi_exit_enabled", "bool", False, optimize=False,
                  label="RSI exit enabled", order=0),
        ParamSpec("rsi_exit_length", "int", 14, min=2, max=100,
                  label="Length", order=1),
        ParamSpec("rsi_exit_long", "int", 50, min=1, max=99,
                  label="Long exit level", direction="long", order=2),
        ParamSpec("rsi_exit_short", "int", 50, min=1, max=99,
                  label="Short exit level", direction="short", order=3),
    ],
    compute=compute_rsi_exit,
    outputs=["rsi_exit"],
    long_signal=long_signal_rsi_exit,
    short_signal=short_signal_rsi_exit,
    reuses_outputs_from=["rsi_entry"],
    plot=PlotSpec(
        kind="panel",
        render=render_rsi_exit,
        panel_title="RSI",
        panel_y_range=(0, 100),
        contribute=contribute_rsi_exit,
    ),
))
