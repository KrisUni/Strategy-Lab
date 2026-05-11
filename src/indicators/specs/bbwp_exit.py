"""BBWP exit indicator spec.

Reuses the 'bbwp' column computed by bbwp_entry. If bbwp_entry is disabled
but bbwp_exit is enabled, this compute function computes bbwp itself.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import bbwp, sma


def compute_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "bbwp" not in df.columns:
        df["bbwp"] = bbwp(df["close"], params["bbwp_length"], params["bbwp_lookback"])
        df["bbwp_sma"] = sma(df["bbwp"], params["bbwp_sma_length"])
    return df


def long_signal_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["bbwp"] > params["bbwp_exit_threshold_long"]


def short_signal_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["bbwp"] < params["bbwp_exit_threshold_short"]


def _bbwp_exit_hlines(ctx: PlotContext) -> None:
    ctx.fig.add_hline(y=ctx.params.get("bbwp_exit_threshold_long", 80), line_dash="dot",
        line_color=ctx.palette.exit_hline, row=ctx.row, col=ctx.col)
    ctx.fig.add_hline(y=ctx.params.get("bbwp_exit_threshold_short", 20), line_dash="dot",
        line_color="rgba(249,115,22,0.7)", row=ctx.row, col=ctx.col)


def render_bbwp_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["bbwp"], mode="lines",
        line=dict(color=ctx.palette.sky, width=1.2),
        name="BBWP", showlegend=True,
    ), **rn)
    if "bbwp_sma" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["bbwp_sma"], mode="lines",
            line=dict(color=ctx.palette.secondary, width=1),
            name="BBWP SMA", showlegend=True,
        ), **rn)
    _bbwp_exit_hlines(ctx)
    ctx.fig.update_yaxes(title_text="BBWP", range=[0, 100],
        title_font=dict(size=8), row=ctx.row, col=ctx.col)


def contribute_bbwp_exit(ctx: PlotContext) -> None:
    _bbwp_exit_hlines(ctx)


register(IndicatorSpec(
    key="bbwp_exit",
    name="BBWP Exit",
    group="exit",
    order=4,
    enable_param="bbwp_exit_enabled",
    params=[
        ParamSpec("bbwp_exit_enabled", "bool", False, optimize=False,
                  label="BBWP exit enabled", order=0),
        ParamSpec("bbwp_exit_threshold_long", "int", 80, min=1, max=99,
                  label="Long exit threshold", direction="long", order=1),
        ParamSpec("bbwp_exit_threshold_short", "int", 20, min=1, max=99,
                  label="Short exit threshold", direction="short", order=2),
    ],
    compute=compute_bbwp_exit,
    outputs=[],   # bbwp already written by bbwp_entry when both enabled
    long_signal=long_signal_bbwp_exit,
    short_signal=short_signal_bbwp_exit,
    reuses_outputs_from=["bbwp_entry"],
    plot=PlotSpec(
        kind="panel",
        render=render_bbwp_exit,
        panel_title="BBWP",
        panel_y_range=(0, 100),
        contribute=contribute_bbwp_exit,
    ),
))
