"""BBWP exit indicator spec.

Self-contained — owns its own computation params (length, lookback, sma_length)
and decision params (exit thresholds). Opportunistically reuses the entry's
'bbwp' column when entry is enabled AND its params match, to avoid redundant
computation. Writes to its own column 'bbwp_exit'.

bbwp_exit_sma_length is declared to match the entry's param for opportunistic
reuse, but the exit does not compute a separate bbwp_sma_exit column (filter
logic is out of scope for the exit side).
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import bbwp


def compute_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("bbwp_enabled", False)
        and params["bbwp_exit_length"]     == params["bbwp_length"]
        and params["bbwp_exit_lookback"]   == params["bbwp_lookback"]
        and params["bbwp_exit_sma_length"] == params["bbwp_sma_length"]
        and "bbwp" in df.columns
    )
    if entry_reusable:
        df["bbwp_exit"] = df["bbwp"]
    else:
        df["bbwp_exit"] = bbwp(
            df["close"], params["bbwp_exit_length"], params["bbwp_exit_lookback"]
        )
    return df


def long_signal_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["bbwp_exit"] > params["bbwp_exit_threshold_long"]


def short_signal_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["bbwp_exit"] < params["bbwp_exit_threshold_short"]


def _bbwp_exit_hlines(ctx: PlotContext) -> None:
    ctx.fig.add_hline(y=ctx.params.get("bbwp_exit_threshold_long", 80), line_dash="dot",
        line_color=ctx.palette.exit_hline, row=ctx.row, col=ctx.col)
    ctx.fig.add_hline(y=ctx.params.get("bbwp_exit_threshold_short", 20), line_dash="dot",
        line_color="rgba(249,115,22,0.7)", row=ctx.row, col=ctx.col)


def render_bbwp_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["bbwp_exit"], mode="lines",
        line=dict(color=ctx.palette.sky, width=1.2),
        name="BBWP Exit", showlegend=True,
    ), **rn)
    _bbwp_exit_hlines(ctx)
    ctx.fig.update_yaxes(title_text="BBWP", range=[0, 100],
        title_font=dict(size=8), row=ctx.row, col=ctx.col)


def contribute_bbwp_exit(ctx: PlotContext) -> None:
    same_config = (
        ctx.params.get("bbwp_exit_length",      13)  == ctx.params.get("bbwp_length",     13)
        and ctx.params.get("bbwp_exit_lookback", 252) == ctx.params.get("bbwp_lookback",  252)
        and ctx.params.get("bbwp_exit_sma_length", 5) == ctx.params.get("bbwp_sma_length",  5)
    )
    if not same_config and "bbwp_exit" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["bbwp_exit"], mode="lines",
            line=dict(color=ctx.palette.secondary, width=1.2, dash="dot"),
            name="BBWP Exit", showlegend=True,
        ), row=ctx.row, col=ctx.col)
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
        ParamSpec("bbwp_exit_length", "int", 13, min=1, max=50,
                  label="Length", order=1),
        ParamSpec("bbwp_exit_lookback", "int", 252, min=50, max=500,
                  label="Lookback", order=2),
        ParamSpec("bbwp_exit_sma_length", "int", 5, min=1, max=20,
                  label="SMA length", order=3),
        ParamSpec("bbwp_exit_threshold_long", "int", 80, min=1, max=99,
                  label="Long exit threshold", direction="long", order=4),
        ParamSpec("bbwp_exit_threshold_short", "int", 20, min=1, max=99,
                  label="Short exit threshold", direction="short", order=5),
    ],
    compute=compute_bbwp_exit,
    outputs=["bbwp_exit"],
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
