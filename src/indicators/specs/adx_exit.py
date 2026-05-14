"""ADX exit spec — exit when trend strength fades below threshold.

Self-contained — owns its own computation params (length, smoothing) and
decision params (threshold). Opportunistically reuses the entry's 'adx'
column when params match. Writes to its own column 'adx_exit'.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import adx as _adx


def compute_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("adx_enabled", False)
        and params["adx_exit_length"]    == params["adx_length"]
        and params["adx_exit_smoothing"] == params["adx_smoothing"]
        and "adx" in df.columns
    )
    if entry_reusable:
        df["adx_exit"] = df["adx"]
    else:
        _, _, df["adx_exit"] = _adx(
            df["high"], df["low"], df["close"],
            params["adx_exit_length"], params["adx_exit_smoothing"],
        )
    return df


def long_signal_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["adx_exit"] < params["adx_exit_threshold"]


def short_signal_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["adx_exit"] < params["adx_exit_threshold"]


def _adx_exit_hline(ctx: PlotContext) -> None:
    ctx.fig.add_hline(y=ctx.params.get("adx_exit_threshold", 20), line_dash="dot",
        line_color=ctx.palette.exit_hline, row=ctx.row, col=ctx.col)


def render_adx_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["adx_exit"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1.2),
        name="ADX Exit", showlegend=True,
    ), **rn)
    _adx_exit_hline(ctx)
    ctx.fig.update_yaxes(title_text="ADX", title_font=dict(size=8),
        row=ctx.row, col=ctx.col)


def contribute_adx_exit(ctx: PlotContext) -> None:
    _adx_exit_hline(ctx)


register(IndicatorSpec(
    key="adx_exit",
    name="ADX Exit",
    group="exit",
    order=5,
    enable_param="adx_exit_enabled",
    params=[
        ParamSpec("adx_exit_enabled", "bool", False, optimize=False,
                  label="ADX exit enabled", order=0),
        ParamSpec("adx_exit_length", "int", 14, min=1, max=50,
                  label="Length", order=1),
        ParamSpec("adx_exit_smoothing", "int", 14, min=1, max=50,
                  label="Smoothing", order=2),
        ParamSpec("adx_exit_threshold", "int", 20, min=1, max=99,
                  label="Exit threshold", order=3),
    ],
    compute=compute_adx_exit,
    outputs=["adx_exit"],
    long_signal=long_signal_adx_exit,
    short_signal=short_signal_adx_exit,
    reuses_outputs_from=["adx_entry"],
    plot=PlotSpec(
        kind="panel",
        render=render_adx_exit,
        panel_title="ADX",
        contribute=contribute_adx_exit,
    ),
))
