"""ADX exit spec — exit when trend strength fades below threshold."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import adx as _adx


def compute_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "adx" not in df.columns:
        df["di_plus"], df["di_minus"], df["adx"] = _adx(
            df["high"], df["low"], df["close"],
            params["adx_length"], params["adx_smoothing"],
        )
    return df


def long_signal_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["adx"] < params["adx_exit_threshold"]


def short_signal_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["adx"] < params["adx_exit_threshold"]


def _adx_exit_hline(ctx: PlotContext) -> None:
    ctx.fig.add_hline(y=ctx.params.get("adx_exit_threshold", 20), line_dash="dot",
        line_color=ctx.palette.exit_hline, row=ctx.row, col=ctx.col)


def render_adx_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["adx"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1.2),
        name="ADX", showlegend=True,
    ), **rn)
    if "di_plus" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["di_plus"], mode="lines",
            line=dict(color=ctx.palette.bullish, width=0.8),
            name="+DI", showlegend=True,
        ), **rn)
    if "di_minus" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["di_minus"], mode="lines",
            line=dict(color=ctx.palette.bearish, width=0.8),
            name="-DI", showlegend=True,
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
        ParamSpec("adx_exit_threshold", "int", 20, min=1, max=99,
                  label="Exit threshold", order=1),
    ],
    compute=compute_adx_exit,
    outputs=[],
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
