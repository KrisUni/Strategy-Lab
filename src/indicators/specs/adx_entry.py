"""ADX trend-strength filter spec."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import adx as _adx


def compute_adx(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["di_plus"], df["di_minus"], df["adx"] = _adx(
        df["high"], df["low"], df["close"],
        params["adx_length"], params["adx_smoothing"],
    )
    return df


def long_signal_adx(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mask = df["adx"] > params["adx_threshold"]
    if params["adx_require_di"]:
        mask = mask & (df["di_plus"] > df["di_minus"])
    return mask


def short_signal_adx(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mask = df["adx"] > params["adx_threshold"]
    if params["adx_require_di"]:
        mask = mask & (df["di_minus"] > df["di_plus"])
    return mask


def render_adx(ctx: PlotContext) -> None:
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
    ctx.fig.add_hline(y=ctx.params.get("adx_threshold", 20), line_dash="dash",
        line_color=ctx.palette.neutral_grid, row=ctx.row, col=ctx.col)
    ctx.fig.update_yaxes(title_text="ADX", title_font=dict(size=8),
        row=ctx.row, col=ctx.col)


register(IndicatorSpec(
    key="adx_entry",
    name="ADX",
    group="entry",
    order=3,
    enable_param="adx_enabled",
    params=[
        ParamSpec("adx_enabled", "bool", False, optimize=False,
                  label="ADX enabled", order=0),
        ParamSpec("adx_length", "int", 14, min=1, max=50,
                  label="Length", order=1),
        ParamSpec("adx_smoothing", "int", 14, min=1, max=50,
                  label="Smoothing", order=2),
        ParamSpec("adx_threshold", "int", 20, min=1, max=99,
                  label="Threshold", order=3),
        ParamSpec("adx_require_di", "bool", False,
                  label="Require DI alignment", order=4),
    ],
    compute=compute_adx,
    outputs=["di_plus", "di_minus", "adx"],
    long_signal=long_signal_adx,
    short_signal=short_signal_adx,
    plot=PlotSpec(
        kind="panel",
        render=render_adx,
        panel_title="ADX",
        owner_for_columns=["adx", "di_plus", "di_minus"],
    ),
))
