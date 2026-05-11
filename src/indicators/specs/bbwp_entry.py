"""BBWP entry indicator spec."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import bbwp, sma


def compute_bbwp_entry(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["bbwp"] = bbwp(df["close"], params["bbwp_length"], params["bbwp_lookback"])
    df["bbwp_sma"] = sma(df["bbwp"], params["bbwp_sma_length"])
    return df


def long_signal_bbwp_entry(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mask = df["bbwp"] < params["bbwp_threshold_long"]
    filt = params["bbwp_ma_filter"]
    if filt == "decreasing":
        mask = mask & (df["bbwp_sma"] < df["bbwp_sma"].shift(1))
    elif filt == "increasing":
        mask = mask & (df["bbwp_sma"] > df["bbwp_sma"].shift(1))
    return mask


def short_signal_bbwp_entry(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mask = df["bbwp"] > params["bbwp_threshold_short"]
    filt = params["bbwp_ma_filter"]
    if filt == "decreasing":
        mask = mask & (df["bbwp_sma"] < df["bbwp_sma"].shift(1))
    elif filt == "increasing":
        mask = mask & (df["bbwp_sma"] > df["bbwp_sma"].shift(1))
    return mask


def render_bbwp(ctx: PlotContext) -> None:
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
    ctx.fig.add_hline(y=ctx.params.get("bbwp_threshold_long", 50), line_dash="dash",
        line_color=ctx.palette.os_line, row=ctx.row, col=ctx.col)
    ctx.fig.add_hline(y=ctx.params.get("bbwp_threshold_short", 50), line_dash="dash",
        line_color=ctx.palette.ob_line, row=ctx.row, col=ctx.col)
    ctx.fig.update_yaxes(title_text="BBWP", range=[0, 100],
        title_font=dict(size=8), row=ctx.row, col=ctx.col)


register(IndicatorSpec(
    key="bbwp_entry",
    name="BBWP",
    group="entry",
    order=2,
    enable_param="bbwp_enabled",
    params=[
        ParamSpec("bbwp_enabled", "bool", True, optimize=False,
                  label="BBWP enabled", order=0),
        ParamSpec("bbwp_length", "int", 13, min=1, max=50,
                  label="BB length", order=1),
        ParamSpec("bbwp_lookback", "int", 252, min=50, max=500,
                  label="Lookback", order=2),
        ParamSpec("bbwp_sma_length", "int", 5, min=1, max=20,
                  label="SMA length", order=3),
        ParamSpec("bbwp_threshold_long", "int", 50, min=1, max=99,
                  label="Long threshold", direction="long", order=4),
        ParamSpec("bbwp_threshold_short", "int", 50, min=1, max=99,
                  label="Short threshold", direction="short", order=5),
        ParamSpec("bbwp_ma_filter", "categorical", "disabled",
                  choices=("disabled", "decreasing", "increasing"),
                  label="MA filter", order=6),
    ],
    compute=compute_bbwp_entry,
    outputs=["bbwp", "bbwp_sma"],
    long_signal=long_signal_bbwp_entry,
    short_signal=short_signal_bbwp_entry,
    plot=PlotSpec(
        kind="panel",
        render=render_bbwp,
        panel_title="BBWP",
        panel_y_range=(0, 100),
        owner_for_columns=["bbwp", "bbwp_sma"],
    ),
))
