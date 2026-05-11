"""PAMRP entry indicator spec."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import pamrp


def compute_pamrp_entry(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    vol = df["volume"] if "volume" in df.columns else None
    df["pamrp_entry"] = pamrp(
        df["close"],
        params["pamrp_entry_ma_length"],
        params["pamrp_entry_lookback"],
        params["pamrp_entry_ma_type"],
        vol,
    )
    df["pamrp"] = df["pamrp_entry"]   # legacy column kept for callers
    return df


def long_signal_pamrp_entry(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["pamrp_entry"] < params["pamrp_entry_long"]


def short_signal_pamrp_entry(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["pamrp_entry"] > params["pamrp_entry_short"]


def render_pamrp(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    col = "pamrp_entry" if "pamrp_entry" in ctx.idf.columns else "pamrp"
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf[col], mode="lines",
        line=dict(color=ctx.palette.sky, width=1.2),
        name="PAMRP", showlegend=True,
    ), **rn)
    ctx.fig.add_hline(y=ctx.params.get("pamrp_entry_long", 20), line_dash="dash",
        line_color=ctx.palette.os_line, row=ctx.row, col=ctx.col)
    ctx.fig.add_hline(y=ctx.params.get("pamrp_entry_short", 80), line_dash="dash",
        line_color=ctx.palette.ob_line, row=ctx.row, col=ctx.col)
    ctx.fig.update_yaxes(title_text="PAMRP", range=[0, 100],
        title_font=dict(size=8), row=ctx.row, col=ctx.col)


register(IndicatorSpec(
    key="pamrp_entry",
    name="PAMRP",
    group="entry",
    order=1,
    enable_param="pamrp_enabled",
    params=[
        ParamSpec("pamrp_enabled", "bool", True, optimize=False,
                  label="PAMRP enabled", order=0),
        ParamSpec("pamrp_entry_ma_length", "int", 20, min=1, max=200,
                  label="MA length", order=1),
        ParamSpec("pamrp_entry_lookback", "int", 350, min=50, max=1000,
                  label="Lookback", order=2),
        ParamSpec("pamrp_entry_ma_type", "categorical", "sma",
                  choices=("sma", "ema", "wma", "rma"),
                  label="MA type", order=3),
        ParamSpec("pamrp_entry_long", "int", 20, min=1, max=49,
                  label="Long threshold", direction="long", order=4),
        ParamSpec("pamrp_entry_short", "int", 80, min=51, max=99,
                  label="Short threshold", direction="short", order=5),
    ],
    compute=compute_pamrp_entry,
    outputs=["pamrp_entry", "pamrp"],
    long_signal=long_signal_pamrp_entry,
    short_signal=short_signal_pamrp_entry,
    plot=PlotSpec(
        kind="panel",
        render=render_pamrp,
        panel_title="PAMRP",
        panel_y_range=(0, 100),
        owner_for_columns=["pamrp_entry", "pamrp"],
    ),
))
