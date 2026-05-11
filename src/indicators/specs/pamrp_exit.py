"""PAMRP exit indicator spec.

PAMRP exit and entry share the same underlying computation but have
independent parameters (MA length, lookback, MA type, thresholds).
When params match and pamrp_entry was already computed, the exit reuses
the pamrp_entry column to avoid redundant computation.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import pamrp


def compute_pamrp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("pamrp_enabled", False)
        and params["pamrp_exit_ma_length"] == params["pamrp_entry_ma_length"]
        and params["pamrp_exit_lookback"] == params["pamrp_entry_lookback"]
        and params["pamrp_exit_ma_type"] == params["pamrp_entry_ma_type"]
        and "pamrp_entry" in df.columns
    )
    if entry_reusable:
        df["pamrp_exit"] = df["pamrp_entry"]
    else:
        vol = df["volume"] if "volume" in df.columns else None
        df["pamrp_exit"] = pamrp(
            df["close"],
            params["pamrp_exit_ma_length"],
            params["pamrp_exit_lookback"],
            params["pamrp_exit_ma_type"],
            vol,
        )
        # Populate legacy pamrp column if entry didn't run
        if "pamrp" not in df.columns:
            df["pamrp"] = df["pamrp_exit"]
    return df


def long_signal_pamrp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["pamrp_exit"] > params["pamrp_exit_long"]


def short_signal_pamrp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["pamrp_exit"] < params["pamrp_exit_short"]


def _pamrp_exit_hlines(ctx: PlotContext) -> None:
    ctx.fig.add_hline(y=ctx.params.get("pamrp_exit_long", 70), line_dash="dot",
        line_color=ctx.palette.exit_hline, row=ctx.row, col=ctx.col)
    ctx.fig.add_hline(y=ctx.params.get("pamrp_exit_short", 30), line_dash="dot",
        line_color="rgba(249,115,22,0.7)", row=ctx.row, col=ctx.col)


def render_pamrp_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    col = "pamrp_exit" if "pamrp_exit" in ctx.idf.columns else "pamrp"
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf[col], mode="lines",
        line=dict(color=ctx.palette.sky, width=1.2),
        name="PAMRP Exit", showlegend=True,
    ), **rn)
    _pamrp_exit_hlines(ctx)
    ctx.fig.update_yaxes(title_text="PAMRP", range=[0, 100],
        title_font=dict(size=8), row=ctx.row, col=ctx.col)


def contribute_pamrp_exit(ctx: PlotContext) -> None:
    same_config = (
        ctx.params.get("pamrp_exit_ma_length", 20) == ctx.params.get("pamrp_entry_ma_length", 20)
        and ctx.params.get("pamrp_exit_lookback", 350) == ctx.params.get("pamrp_entry_lookback", 350)
        and ctx.params.get("pamrp_exit_ma_type", "sma") == ctx.params.get("pamrp_entry_ma_type", "sma")
    )
    if not same_config and "pamrp_exit" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["pamrp_exit"], mode="lines",
            line=dict(color=ctx.palette.secondary, width=1.2, dash="dot"),
            name="PAMRP Exit", showlegend=True,
        ), row=ctx.row, col=ctx.col)
    _pamrp_exit_hlines(ctx)


register(IndicatorSpec(
    key="pamrp_exit",
    name="PAMRP Exit",
    group="exit",
    order=1,
    enable_param="pamrp_exit_enabled",
    params=[
        ParamSpec("pamrp_exit_enabled", "bool", True, optimize=False,
                  label="PAMRP exit enabled", order=0),
        ParamSpec("pamrp_exit_ma_length", "int", 20, min=1, max=200,
                  label="MA length", order=1),
        ParamSpec("pamrp_exit_lookback", "int", 350, min=50, max=1000,
                  label="Lookback", order=2),
        ParamSpec("pamrp_exit_ma_type", "categorical", "sma",
                  choices=("sma", "ema", "wma", "rma"),
                  label="MA type", order=3),
        ParamSpec("pamrp_exit_long", "int", 70, min=51, max=99,
                  label="Long exit threshold", direction="long", order=4),
        ParamSpec("pamrp_exit_short", "int", 30, min=1, max=49,
                  label="Short exit threshold", direction="short", order=5),
    ],
    compute=compute_pamrp_exit,
    outputs=["pamrp_exit"],
    long_signal=long_signal_pamrp_exit,
    short_signal=short_signal_pamrp_exit,
    reuses_outputs_from=["pamrp_entry"],
    plot=PlotSpec(
        kind="panel",
        render=render_pamrp_exit,
        panel_title="PAMRP",
        panel_y_range=(0, 100),
        contribute=contribute_pamrp_exit,
    ),
))
