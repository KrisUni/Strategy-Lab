"""Volume exit spec — exit when volume drops below the moving average.

Self-contained — owns its own computation param (ma_length) and decision
params (multiplier). Opportunistically reuses the entry's 'volume_ma' column
when params match. Writes to its own column 'volume_ma_exit'.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import sma


def compute_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "volume" not in df.columns:
        return df
    entry_reusable = (
        params.get("volume_enabled", False)
        and params["volume_exit_ma_length"] == params["volume_ma_length"]
        and "volume_ma" in df.columns
    )
    if entry_reusable:
        df["volume_ma_exit"] = df["volume_ma"]
    else:
        df["volume_ma_exit"] = sma(df["volume"], params["volume_exit_ma_length"])
    return df


def long_signal_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma_exit" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] < df["volume_ma_exit"] * params["volume_exit_multiplier"]


def short_signal_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma_exit" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] < df["volume_ma_exit"] * params["volume_exit_multiplier"]


def _volume_exit_line(ctx: PlotContext) -> None:
    if "volume_ma_exit" not in ctx.idf.columns:
        return
    mult = ctx.params.get("volume_exit_multiplier", 1.0)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["volume_ma_exit"] * mult, mode="lines",
        line=dict(color=ctx.palette.exit_hline, width=0.8, dash="dot"),
        name=f"Exit Vol ×{mult}", showlegend=True,
    ), row=ctx.row, col=ctx.col)


def render_volume_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    if "volume" not in ctx.df.columns:
        return
    bar_colors = [
        "rgba(16,185,129,0.4)" if c >= o else "rgba(239,68,68,0.4)"
        for o, c in zip(ctx.df["open"], ctx.df["close"])
    ]
    ctx.fig.add_trace(go.Bar(
        x=ctx.df.index, y=ctx.df["volume"],
        marker_color=bar_colors,
        name="Volume", showlegend=True,
    ), **rn)
    if "volume_ma_exit" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["volume_ma_exit"], mode="lines",
            line=dict(color=ctx.palette.secondary, width=1.2),
            name=f"Vol MA Exit({ctx.params.get('volume_exit_ma_length', 20)})",
            showlegend=True,
        ), **rn)
    _volume_exit_line(ctx)
    ctx.fig.update_yaxes(title_text="Volume", title_font=dict(size=8),
        row=ctx.row, col=ctx.col)


def contribute_volume_exit(ctx: PlotContext) -> None:
    _volume_exit_line(ctx)


register(IndicatorSpec(
    key="volume_exit",
    name="Volume Exit",
    group="exit",
    order=9,
    enable_param="volume_exit_enabled",
    params=[
        ParamSpec("volume_exit_enabled", "bool", False, optimize=False,
                  label="Volume exit enabled", order=0),
        ParamSpec("volume_exit_ma_length", "int", 20, min=1, max=200,
                  label="MA length", order=1),
        ParamSpec("volume_exit_multiplier", "float", 1.0, min=0.1, max=5.0, step=0.1,
                  label="Exit multiplier", order=2),
    ],
    compute=compute_volume_exit,
    outputs=["volume_ma_exit"],
    long_signal=long_signal_volume_exit,
    short_signal=short_signal_volume_exit,
    reuses_outputs_from=["volume_entry"],
    plot=PlotSpec(
        kind="panel",
        render=render_volume_exit,
        panel_title="Volume",
        contribute=contribute_volume_exit,
    ),
))
