"""Volume participation confirmation spec."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import sma


def compute_volume(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "volume" in df.columns:
        df["volume_ma"] = sma(df["volume"], params["volume_ma_length"])
    return df


def long_signal_volume(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] > df["volume_ma"] * params["volume_multiplier"]


def short_signal_volume(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] > df["volume_ma"] * params["volume_multiplier"]


def render_volume(ctx: PlotContext) -> None:
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
    if "volume_ma" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["volume_ma"], mode="lines",
            line=dict(color=ctx.palette.secondary, width=1.2),
            name=f"Vol MA({ctx.params.get('volume_ma_length', 20)})",
            showlegend=True,
        ), **rn)
    mult = ctx.params.get("volume_multiplier", 1.0)
    if mult != 1.0 and "volume_ma" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["volume_ma"] * mult, mode="lines",
            line=dict(color=ctx.palette.os_line, width=0.8, dash="dot"),
            name=f"Vol ×{mult}", showlegend=True,
        ), **rn)
    ctx.fig.update_yaxes(title_text="Volume", title_font=dict(size=8),
        row=ctx.row, col=ctx.col)


register(IndicatorSpec(
    key="volume_entry",
    name="Volume",
    group="entry",
    order=6,
    enable_param="volume_enabled",
    params=[
        ParamSpec("volume_enabled", "bool", False, optimize=False,
                  label="Volume enabled", order=0),
        ParamSpec("volume_ma_length", "int", 20, min=1, max=200,
                  label="MA length", order=1),
        ParamSpec("volume_multiplier", "float", 1.0, min=0.1, max=5.0, step=0.1,
                  label="Multiplier", order=2),
    ],
    compute=compute_volume,
    outputs=["volume_ma"],
    long_signal=long_signal_volume,
    short_signal=short_signal_volume,
    plot=PlotSpec(
        kind="panel",
        render=render_volume,
        panel_title="Volume",
        owner_for_columns=["volume_ma"],
    ),
))
