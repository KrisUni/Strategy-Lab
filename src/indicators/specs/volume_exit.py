"""Volume exit spec — exit when volume drops below the moving average."""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import sma


def compute_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "volume_ma" not in df.columns and "volume" in df.columns:
        df["volume_ma"] = sma(df["volume"], params["volume_ma_length"])
    return df


def long_signal_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] < df["volume_ma"] * params["volume_exit_multiplier"]


def short_signal_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] < df["volume_ma"] * params["volume_exit_multiplier"]


def _volume_exit_line(ctx: PlotContext) -> None:
    if "volume_ma" not in ctx.idf.columns:
        return
    mult = ctx.params.get("volume_exit_multiplier", 1.0)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["volume_ma"] * mult, mode="lines",
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
    if "volume_ma" in ctx.idf.columns:
        ctx.fig.add_trace(go.Scatter(
            x=ctx.idf.index, y=ctx.idf["volume_ma"], mode="lines",
            line=dict(color=ctx.palette.secondary, width=1.2),
            name=f"Vol MA({ctx.params.get('volume_ma_length', 20)})",
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
        ParamSpec("volume_exit_multiplier", "float", 1.0, min=0.1, max=5.0, step=0.1,
                  label="Exit multiplier", order=1),
    ],
    compute=compute_volume_exit,
    outputs=[],
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
