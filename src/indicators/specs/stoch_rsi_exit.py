"""Stochastic RSI exit spec — reuses computation from stoch_rsi_entry,
has its own exit-specific overbought/oversold thresholds.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import stoch_rsi as _stoch_rsi


def compute_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "stoch_k" not in df.columns:
        df["stoch_k"], df["stoch_d"] = _stoch_rsi(
            df["close"],
            params["stoch_rsi_length"],
            params["stoch_rsi_length"],
            params["stoch_rsi_k"],
            params["stoch_rsi_d"],
        )
    return df


def long_signal_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    # Long exit: stoch_k climbs back above the EXIT overbought level
    return df["stoch_k"] > params["stoch_rsi_exit_overbought"]


def short_signal_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    # Short exit: stoch_k drops back below the EXIT oversold level
    return df["stoch_k"] < params["stoch_rsi_exit_oversold"]


def _stoch_exit_hlines(ctx: PlotContext) -> None:
    ctx.fig.add_hline(
        y=ctx.params.get("stoch_rsi_exit_overbought", 80),
        line_dash="dot", line_color=ctx.palette.exit_hline,
        row=ctx.row, col=ctx.col,
    )
    ctx.fig.add_hline(
        y=ctx.params.get("stoch_rsi_exit_oversold", 20),
        line_dash="dot", line_color="rgba(249,115,22,0.7)",
        row=ctx.row, col=ctx.col,
    )


def render_stoch_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["stoch_k"], mode="lines",
        line=dict(color=ctx.palette.sky, width=1.2),
        name="Stoch %K", showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["stoch_d"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1),
        name="Stoch %D", showlegend=True,
    ), **rn)
    _stoch_exit_hlines(ctx)
    ctx.fig.update_yaxes(title_text="Stoch RSI", range=[0, 100],
        title_font=dict(size=8), row=ctx.row, col=ctx.col)


def contribute_stoch_exit(ctx: PlotContext) -> None:
    _stoch_exit_hlines(ctx)


register(IndicatorSpec(
    key="stoch_rsi_exit",
    name="Stoch RSI Exit",
    group="exit",
    order=2,
    enable_param="stoch_rsi_exit_enabled",
    params=[
        ParamSpec("stoch_rsi_exit_enabled", "bool", False, optimize=False,
                  label="Stoch RSI exit enabled", order=0),
        ParamSpec("stoch_rsi_exit_overbought", "int", 80, min=51, max=99,
                  label="Long exit overbought", direction="long", order=1),
        ParamSpec("stoch_rsi_exit_oversold", "int", 20, min=1, max=49,
                  label="Short exit oversold", direction="short", order=2),
    ],
    compute=compute_stoch_rsi_exit,
    outputs=[],
    long_signal=long_signal_stoch_rsi_exit,
    short_signal=short_signal_stoch_rsi_exit,
    reuses_outputs_from=["stoch_rsi_entry"],
    plot=PlotSpec(
        kind="panel",
        render=render_stoch_exit,
        panel_title="Stoch RSI",
        panel_y_range=(0, 100),
        contribute=contribute_stoch_exit,
    ),
))
