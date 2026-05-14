"""Stochastic RSI exit spec — self-contained with independent compute params.

Owns its own computation params (length, k, d) and decision params
(overbought, oversold). Opportunistically reuses the entry's stoch_k/stoch_d
columns when params match. Writes to its own columns 'stoch_k_exit'/'stoch_d_exit'.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import stoch_rsi as _stoch_rsi


def compute_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("stoch_rsi_entry_enabled", False)
        and params["stoch_rsi_exit_length"] == params["stoch_rsi_length"]
        and params["stoch_rsi_exit_k"]      == params["stoch_rsi_k"]
        and params["stoch_rsi_exit_d"]      == params["stoch_rsi_d"]
        and "stoch_k" in df.columns
    )
    if entry_reusable:
        df["stoch_k_exit"] = df["stoch_k"]
        df["stoch_d_exit"] = df["stoch_d"]
    else:
        df["stoch_k_exit"], df["stoch_d_exit"] = _stoch_rsi(
            df["close"],
            params["stoch_rsi_exit_length"],
            params["stoch_rsi_exit_length"],
            params["stoch_rsi_exit_k"],
            params["stoch_rsi_exit_d"],
        )
    return df


def long_signal_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["stoch_k_exit"] > params["stoch_rsi_exit_overbought"]


def short_signal_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["stoch_k_exit"] < params["stoch_rsi_exit_oversold"]


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
        x=ctx.idf.index, y=ctx.idf["stoch_k_exit"], mode="lines",
        line=dict(color=ctx.palette.sky, width=1.2),
        name="Stoch Exit %K", showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["stoch_d_exit"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1),
        name="Stoch Exit %D", showlegend=True,
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
        ParamSpec("stoch_rsi_exit_length", "int", 14, min=2, max=50,
                  label="Length", order=1),
        ParamSpec("stoch_rsi_exit_k", "int", 3, min=1, max=20,
                  label="%K smooth", order=2),
        ParamSpec("stoch_rsi_exit_d", "int", 3, min=1, max=20,
                  label="%D smooth", order=3),
        ParamSpec("stoch_rsi_exit_overbought", "int", 80, min=51, max=99,
                  label="Long exit overbought", direction="long", order=4),
        ParamSpec("stoch_rsi_exit_oversold", "int", 20, min=1, max=49,
                  label="Short exit oversold", direction="short", order=5),
    ],
    compute=compute_stoch_rsi_exit,
    outputs=["stoch_k_exit", "stoch_d_exit"],
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
