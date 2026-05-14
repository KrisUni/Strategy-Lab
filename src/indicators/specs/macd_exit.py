"""MACD exit spec — exit when MACD gives signal opposite to entry direction.

Self-contained — owns its own computation params (fast, slow, signal) and
decision params (mode). Opportunistically reuses the entry's MACD columns
when params match. Writes to its own columns 'macd_exit_line',
'macd_exit_signal_line', 'macd_exit_hist'.

Note: param 'macd_exit_signal' is the signal-line smoothing length;
column 'macd_exit_signal_line' is the output column — intentionally distinct names.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import macd as _macd


def compute_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("macd_enabled", False)
        and params["macd_exit_fast"]   == params["macd_fast"]
        and params["macd_exit_slow"]   == params["macd_slow"]
        and params["macd_exit_signal"] == params["macd_signal"]
        and "macd" in df.columns
    )
    if entry_reusable:
        df["macd_exit_line"]        = df["macd"]
        df["macd_exit_signal_line"] = df["macd_signal"]
        df["macd_exit_hist"]        = df["macd_hist"]
    else:
        df["macd_exit_line"], df["macd_exit_signal_line"], df["macd_exit_hist"] = _macd(
            df["close"],
            params["macd_exit_fast"],
            params["macd_exit_slow"],
            params["macd_exit_signal"],
        )
    return df


def long_signal_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_exit_mode"]
    if mode == "histogram":
        return df["macd_exit_hist"] < 0
    elif mode == "crossover":
        return df["macd_exit_line"] < df["macd_exit_signal_line"]
    else:   # zero-line
        return df["macd_exit_line"] < 0


def short_signal_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_exit_mode"]
    if mode == "histogram":
        return df["macd_exit_hist"] > 0
    elif mode == "crossover":
        return df["macd_exit_line"] > df["macd_exit_signal_line"]
    else:   # zero-line
        return df["macd_exit_line"] > 0


def render_macd_exit(ctx: PlotContext) -> None:
    rn = dict(row=ctx.row, col=ctx.col)
    hist = ctx.idf["macd_exit_hist"].fillna(0)
    bar_colors = [
        "rgba(16,185,129,0.7)" if v >= 0 else "rgba(239,68,68,0.7)"
        for v in hist
    ]
    ctx.fig.add_trace(go.Bar(
        x=ctx.idf.index, y=hist, marker_color=bar_colors,
        name="MACD Exit Hist", showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["macd_exit_line"], mode="lines",
        line=dict(color=ctx.palette.sky, width=1.2),
        name="MACD Exit", showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["macd_exit_signal_line"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1),
        name="MACD Exit Signal", showlegend=True,
    ), **rn)
    ctx.fig.add_hline(y=0, line_color=ctx.palette.neutral_grid,
        row=ctx.row, col=ctx.col)
    ctx.fig.update_yaxes(title_text="MACD", title_font=dict(size=8),
        row=ctx.row, col=ctx.col)


def contribute_macd_exit(ctx: PlotContext) -> None:
    pass  # exit mode is behavioral only — no additional visual element


register(IndicatorSpec(
    key="macd_exit",
    name="MACD Exit",
    group="exit",
    order=6,
    enable_param="macd_exit_enabled",
    params=[
        ParamSpec("macd_exit_enabled", "bool", False, optimize=False,
                  label="MACD exit enabled", order=0),
        ParamSpec("macd_exit_fast", "int", 12, min=1, max=100,
                  label="Fast length", order=1),
        ParamSpec("macd_exit_slow", "int", 26, min=2, max=200,
                  label="Slow length", order=2),
        ParamSpec("macd_exit_signal", "int", 9, min=1, max=50,
                  label="Signal length", order=3),
        ParamSpec("macd_exit_mode", "categorical", "histogram",
                  choices=("histogram", "crossover", "zero-line"),
                  label="Mode", order=4),
    ],
    compute=compute_macd_exit,
    outputs=["macd_exit_line", "macd_exit_signal_line", "macd_exit_hist"],
    long_signal=long_signal_macd_exit,
    short_signal=short_signal_macd_exit,
    reuses_outputs_from=["macd_entry"],
    plot=PlotSpec(
        kind="panel",
        render=render_macd_exit,
        panel_title="MACD",
        panel_height_weight=1.5,
        contribute=contribute_macd_exit,
    ),
))
