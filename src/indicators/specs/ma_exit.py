"""MA crossover exit spec — pure mirror of ma_trend direction.

Uses ma_trend's fast/slow lengths and ma_type. No exit-specific decision
params: for an MA crossover, the lengths *are* the decision. If you want a
tactical fast-crossover exit on top of a separate slow trend filter, that
is a different indicator and should be filed as a new spec.

Falls under Pattern C (pure mirror) in the entry/exit consistency model.
"""
from typing import Any, Dict
import pandas as pd
from plotly import graph_objects as go
from ..registry import IndicatorSpec, ParamSpec, PlotSpec, PlotContext, register
from .. import ma


def compute_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "ma_fast" not in df.columns:
        df["ma_fast"] = ma(df["close"], params["ma_fast_length"], params["ma_type"])
    if "ma_slow" not in df.columns:
        df["ma_slow"] = ma(df["close"], params["ma_slow_length"], params["ma_type"])
    return df


def long_signal_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    # Exit long when fast crosses below slow (death cross)
    return df["ma_fast"] < df["ma_slow"]


def short_signal_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    # Exit short when fast crosses above slow (golden cross)
    return df["ma_fast"] > df["ma_slow"]


def render_ma_exit(ctx: PlotContext) -> None:
    # When ma_trend is also enabled, contribute() handles the visual on ma_trend's
    # row. When only ma_exit is enabled, we render the MAs ourselves.
    rn = dict(row=ctx.row, col=ctx.col) if ctx.is_subplot else {}
    ma_type = ctx.params.get("ma_type", "sma").upper()
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["ma_fast"], mode="lines",
        line=dict(color=ctx.palette.primary, width=1.2),
        name=f"Exit {ma_type}({ctx.params.get('ma_fast_length', 50)})",
        showlegend=True,
    ), **rn)
    ctx.fig.add_trace(go.Scatter(
        x=ctx.idf.index, y=ctx.idf["ma_slow"], mode="lines",
        line=dict(color=ctx.palette.secondary, width=1.2),
        name=f"Exit {ma_type}({ctx.params.get('ma_slow_length', 200)})",
        showlegend=True,
    ), **rn)


def contribute_ma_exit(ctx: PlotContext) -> None:
    # ma_trend already plots the same MAs — nothing to add visually
    pass


register(IndicatorSpec(
    key="ma_exit",
    name="MA Exit",
    group="exit",
    order=3,
    enable_param="ma_exit_enabled",
    params=[
        ParamSpec("ma_exit_enabled", "bool", False, optimize=False,
                  label="MA exit enabled", order=0),
    ],
    compute=compute_ma_exit,
    outputs=[],
    long_signal=long_signal_ma_exit,
    short_signal=short_signal_ma_exit,
    reuses_outputs_from=["ma_trend"],
    plot=PlotSpec(
        kind="overlay",
        render=render_ma_exit,
        contribute=contribute_ma_exit,
    ),
))
