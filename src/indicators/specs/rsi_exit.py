"""RSI exit spec — exit when RSI crosses back through a neutral level."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import rsi as _rsi


def compute_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "rsi" not in df.columns:
        df["rsi"] = _rsi(df["close"], params["rsi_length"])
    return df


def long_signal_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["rsi"] > params["rsi_exit_long"]


def short_signal_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["rsi"] < params["rsi_exit_short"]


register(IndicatorSpec(
    key="rsi_exit",
    name="RSI Exit",
    group="exit",
    order=7,
    enable_param="rsi_exit_enabled",
    params=[
        ParamSpec("rsi_exit_enabled", "bool", False, optimize=False,
                  label="RSI exit enabled", order=0),
        ParamSpec("rsi_exit_long", "int", 50, min=1, max=99,
                  label="Long exit level", direction="long", order=1),
        ParamSpec("rsi_exit_short", "int", 50, min=1, max=99,
                  label="Short exit level", direction="short", order=2),
    ],
    compute=compute_rsi_exit,
    outputs=[],
    long_signal=long_signal_rsi_exit,
    short_signal=short_signal_rsi_exit,
    reuses_outputs_from=["rsi_entry"],
))
