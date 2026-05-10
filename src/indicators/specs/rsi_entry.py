"""RSI mean-reversion entry spec."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import rsi as _rsi


def compute_rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["rsi"] = _rsi(df["close"], params["rsi_length"])
    return df


def long_signal_rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["rsi"] < params["rsi_oversold"]


def short_signal_rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["rsi"] > params["rsi_overbought"]


register(IndicatorSpec(
    key="rsi_entry",
    name="RSI",
    group="entry",
    order=5,
    enable_param="rsi_enabled",
    params=[
        ParamSpec("rsi_enabled", "bool", False, optimize=False,
                  label="RSI enabled", order=0),
        ParamSpec("rsi_length", "int", 14, min=2, max=100,
                  label="Length", order=1),
        ParamSpec("rsi_oversold", "int", 30, min=1, max=49,
                  label="Oversold", direction="long", order=2),
        ParamSpec("rsi_overbought", "int", 70, min=51, max=99,
                  label="Overbought", direction="short", order=3),
    ],
    compute=compute_rsi,
    outputs=["rsi"],
    long_signal=long_signal_rsi,
    short_signal=short_signal_rsi,
))
