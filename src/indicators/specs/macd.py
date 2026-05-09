"""MACD momentum entry spec."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import macd as _macd


def compute_macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(
        df["close"], params["macd_fast"], params["macd_slow"], params["macd_signal"],
    )
    return df


def long_signal_macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_mode"]
    if mode == "histogram":
        return df["macd_hist"] > 0
    elif mode == "crossover":
        return df["macd"] > df["macd_signal"]
    else:   # zero-line
        return df["macd"] > 0


def short_signal_macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_mode"]
    if mode == "histogram":
        return df["macd_hist"] < 0
    elif mode == "crossover":
        return df["macd"] < df["macd_signal"]
    else:   # zero-line
        return df["macd"] < 0


register(IndicatorSpec(
    key="macd",
    name="MACD",
    group="entry",
    order=9,
    enable_param="macd_enabled",
    params=[
        ParamSpec("macd_enabled", "bool", False, optimize=False,
                  label="MACD enabled", order=0),
        ParamSpec("macd_fast", "int", 12, min=1, max=100,
                  label="Fast", order=1),
        ParamSpec("macd_slow", "int", 26, min=2, max=200,
                  label="Slow", order=2),
        ParamSpec("macd_signal", "int", 9, min=1, max=50,
                  label="Signal", order=3),
        ParamSpec("macd_mode", "categorical", "histogram",
                  choices=("histogram", "crossover", "zero-line"),
                  label="Mode", order=4),
    ],
    compute=compute_macd,
    outputs=["macd", "macd_signal", "macd_hist"],
    long_signal=long_signal_macd,
    short_signal=short_signal_macd,
))
