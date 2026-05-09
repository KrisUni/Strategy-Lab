"""MA crossover exit spec."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import ema


def compute_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["exit_ma_fast"] = ema(df["close"], params["ma_exit_fast"])
    df["exit_ma_slow"] = ema(df["close"], params["ma_exit_slow"])
    return df


def long_signal_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["exit_ma_fast"] < df["exit_ma_slow"]


def short_signal_ma_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["exit_ma_fast"] > df["exit_ma_slow"]


register(IndicatorSpec(
    key="ma_exit",
    name="MA Exit",
    group="exit",
    order=3,
    enable_param="ma_exit_enabled",
    params=[
        ParamSpec("ma_exit_enabled", "bool", False, optimize=False,
                  label="MA exit enabled", order=0),
        ParamSpec("ma_exit_fast", "int", 10, min=1, max=100,
                  label="Fast EMA", order=1),
        ParamSpec("ma_exit_slow", "int", 20, min=2, max=200,
                  label="Slow EMA", order=2),
    ],
    compute=compute_ma_exit,
    outputs=["exit_ma_fast", "exit_ma_slow"],
    long_signal=long_signal_ma_exit,
    short_signal=short_signal_ma_exit,
))
