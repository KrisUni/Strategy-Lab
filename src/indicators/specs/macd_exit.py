"""MACD exit spec — exit when MACD gives signal opposite to entry direction."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import macd as _macd


def compute_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "macd" not in df.columns:
        df["macd"], df["macd_signal"], df["macd_hist"] = _macd(
            df["close"], params["macd_fast"], params["macd_slow"], params["macd_signal"],
        )
    return df


def long_signal_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_exit_mode"]
    if mode == "histogram":
        return df["macd_hist"] < 0
    elif mode == "crossover":
        return df["macd"] < df["macd_signal"]
    else:   # zero-line
        return df["macd"] < 0


def short_signal_macd_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    mode = params["macd_exit_mode"]
    if mode == "histogram":
        return df["macd_hist"] > 0
    elif mode == "crossover":
        return df["macd"] > df["macd_signal"]
    else:   # zero-line
        return df["macd"] > 0


register(IndicatorSpec(
    key="macd_exit",
    name="MACD Exit",
    group="exit",
    order=6,
    enable_param="macd_exit_enabled",
    params=[
        ParamSpec("macd_exit_enabled", "bool", False, optimize=False,
                  label="MACD exit enabled", order=0),
        ParamSpec("macd_exit_mode", "categorical", "histogram",
                  choices=("histogram", "crossover", "zero-line"),
                  label="Mode", order=1),
    ],
    compute=compute_macd_exit,
    outputs=[],
    long_signal=long_signal_macd_exit,
    short_signal=short_signal_macd_exit,
    reuses_outputs_from=["macd_entry"],
))
