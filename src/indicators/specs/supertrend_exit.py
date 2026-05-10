"""Supertrend exit spec — exit when trend direction flips."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import supertrend as _supertrend


def compute_supertrend_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "st_direction" not in df.columns:
        df["supertrend"], df["st_direction"] = _supertrend(
            df["high"], df["low"], df["close"],
            params["supertrend_period"], params["supertrend_multiplier"],
        )
    return df


def long_signal_supertrend_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["st_direction"] < 0


def short_signal_supertrend_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["st_direction"] > 0


register(IndicatorSpec(
    key="supertrend_exit",
    name="Supertrend Exit",
    group="exit",
    order=8,
    enable_param="supertrend_exit_enabled",
    params=[
        ParamSpec("supertrend_exit_enabled", "bool", False, optimize=False,
                  label="Supertrend exit enabled", order=0),
    ],
    compute=compute_supertrend_exit,
    outputs=[],
    long_signal=long_signal_supertrend_exit,
    short_signal=short_signal_supertrend_exit,
    reuses_outputs_from=["supertrend_entry"],
))
