"""Supertrend exit spec — exit when trend direction flips.

Self-contained — owns its own computation params (period, multiplier).
Opportunistically reuses the entry's 'st_direction' and 'supertrend' columns
when params match. Writes to its own columns 'supertrend_exit_line' and
'supertrend_exit_dir'.
"""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import supertrend as _supertrend


def compute_supertrend_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("supertrend_enabled", False)
        and params["supertrend_exit_period"]     == params["supertrend_period"]
        and params["supertrend_exit_multiplier"] == params["supertrend_multiplier"]
        and "st_direction" in df.columns
    )
    if entry_reusable:
        df["supertrend_exit_line"] = df["supertrend"]
        df["supertrend_exit_dir"]  = df["st_direction"]
    else:
        df["supertrend_exit_line"], df["supertrend_exit_dir"] = _supertrend(
            df["high"], df["low"], df["close"],
            params["supertrend_exit_period"], params["supertrend_exit_multiplier"],
        )
    return df


def long_signal_supertrend_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["supertrend_exit_dir"] < 0


def short_signal_supertrend_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["supertrend_exit_dir"] > 0


register(IndicatorSpec(
    key="supertrend_exit",
    name="Supertrend Exit",
    group="exit",
    order=8,
    enable_param="supertrend_exit_enabled",
    params=[
        ParamSpec("supertrend_exit_enabled", "bool", False, optimize=False,
                  label="Supertrend exit enabled", order=0),
        ParamSpec("supertrend_exit_period", "int", 10, min=2, max=50,
                  label="Period", order=1),
        ParamSpec("supertrend_exit_multiplier", "float", 3.0, min=0.5, max=10.0, step=0.5,
                  label="Multiplier", order=2),
    ],
    compute=compute_supertrend_exit,
    outputs=["supertrend_exit_line", "supertrend_exit_dir"],
    long_signal=long_signal_supertrend_exit,
    short_signal=short_signal_supertrend_exit,
    reuses_outputs_from=["supertrend_entry"],
))
