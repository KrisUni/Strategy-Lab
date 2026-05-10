"""BBWP exit indicator spec.

Reuses the 'bbwp' column computed by bbwp_entry. If bbwp_entry is disabled
but bbwp_exit is enabled, this compute function computes bbwp itself.
"""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import bbwp, sma


def compute_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "bbwp" not in df.columns:
        df["bbwp"] = bbwp(df["close"], params["bbwp_length"], params["bbwp_lookback"])
        df["bbwp_sma"] = sma(df["bbwp"], params["bbwp_sma_length"])
    return df


def long_signal_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["bbwp"] > params["bbwp_exit_threshold_long"]


def short_signal_bbwp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["bbwp"] < params["bbwp_exit_threshold_short"]


register(IndicatorSpec(
    key="bbwp_exit",
    name="BBWP Exit",
    group="exit",
    order=4,
    enable_param="bbwp_exit_enabled",
    params=[
        ParamSpec("bbwp_exit_enabled", "bool", False, optimize=False,
                  label="BBWP exit enabled", order=0),
        ParamSpec("bbwp_exit_threshold_long", "int", 80, min=1, max=99,
                  label="Long exit threshold", direction="long", order=1),
        ParamSpec("bbwp_exit_threshold_short", "int", 20, min=1, max=99,
                  label="Short exit threshold", direction="short", order=2),
    ],
    compute=compute_bbwp_exit,
    outputs=[],   # bbwp already written by bbwp_entry when both enabled
    long_signal=long_signal_bbwp_exit,
    short_signal=short_signal_bbwp_exit,
    reuses_outputs_from=["bbwp_entry"],
))
