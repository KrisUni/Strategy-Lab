"""PAMRP exit indicator spec.

PAMRP exit and entry share the same underlying computation but have
independent parameters (MA length, lookback, MA type, thresholds).
When params match and pamrp_entry was already computed, the exit reuses
the pamrp_entry column to avoid redundant computation.
"""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import pamrp


def compute_pamrp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    entry_reusable = (
        params.get("pamrp_enabled", False)
        and params["pamrp_exit_ma_length"] == params["pamrp_entry_ma_length"]
        and params["pamrp_exit_lookback"] == params["pamrp_entry_lookback"]
        and params["pamrp_exit_ma_type"] == params["pamrp_entry_ma_type"]
        and "pamrp_entry" in df.columns
    )
    if entry_reusable:
        df["pamrp_exit"] = df["pamrp_entry"]
    else:
        vol = df["volume"] if "volume" in df.columns else None
        df["pamrp_exit"] = pamrp(
            df["close"],
            params["pamrp_exit_ma_length"],
            params["pamrp_exit_lookback"],
            params["pamrp_exit_ma_type"],
            vol,
        )
        # Populate legacy pamrp column if entry didn't run
        if "pamrp" not in df.columns:
            df["pamrp"] = df["pamrp_exit"]
    return df


def long_signal_pamrp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["pamrp_exit"] > params["pamrp_exit_long"]


def short_signal_pamrp_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["pamrp_exit"] < params["pamrp_exit_short"]


register(IndicatorSpec(
    key="pamrp_exit",
    name="PAMRP Exit",
    group="exit",
    order=1,
    enable_param="pamrp_exit_enabled",
    params=[
        ParamSpec("pamrp_exit_enabled", "bool", True, optimize=False,
                  label="PAMRP exit enabled", order=0),
        ParamSpec("pamrp_exit_ma_length", "int", 20, min=1, max=200,
                  label="MA length", order=1),
        ParamSpec("pamrp_exit_lookback", "int", 350, min=50, max=1000,
                  label="Lookback", order=2),
        ParamSpec("pamrp_exit_ma_type", "categorical", "sma",
                  choices=("sma", "ema", "wma", "rma"),
                  label="MA type", order=3),
        ParamSpec("pamrp_exit_long", "int", 70, min=51, max=99,
                  label="Long exit threshold", direction="long", order=4),
        ParamSpec("pamrp_exit_short", "int", 30, min=1, max=49,
                  label="Short exit threshold", direction="short", order=5),
    ],
    compute=compute_pamrp_exit,
    outputs=["pamrp_exit"],
    long_signal=long_signal_pamrp_exit,
    short_signal=short_signal_pamrp_exit,
    reuses_outputs_from=["pamrp_entry"],
))
