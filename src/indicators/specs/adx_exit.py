"""ADX exit spec — exit when trend strength fades below threshold."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import adx as _adx


def compute_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "adx" not in df.columns:
        df["di_plus"], df["di_minus"], df["adx"] = _adx(
            df["high"], df["low"], df["close"],
            params["adx_length"], params["adx_smoothing"],
        )
    return df


def long_signal_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["adx"] < params["adx_exit_threshold"]


def short_signal_adx_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["adx"] < params["adx_exit_threshold"]


register(IndicatorSpec(
    key="adx_exit",
    name="ADX Exit",
    group="exit",
    order=5,
    enable_param="adx_exit_enabled",
    params=[
        ParamSpec("adx_exit_enabled", "bool", False, optimize=False,
                  label="ADX exit enabled", order=0),
        ParamSpec("adx_exit_threshold", "int", 20, min=1, max=99,
                  label="Exit threshold", order=1),
    ],
    compute=compute_adx_exit,
    outputs=[],
    long_signal=long_signal_adx_exit,
    short_signal=short_signal_adx_exit,
    reuses_outputs_from=["adx_entry"],
))
