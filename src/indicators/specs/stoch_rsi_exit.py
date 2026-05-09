"""Stochastic RSI exit spec."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import stoch_rsi as _stoch_rsi


def compute_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["stoch_k"], df["stoch_d"] = _stoch_rsi(
        df["close"],
        params["stoch_rsi_length"],   # rsi_length
        params["stoch_rsi_length"],   # stoch_length — same param (FIX-SR)
        params["stoch_rsi_k"],
        params["stoch_rsi_d"],
    )
    return df


def long_signal_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["stoch_k"] > params["stoch_rsi_overbought"]


def short_signal_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["stoch_k"] < params["stoch_rsi_oversold"]


register(IndicatorSpec(
    key="stoch_rsi_exit",
    name="Stoch RSI Exit",
    group="exit",
    order=2,
    enable_param="stoch_rsi_exit_enabled",
    params=[
        ParamSpec("stoch_rsi_exit_enabled", "bool", False, optimize=False,
                  label="Stoch RSI exit enabled", order=0),
        ParamSpec("stoch_rsi_length", "int", 14, min=2, max=50,
                  label="Length", order=1),
        ParamSpec("stoch_rsi_k", "int", 3, min=1, max=20,
                  label="%K smooth", order=2),
        ParamSpec("stoch_rsi_d", "int", 3, min=1, max=20,
                  label="%D smooth", order=3),
        ParamSpec("stoch_rsi_overbought", "int", 80, min=51, max=99,
                  label="Overbought", direction="long", order=4),
        ParamSpec("stoch_rsi_oversold", "int", 20, min=1, max=49,
                  label="Oversold", direction="short", order=5),
    ],
    compute=compute_stoch_rsi_exit,
    outputs=["stoch_k", "stoch_d"],
    long_signal=long_signal_stoch_rsi_exit,
    short_signal=short_signal_stoch_rsi_exit,
))
