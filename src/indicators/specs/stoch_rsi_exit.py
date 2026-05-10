"""Stochastic RSI exit spec — reuses columns and params from stoch_rsi_entry."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import stoch_rsi as _stoch_rsi


def compute_stoch_rsi_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "stoch_k" not in df.columns:
        df["stoch_k"], df["stoch_d"] = _stoch_rsi(
            df["close"],
            params["stoch_rsi_length"],
            params["stoch_rsi_length"],
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
    ],
    compute=compute_stoch_rsi_exit,
    outputs=[],
    long_signal=long_signal_stoch_rsi_exit,
    short_signal=short_signal_stoch_rsi_exit,
    reuses_outputs_from=["stoch_rsi_entry"],
))
