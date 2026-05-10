"""Volume exit spec — exit when volume drops below the moving average."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import sma


def compute_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "volume_ma" not in df.columns and "volume" in df.columns:
        df["volume_ma"] = sma(df["volume"], params["volume_ma_length"])
    return df


def long_signal_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] < df["volume_ma"] * params["volume_exit_multiplier"]


def short_signal_volume_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] < df["volume_ma"] * params["volume_exit_multiplier"]


register(IndicatorSpec(
    key="volume_exit",
    name="Volume Exit",
    group="exit",
    order=9,
    enable_param="volume_exit_enabled",
    params=[
        ParamSpec("volume_exit_enabled", "bool", False, optimize=False,
                  label="Volume exit enabled", order=0),
        ParamSpec("volume_exit_multiplier", "float", 1.0, min=0.1, max=5.0, step=0.1,
                  label="Exit multiplier", order=1),
    ],
    compute=compute_volume_exit,
    outputs=[],
    long_signal=long_signal_volume_exit,
    short_signal=short_signal_volume_exit,
    reuses_outputs_from=["volume_entry"],
))
