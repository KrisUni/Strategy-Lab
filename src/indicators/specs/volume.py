"""Volume participation confirmation spec."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import sma


def compute_volume(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "volume" in df.columns:
        df["volume_ma"] = sma(df["volume"], params["volume_ma_length"])
    return df


def long_signal_volume(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] > df["volume_ma"] * params["volume_multiplier"]


def short_signal_volume(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    if "volume_ma" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["volume"] > df["volume_ma"] * params["volume_multiplier"]


register(IndicatorSpec(
    key="volume",
    name="Volume",
    group="entry",
    order=6,
    enable_param="volume_enabled",
    params=[
        ParamSpec("volume_enabled", "bool", False, optimize=False,
                  label="Volume enabled", order=0),
        ParamSpec("volume_ma_length", "int", 20, min=1, max=200,
                  label="MA length", order=1),
        ParamSpec("volume_multiplier", "float", 1.0, min=0.1, max=5.0, step=0.1,
                  label="Multiplier", order=2),
    ],
    compute=compute_volume,
    outputs=["volume_ma"],
    long_signal=long_signal_volume,
    short_signal=short_signal_volume,
))
