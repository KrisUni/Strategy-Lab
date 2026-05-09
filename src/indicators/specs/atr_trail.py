"""ATR-based trailing stop spec.

Computes the 'atr' column used by the backtest engine for ATR trailing stops.
No signal callables — engine handles the exit logic.
"""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import atr as _atr


def compute_atr_trail(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df["atr"] = _atr(df["high"], df["low"], df["close"], params["atr_length"])
    return df


register(IndicatorSpec(
    key="atr_trail",
    name="ATR Trailing",
    group="risk",
    order=4,
    enable_param="atr_trailing_enabled",
    params=[
        ParamSpec("atr_trailing_enabled", "bool", False, optimize=False,
                  label="ATR trailing enabled", order=0),
        ParamSpec("atr_length", "int", 14, min=1, max=50,
                  label="ATR length", order=1),
        ParamSpec("atr_multiplier", "float", 2.0, min=0.5, max=10.0, step=0.5,
                  label="Multiplier", order=2),
    ],
    compute=compute_atr_trail,
    outputs=["atr"],
    long_signal=None,
    short_signal=None,
))
