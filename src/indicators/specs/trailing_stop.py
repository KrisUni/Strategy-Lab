"""Trailing stop risk spec. Params only — engine handles execution."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register


def compute_trailing_stop(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    return df


register(IndicatorSpec(
    key="trailing_stop",
    name="Trailing Stop",
    group="risk",
    order=3,
    enable_param="trailing_stop_enabled",
    params=[
        ParamSpec("trailing_stop_enabled", "bool", False, optimize=False,
                  label="Trailing stop enabled", order=0),
        ParamSpec("trailing_stop_pct", "float", 2.0, min=0.5, max=20.0, step=0.5,
                  label="Trail %", order=1),
        ParamSpec("trailing_stop_activation", "float", 1.0, min=0.0, max=10.0, step=0.5,
                  label="Activation %", order=2),
    ],
    compute=compute_trailing_stop,
    outputs=[],
    long_signal=None,
    short_signal=None,
))
