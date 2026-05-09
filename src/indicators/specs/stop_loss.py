"""Stop loss risk spec.

Params only — no compute columns, no signal callables.
The backtest engine reads stop_loss_enabled / stop_loss_pct_* directly.
"""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register


def compute_stop_loss(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    return df


register(IndicatorSpec(
    key="stop_loss",
    name="Stop Loss",
    group="risk",
    order=1,
    enable_param="stop_loss_enabled",
    params=[
        ParamSpec("stop_loss_enabled", "bool", True, optimize=False,
                  label="Stop loss enabled", order=0),
        ParamSpec("stop_loss_pct_long", "float", 3.0, min=0.5, max=30.0, step=0.5,
                  label="Long SL %", direction="long", order=1),
        ParamSpec("stop_loss_pct_short", "float", 3.0, min=0.5, max=30.0, step=0.5,
                  label="Short SL %", direction="short", order=2),
    ],
    compute=compute_stop_loss,
    outputs=[],
    long_signal=None,
    short_signal=None,
))
