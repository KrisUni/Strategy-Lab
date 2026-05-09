"""Time exit spec. Params only — backtest engine executes the time-based exit."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register


def compute_time_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    return df


register(IndicatorSpec(
    key="time_exit",
    name="Time Exit",
    group="risk",
    order=5,
    enable_param="time_exit_enabled",
    params=[
        ParamSpec("time_exit_enabled", "bool", False, optimize=False,
                  label="Time exit enabled", order=0),
        ParamSpec("time_exit_bars_long", "int", 20, min=1, max=200,
                  label="Long bars", direction="long", order=1),
        ParamSpec("time_exit_bars_short", "int", 20, min=1, max=200,
                  label="Short bars", direction="short", order=2),
    ],
    compute=compute_time_exit,
    outputs=[],
    long_signal=None,
    short_signal=None,
))
