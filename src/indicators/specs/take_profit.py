"""Take profit risk spec. Params only — engine handles execution."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register


def compute_take_profit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    return df


register(IndicatorSpec(
    key="take_profit",
    name="Take Profit",
    group="risk",
    order=2,
    enable_param="take_profit_enabled",
    params=[
        ParamSpec("take_profit_enabled", "bool", False, optimize=False,
                  label="Take profit enabled", order=0),
        ParamSpec("take_profit_pct_long", "float", 5.0, min=0.5, max=50.0, step=0.5,
                  label="Long TP %", direction="long", order=1),
        ParamSpec("take_profit_pct_short", "float", 5.0, min=0.5, max=50.0, step=0.5,
                  label="Short TP %", direction="short", order=2),
    ],
    compute=compute_take_profit,
    outputs=[],
    long_signal=None,
    short_signal=None,
))
