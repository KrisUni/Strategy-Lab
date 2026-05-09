"""VWAP spec.

VWAP is meaningful only for intraday data (price anchors to the session's
value area). On daily OHLCV data it degenerates to a cumulative price-volume
average with no session reset, producing a lagged trend-following signal
rather than a true intraday mean-reversion anchor.

For this reason long_signal and short_signal are set to None in the registry.
The current strategy engine (src/strategy/__init__.py) does generate VWAP
entry signals when vwap_enabled=True and the 'vwap' column is present, but
this will not be wired into registry-driven signal generation in Phase 4.

TODO: Revisit when intraday session-aware data loading is supported.
"""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register
from .. import vwap as _vwap


def compute_vwap(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if "volume" in df.columns:
        df["vwap"] = _vwap(df["high"], df["low"], df["close"], df["volume"])
    return df


register(IndicatorSpec(
    key="vwap",
    name="VWAP",
    group="entry",
    order=8,
    enable_param="vwap_enabled",
    params=[
        ParamSpec("vwap_enabled", "bool", False, optimize=False,
                  label="VWAP enabled", order=0),
    ],
    compute=compute_vwap,
    outputs=["vwap"],
    long_signal=None,
    short_signal=None,
))
