"""
src/strategy/atradeaday.py
==========================
A Trade A Day strategy stub.

Logic (to be implemented in backend phase):
  - Mark high/low of the 9:30 AM 5-min candle
  - On 1-min chart: detect FVG breakout of those levels
  - Wait for pullback retest of the FVG
  - Enter on engulfing candle closing over the retest candle
  - SL: first candle of the FVG
  - TP: fixed 3:1 risk/reward
  - Max 1 trade per day
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import pandas as pd


@dataclass
class ATradeADayParams:
    rr_ratio: float = 3.0
    risk_per_trade: float = 100.0
    entry_time: str = "09:30"


def run_atradeaday(df: pd.DataFrame, params: ATradeADayParams) -> Dict[str, Any]:
    """
    Stub — returns an empty result dict compatible with the results
    display in ui/tabs/atradeaday.py.
    Replace the body of this function in the backend phase.
    """
    return {
        "trades": [],
        "num_trades": 0,
        "total_return_pct": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "equity_curve": [],
    }