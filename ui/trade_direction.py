"""
ui/trade_direction.py
=====================
Shared trade-direction labels and conversion helpers used across the UI.
"""

from src.strategy import TradeDirection


SIDEBAR_TRADE_DIRECTION_OPTIONS = ["Long Only", "Short Only", "Both"]

SIDEBAR_TO_TRADE_DIRECTION = {
    "Long Only": TradeDirection.LONG_ONLY,
    "Short Only": TradeDirection.SHORT_ONLY,
    "Both": TradeDirection.BOTH,
}

TRADE_DIRECTION_TO_SIDEBAR = {
    TradeDirection.LONG_ONLY: "Long Only",
    TradeDirection.SHORT_ONLY: "Short Only",
    TradeDirection.BOTH: "Both",
}

OPT_DIRECTION_OPTIONS = ["long_only", "short_only", "both"]

OPT_DIRECTION_LABELS = {
    "long_only": "Long Only",
    "short_only": "Short Only",
    "both": "Both",
}

SIDEBAR_TO_OPT_DIRECTION = {
    "Long Only": "long_only",
    "Short Only": "short_only",
    "Both": "both",
}


def sidebar_direction_to_trade_direction(value: str, default: TradeDirection = TradeDirection.LONG_ONLY) -> TradeDirection:
    return SIDEBAR_TO_TRADE_DIRECTION.get(value, default)


def sidebar_direction_to_opt(value: str, default: str = "long_only") -> str:
    return SIDEBAR_TO_OPT_DIRECTION.get(value, default)


def opt_direction_to_sidebar(value: str, default: str = "Long Only") -> str:
    return OPT_DIRECTION_LABELS.get(value, default)


def format_opt_direction(value: str) -> str:
    return OPT_DIRECTION_LABELS[value]
