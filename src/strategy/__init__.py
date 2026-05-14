"""
Strategy Module
===============
Generates entry and exit signals from StrategyParams + indicator layer.

Changes from previous version
──────────────────────────────
[FIX-ST]  supertrend direction convention corrected in generate_entry_signals.
          Old code checked `st_direction < 0` for long (legacy inverted convention).
          Now correctly checks `st_direction > 0` (1 = bullish, industry standard).
          The indicators layer handles the actual flip; this file just uses the
          correct sign.

[FIX-SR]  stoch_rsi call updated to pass `stoch_length` as the new second argument.
          Previously the call was stoch_rsi(close, rsi_length, k, d).
          Now:                       stoch_rsi(close, rsi_length, stoch_length, k, d).
          stoch_length defaults to stoch_rsi_length (same parameter) since no
          separate UI param exists yet — behaviour is unchanged for existing configs.
"""

import pandas as pd
from typing import Dict, Any, Iterable
from enum import Enum

from ..indicators.registry import (
    enabled_specs, topological_sort, build_defaults_from_registry,
)
from ..indicators import specs as _indicator_specs  # noqa: F401 — triggers registration


class TradeDirection(Enum):
    LONG_ONLY  = "long_only"
    SHORT_ONLY = "short_only"
    BOTH       = "both"


class ConditionOperator(str, Enum):
    AND = "and"
    OR = "or"


class EntryConflictMode(str, Enum):
    SKIP = "skip"
    PREFER_LONG = "prefer_long"
    PREFER_SHORT = "prefer_short"


_STRATEGY_LEVEL_DEFAULTS: Dict[str, Any] = {
    "trade_direction":        TradeDirection.LONG_ONLY,
    "entry_operator":         ConditionOperator.AND,
    "exit_operator":          ConditionOperator.OR,
    "allow_same_bar_exit":    True,
    "allow_same_bar_reversal": False,
    "entry_conflict_mode":    EntryConflictMode.SKIP,
    "position_size_pct":      100.0,
    "use_kelly":              False,
    "kelly_fraction":         0.5,
}

_DIRECTION_MAP: Dict[str, TradeDirection] = {
    "Long Only":  TradeDirection.LONG_ONLY,
    "Short Only": TradeDirection.SHORT_ONLY,
    "Both":       TradeDirection.BOTH,
    "long_only":  TradeDirection.LONG_ONLY,
    "short_only": TradeDirection.SHORT_ONLY,
    "both":       TradeDirection.BOTH,
}

# Provisional exit params. These exist in user state and ui/session.py defaults
# but are not yet declared in ParamSpecs (Issue A). Issue B moves them into the
# registry properly; remove this set when that lands.
_PROVISIONAL_EXIT_PARAMS: set = {
    "rsi_exit_length",
    "bbwp_exit_length", "bbwp_exit_lookback", "bbwp_exit_sma_length",
    "adx_exit_length", "adx_exit_smoothing",
    "macd_exit_fast", "macd_exit_slow", "macd_exit_signal",
    "volume_exit_ma_length",
    "supertrend_exit_period", "supertrend_exit_multiplier",
    "stoch_rsi_exit_length", "stoch_rsi_exit_k", "stoch_rsi_exit_d",
    "ma_exit_fast", "ma_exit_slow", "ma_exit_type",
}


class StrategyParams:
    """Dict-backed param container. Attribute-access shim keeps call sites unchanged."""

    def __init__(self, **overrides: Any) -> None:
        d: Dict[str, Any] = {**_STRATEGY_LEVEL_DEFAULTS, **build_defaults_from_registry()}
        # Seed provisional exit defaults from the UI param map (Issue A).
        # Function-local import avoids a top-level ui → src import cycle.
        from ui.session import get_default_params  # noqa: PLC0415
        _ui_defaults = get_default_params()
        for k in _PROVISIONAL_EXIT_PARAMS:
            if k in _ui_defaults:
                d.setdefault(k, _ui_defaults[k])
        d.update(overrides)
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "_d")[name]
        except KeyError:
            raise AttributeError(f"StrategyParams has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        object.__getattribute__(self, "_d")[name] = value

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in object.__getattribute__(self, "_d").items():
            result[k] = v.value if isinstance(v, Enum) else v
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        d = dict(d)
        # PAMRP migration layer 1: pamrp_length → entry/exit split
        legacy_pamrp_length = d.pop("pamrp_length", None)
        if legacy_pamrp_length is not None:
            d.setdefault("pamrp_entry_length", legacy_pamrp_length)
            d.setdefault("pamrp_exit_length", legacy_pamrp_length)
        # PAMRP migration layer 2: pamrp_entry/exit_length → ma_length
        legacy_entry_length = d.pop("pamrp_entry_length", None)
        if legacy_entry_length is not None:
            d.setdefault("pamrp_entry_ma_length", legacy_entry_length)
        legacy_exit_length = d.pop("pamrp_exit_length", None)
        if legacy_exit_length is not None:
            d.setdefault("pamrp_exit_ma_length", legacy_exit_length)
        # stoch_rsi_exit migration: seed new exit thresholds from entry thresholds
        stoch_ob = d.get("stoch_rsi_overbought")
        stoch_os = d.get("stoch_rsi_oversold")
        if stoch_ob is not None:
            d.setdefault("stoch_rsi_exit_overbought", stoch_ob)
        if stoch_os is not None:
            d.setdefault("stoch_rsi_exit_oversold", stoch_os)
        # UI migration: time_exit_bars (single) → long/short split
        legacy_bars = d.pop("time_exit_bars", None)
        if legacy_bars is not None:
            d.setdefault("time_exit_bars_long", legacy_bars)
            d.setdefault("time_exit_bars_short", legacy_bars)
        # Enum coercions (handles both UI display strings and storage values)
        td = d.get("trade_direction")
        if isinstance(td, str):
            d["trade_direction"] = _DIRECTION_MAP.get(td, TradeDirection.LONG_ONLY)
        eo = d.get("entry_operator")
        if isinstance(eo, str):
            d["entry_operator"] = ConditionOperator(eo.lower())
        xo = d.get("exit_operator")
        if isinstance(xo, str):
            d["exit_operator"] = ConditionOperator(xo.lower())
        ecm = d.get("entry_conflict_mode")
        if isinstance(ecm, str):
            d["entry_conflict_mode"] = EntryConflictMode(ecm.lower())
        # Issue A: seed independent-exit compute params from entry params.
        # Function-local import avoids a top-level ui → src import cycle.
        from ui.state_migration import _ENTRY_TO_EXIT_PARAM_MAPPINGS  # noqa: PLC0415
        for entry_key, exit_key in _ENTRY_TO_EXIT_PARAM_MAPPINGS:
            if exit_key not in d and entry_key in d:
                d[exit_key] = d[entry_key]
        # Filter to known keys only
        known = set(_STRATEGY_LEVEL_DEFAULTS) | set(build_defaults_from_registry()) | _PROVISIONAL_EXIT_PARAMS
        return cls(**{k: v for k, v in d.items() if k in known})


class SignalGenerator:
    """Generate entry and exit signals."""

    def __init__(self, params: StrategyParams):
        self.params = params

    @staticmethod
    def _combine_condition_masks(
        masks: Iterable[pd.Series],
        operator: ConditionOperator,
        index: pd.Index,
    ) -> pd.Series:
        masks = [mask.fillna(False) for mask in masks]
        if not masks:
            return pd.Series(False, index=index)

        if operator == ConditionOperator.AND:
            result = pd.Series(True, index=index)
            for mask in masks:
                result = result & mask
            return result

        result = pd.Series(False, index=index)
        for mask in masks:
            result = result | mask
        return result

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        params = self.params.to_dict()

        for spec in topological_sort(enabled_specs(params)):
            df = spec.compute(df, params)

        # Legacy fallback columns: signal generation guards access with
        # if p.X_enabled, so these don't affect trades, but external
        # callers may expect them.
        if 'pamrp_entry' not in df.columns:
            df['pamrp_entry'] = 50.0
        if 'pamrp_exit' not in df.columns:
            df['pamrp_exit'] = 50.0
        if 'pamrp' not in df.columns:
            df['pamrp'] = 50.0
        if 'bbwp' not in df.columns:
            df['bbwp'] = 50.0
        if 'bbwp_sma' not in df.columns:
            df['bbwp_sma'] = 50.0

        return df

    def generate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params
        params = p.to_dict()
        td = p.trade_direction
        long_or_both  = td in (TradeDirection.LONG_ONLY,  TradeDirection.BOTH)
        short_or_both = td in (TradeDirection.SHORT_ONLY, TradeDirection.BOTH)

        long_masks:  list[pd.Series] = []
        short_masks: list[pd.Series] = []

        for spec in enabled_specs(params):
            if spec.group != "entry":
                continue
            if long_or_both and spec.long_signal is not None:
                long_masks.append(spec.long_signal(df, params))
            if short_or_both and spec.short_signal is not None:
                short_masks.append(spec.short_signal(df, params))

        long_signal  = self._combine_condition_masks(long_masks,  p.entry_operator, df.index)
        short_signal = self._combine_condition_masks(short_masks, p.entry_operator, df.index)

        if td == TradeDirection.LONG_ONLY:
            short_signal = pd.Series(False, index=df.index)
        elif td == TradeDirection.SHORT_ONLY:
            long_signal  = pd.Series(False, index=df.index)

        df['entry_long']  = long_signal.fillna(False)
        df['entry_short'] = short_signal.fillna(False)

        return df

    def generate_exit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params
        params = p.to_dict()

        exit_long_masks:  list[pd.Series] = []
        exit_short_masks: list[pd.Series] = []

        for spec in enabled_specs(params):
            if spec.group != "exit":
                continue
            if spec.long_signal is not None:
                exit_long_masks.append(spec.long_signal(df, params))
            if spec.short_signal is not None:
                exit_short_masks.append(spec.short_signal(df, params))

        exit_long  = self._combine_condition_masks(exit_long_masks,  p.exit_operator, df.index)
        exit_short = self._combine_condition_masks(exit_short_masks, p.exit_operator, df.index)

        df['exit_long_signal']  = exit_long.fillna(False)
        df['exit_short_signal'] = exit_short.fillna(False)

        return df

    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df = self.generate_entry_signals(df)
        df = self.generate_exit_signals(df)
        return df
