"""
Strategy-level settings must survive optimization.

Verifies that:
1. Every strategy-level pin reaches best_params unchanged (plain and walk-forward)
2. strategy_level_settings extracts only strategy-level keys
3. STRATEGY_LEVEL_KEYS covers every default except trade_direction
4. _count_active_params ignores pins that name non-optimized params
"""
import numpy as np
import pandas as pd
import pytest

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

from src.strategy import (
    STRATEGY_LEVEL_KEYS,
    _STRATEGY_LEVEL_DEFAULTS,
    strategy_level_settings,
)
from src.indicators.registry import INDICATOR_REGISTRY
from src.optimize import BayesianOptimizer, _count_active_params


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def minimal_df():
    """500-bar synthetic OHLCV DataFrame — enough for a small backtest."""
    np.random.seed(0)
    n = 500
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.01,
        "low":    close * 0.99,
        "close":  close,
        "volume": np.random.randint(1000, 5000, n).astype(float),
    }, index=idx)


@pytest.fixture(scope="module")
def filters():
    """PAMRP entry + PAMRP exit on; every other indicator off."""
    f = {spec.enable_param: False for spec in INDICATOR_REGISTRY}
    f["pamrp_enabled"] = True
    f["pamrp_exit_enabled"] = True
    return f


STRATEGY_PINS = {
    "entry_operator": "or", "exit_operator": "and",
    "allow_same_bar_exit": False, "allow_same_bar_reversal": True,
    "entry_conflict_mode": "prefer_long", "entry_exit_conflict_mode": "defer",
    "position_size_pct": 50.0, "use_kelly": True, "kelly_fraction": 0.25,
}


# ─── Pins survive optimization ────────────────────────────────────────────────

def test_strategy_level_pins_survive_optimization(minimal_df, filters):
    res = BayesianOptimizer(
        minimal_df, filters, metric="profit_factor",
        min_trades=0, pinned_params=STRATEGY_PINS,
    ).optimize(n_trials=5, show_progress=False)
    for k, v in STRATEGY_PINS.items():
        assert res.best_params[k] == v, k


def test_strategy_level_pins_survive_walkforward(minimal_df, filters):
    res = BayesianOptimizer(
        minimal_df, filters, metric="profit_factor",
        min_trades=0, pinned_params=STRATEGY_PINS,
        use_walkforward=True, n_folds=3,
    ).optimize(n_trials=5, show_progress=False)
    for k, v in STRATEGY_PINS.items():
        assert res.best_params[k] == v, k


# ─── Single source of truth ───────────────────────────────────────────────────

def test_strategy_level_settings_extracts_only_known_keys():
    got = strategy_level_settings(
        {"position_size_pct": 50.0, "pamrp_entry_long": 20, "trade_direction": "both"}
    )
    assert got == {"position_size_pct": 50.0}


def test_strategy_level_keys_cover_defaults():
    assert set(STRATEGY_LEVEL_KEYS) | {"trade_direction"} == set(_STRATEGY_LEVEL_DEFAULTS)


# ─── Active-dimension count ───────────────────────────────────────────────────

def test_count_active_params_ignores_non_optimizable_pins(filters):
    base = _count_active_params(filters, pinned_params=None)
    assert base == _count_active_params(
        filters, pinned_params={"entry_operator": "and", "use_kelly": False}
    )

    spec = next(s for s in INDICATOR_REGISTRY if s.key == "pamrp_entry")
    param = next(p for p in spec.params if p.optimize)
    assert _count_active_params(filters, pinned_params={param.name: param.default}) == base - 1
