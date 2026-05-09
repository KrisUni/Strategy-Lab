"""
Optimizer registry integration tests (Phase 5).

Verifies that:
1. _count_active_params derives its count from the registry correctly
2. _build_params_from_trial produces StrategyParams with all registry keys
3. Seed reproducibility: same seed → same trial param sequence
4. _is_entry_param uses registry membership, not prefix matching
"""
import numpy as np
import pandas as pd
import pytest

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.ERROR)

from src.strategy import StrategyParams, TradeDirection
from src.indicators.registry import INDICATOR_REGISTRY, build_defaults_from_registry
from src.optimize import _count_active_params, _count_enabled_indicators, _is_entry_param
from src.optimize import BayesianOptimizer


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
def pamrp_only_filters():
    return {
        "pamrp_enabled": True,
        "bbwp_enabled": False,
        "adx_enabled": False,
        "ma_trend_enabled": False,
        "rsi_enabled": False,
        "volume_enabled": False,
        "supertrend_enabled": False,
        "vwap_enabled": False,
        "macd_enabled": False,
        "stop_loss_enabled": False,
        "take_profit_enabled": False,
        "trailing_stop_enabled": False,
        "atr_trailing_enabled": False,
        "pamrp_exit_enabled": True,
        "stoch_rsi_exit_enabled": False,
        "time_exit_enabled": False,
        "ma_exit_enabled": False,
        "bbwp_exit_enabled": False,
    }


# ─── _count_active_params ─────────────────────────────────────────────────────

def test_count_active_params_matches_registry_walk(pamrp_only_filters):
    """_count_active_params result must equal a manual registry walk."""
    # Manual count: enabled specs, optimize=True params, direction=both
    expected = sum(
        1
        for spec in INDICATOR_REGISTRY
        if pamrp_only_filters.get(spec.enable_param, False)
        for p in spec.params
        if p.optimize
    )
    result = _count_active_params(pamrp_only_filters, trade_direction=TradeDirection.BOTH)
    assert result == max(expected, 1)


def test_count_active_params_direction_long_only_skips_short_params(pamrp_only_filters):
    both_count = _count_active_params(pamrp_only_filters, trade_direction=TradeDirection.BOTH)
    long_count  = _count_active_params(pamrp_only_filters, trade_direction=TradeDirection.LONG_ONLY)
    # LONG_ONLY skips direction="short" params — count must be ≤ BOTH count
    assert long_count <= both_count


def test_count_active_params_pinned_reduces_count(pamrp_only_filters):
    base = _count_active_params(pamrp_only_filters)
    pinned = {"pamrp_entry_ma_length": 20, "pamrp_entry_lookback": 350}
    reduced = _count_active_params(pamrp_only_filters, pinned_params=pinned)
    assert reduced == max(base - len(pinned), 1)


# ─── _count_enabled_indicators ────────────────────────────────────────────────

def test_count_enabled_indicators_only_counts_entry_group(pamrp_only_filters):
    count = _count_enabled_indicators(pamrp_only_filters)
    # Only pamrp_enabled=True in entry group
    assert count == 1


def test_count_enabled_indicators_all_off():
    filters = {spec.enable_param: False for spec in INDICATOR_REGISTRY}
    assert _count_enabled_indicators(filters) == 0


# ─── _is_entry_param ──────────────────────────────────────────────────────────

def test_is_entry_param_true_for_entry_group_params():
    # Every param owned by an entry-group spec must be classified as entry
    for spec in INDICATOR_REGISTRY:
        if spec.group != "entry":
            continue
        for p in spec.params:
            assert _is_entry_param(p.name), (
                f"{p.name} (from entry spec {spec.key!r}) not classified as entry"
            )


def test_is_entry_param_false_for_exit_risk_params():
    for spec in INDICATOR_REGISTRY:
        if spec.group not in ("exit", "risk"):
            continue
        for p in spec.params:
            assert not _is_entry_param(p.name), (
                f"{p.name} (from {spec.group} spec {spec.key!r}) falsely classified as entry"
            )


# ─── _build_params_from_trial ─────────────────────────────────────────────────

def test_build_params_from_trial_covers_all_registry_keys(minimal_df, pamrp_only_filters):
    """Every registry param must appear in the StrategyParams produced by a trial."""
    opt = BayesianOptimizer(
        df=minimal_df,
        enabled_filters=pamrp_only_filters,
        trade_direction="long_only",
    )
    study = optuna.create_study(sampler=TPESampler(seed=0))
    study.optimize(lambda t: (opt._build_params_from_trial(t), 0.0)[1], n_trials=1)
    trial = study.trials[0]
    params = opt._build_params_from_trial(
        optuna.trial.FixedTrial(trial.params)
    )
    d = params.to_dict()
    registry_keys = set(build_defaults_from_registry().keys())
    missing = registry_keys - set(d.keys())
    assert not missing, f"Missing registry keys in trial params: {missing}"


def test_build_params_from_trial_disabled_uses_defaults(minimal_df, pamrp_only_filters):
    """Params for disabled specs must equal registry defaults."""
    opt = BayesianOptimizer(
        df=minimal_df,
        enabled_filters=pamrp_only_filters,
        trade_direction="long_only",
    )
    study = optuna.create_study(sampler=TPESampler(seed=0))
    study.optimize(lambda t: (opt._build_params_from_trial(t), 0.0)[1], n_trials=1)
    trial = study.trials[0]
    params = opt._build_params_from_trial(optuna.trial.FixedTrial(trial.params))
    defaults = build_defaults_from_registry()

    # RSI is disabled — its params must equal registry defaults
    for spec in INDICATOR_REGISTRY:
        if spec.key != "rsi":
            continue
        for p in spec.params:
            assert params.to_dict()[p.name] == defaults[p.name], (
                f"Disabled rsi param {p.name!r}: expected {defaults[p.name]!r}, "
                f"got {params.to_dict()[p.name]!r}"
            )


# ─── Seed reproducibility ─────────────────────────────────────────────────────

def test_optimizer_seed_reproducibility(minimal_df, pamrp_only_filters):
    """
    Two optimizer runs with the same TPE seed must produce identical trial
    param sequences, confirming registry iteration is deterministically ordered.
    """
    def collect_trial_params(seed):
        opt = BayesianOptimizer(
            df=minimal_df,
            enabled_filters=pamrp_only_filters,
            trade_direction="both",
        )
        study = optuna.create_study(sampler=TPESampler(seed=seed))
        study.optimize(lambda t: (opt._build_params_from_trial(t), 0.0)[1], n_trials=5)
        return [t.params for t in study.trials]

    run1 = collect_trial_params(42)
    run2 = collect_trial_params(42)
    assert run1 == run2, "Same seed produced different trial sequences — non-deterministic registry order"


def test_different_seeds_produce_different_trials(minimal_df, pamrp_only_filters):
    """Sanity check: different seeds should (very likely) yield different trials."""
    def first_trial_params(seed):
        opt = BayesianOptimizer(
            df=minimal_df,
            enabled_filters=pamrp_only_filters,
            trade_direction="long_only",
        )
        study = optuna.create_study(sampler=TPESampler(seed=seed))
        study.optimize(lambda t: (opt._build_params_from_trial(t), 0.0)[1], n_trials=1)
        return study.trials[0].params

    assert first_trial_params(0) != first_trial_params(99)
