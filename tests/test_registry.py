"""
Registry schema and population tests (Phase 1 & 2).

These tests verify that:
1. Every legacy StrategyParams field appears in the registry (or is a known strategy-level param)
2. Registry defaults match dataclass defaults exactly
3. No duplicate param names or indicator keys
4. validate_registry() passes
5. All compute/signal callables are named functions (not lambdas)
6. calculate_indicators produces the expected column set (Phase 2)
"""
import numpy as np
import pandas as pd
import pytest
from src.strategy import StrategyParams, SignalGenerator
from src.indicators.registry import (
    INDICATOR_REGISTRY,
    STRATEGY_LEVEL_PARAMS,
    all_specs,
    build_defaults_from_registry,
    enabled_specs,
    topological_sort,
    validate_registry,
)
# Importing specs triggers registration and validate_registry()
import src.indicators.specs  # noqa: F401


# ─── Schema tests ─────────────────────────────────────────────────────────────

def test_validate_registry_passes():
    """validate_registry() should not raise."""
    validate_registry()


def test_no_duplicate_indicator_keys():
    keys = [s.key for s in INDICATOR_REGISTRY]
    assert len(keys) == len(set(keys)), f"Duplicate indicator keys: {keys}"


def test_no_duplicate_param_names():
    names = []
    for spec in INDICATOR_REGISTRY:
        for p in spec.params:
            names.append(p.name)
    assert len(names) == len(set(names)), (
        f"Duplicate param names: {[n for n in names if names.count(n) > 1]}"
    )


def test_compute_callables_are_named_not_lambda():
    for spec in all_specs():
        for attr, fn in [
            ("compute", spec.compute),
            ("long_signal", spec.long_signal),
            ("short_signal", spec.short_signal),
        ]:
            if fn is not None:
                assert not fn.__name__.startswith("<lambda>"), (
                    f"{spec.key}.{attr} is a lambda — use a named function"
                )


def test_all_enable_params_are_bool():
    for spec in all_specs():
        ep = next((p for p in spec.params if p.name == spec.enable_param), None)
        assert ep is not None, f"{spec.key}: enable_param '{spec.enable_param}' not in params"
        assert ep.type == "bool", f"{spec.key}: enable_param must be type='bool'"


def test_numeric_params_have_min_max():
    for spec in all_specs():
        for p in spec.params:
            if p.type in ("int", "float"):
                assert p.min is not None and p.max is not None, (
                    f"{spec.key}.{p.name}: numeric param missing min or max"
                )


def test_categorical_params_have_choices():
    for spec in all_specs():
        for p in spec.params:
            if p.type == "categorical":
                assert p.choices, (
                    f"{spec.key}.{p.name}: categorical param missing choices"
                )


# ─── Coverage tests ───────────────────────────────────────────────────────────

def test_all_legacy_params_present_in_registry():
    """Every StrategyParams field is either in the registry or a known strategy-level param."""
    legacy_fields = set(StrategyParams.__dataclass_fields__.keys())
    indicator_fields = legacy_fields - STRATEGY_LEVEL_PARAMS
    registry_params = set(build_defaults_from_registry().keys())
    missing = indicator_fields - registry_params
    assert not missing, (
        f"StrategyParams fields not found in registry: {sorted(missing)}\n"
        "Add a ParamSpec for each, or add to STRATEGY_LEVEL_PARAMS if they're not indicators."
    )


def test_registry_defaults_match_dataclass_defaults():
    """Registry default for each param must equal the StrategyParams dataclass default."""
    defaults_dc = {
        k: v.default
        for k, v in StrategyParams.__dataclass_fields__.items()
        if k not in STRATEGY_LEVEL_PARAMS
    }
    defaults_reg = build_defaults_from_registry()

    mismatches = {}
    for name, dc_default in defaults_dc.items():
        if name in defaults_reg:
            if defaults_reg[name] != dc_default:
                mismatches[name] = (dc_default, defaults_reg[name])

    assert not mismatches, (
        "Registry defaults differ from StrategyParams defaults:\n"
        + "\n".join(f"  {k}: dataclass={v[0]!r}, registry={v[1]!r}" for k, v in mismatches.items())
    )


# ─── Functional tests ─────────────────────────────────────────────────────────

def test_build_defaults_from_registry_returns_all_indicator_params():
    defaults = build_defaults_from_registry()
    # Spot-check a few known params
    assert "pamrp_enabled" in defaults
    assert defaults["pamrp_enabled"] is True
    assert "rsi_length" in defaults
    assert defaults["rsi_length"] == 14
    assert "stop_loss_pct_long" in defaults
    assert defaults["stop_loss_pct_long"] == 3.0


def test_enabled_specs_filters_by_enable_param():
    defaults = build_defaults_from_registry()
    specs = enabled_specs(defaults)
    enabled_keys = {s.key for s in specs}

    # These are enabled by default
    assert "pamrp_entry" in enabled_keys
    assert "pamrp_exit" in enabled_keys
    assert "bbwp_entry" in enabled_keys
    assert "stop_loss" in enabled_keys

    # These are disabled by default
    assert "rsi" not in enabled_keys
    assert "macd" not in enabled_keys
    assert "supertrend" not in enabled_keys


def test_topological_sort_puts_deps_first():
    specs = all_specs()
    sorted_specs = topological_sort(specs)
    key_order = {s.key: i for i, s in enumerate(sorted_specs)}

    for spec in sorted_specs:
        for dep_key in spec.reuses_outputs_from:
            if dep_key in key_order:
                assert key_order[dep_key] < key_order[spec.key], (
                    f"Dependency '{dep_key}' must come before '{spec.key}' in sorted order"
                )


def test_bbwp_exit_comes_after_bbwp_entry_in_sort():
    sorted_specs = topological_sort(all_specs())
    key_order = {s.key: i for i, s in enumerate(sorted_specs)}
    assert key_order["bbwp_entry"] < key_order["bbwp_exit"]


def test_pamrp_exit_comes_after_pamrp_entry_in_sort():
    sorted_specs = topological_sort(all_specs())
    key_order = {s.key: i for i, s in enumerate(sorted_specs)}
    assert key_order["pamrp_entry"] < key_order["pamrp_exit"]


def test_registry_covers_18_indicators():
    """Exact indicator count check — update when adding new indicators."""
    assert len(INDICATOR_REGISTRY) == 18, (
        f"Expected 18 indicators, got {len(INDICATOR_REGISTRY)}. "
        "Update this count if adding a new indicator."
    )


# ─── Phase 2: calculate_indicators column-set tests ───────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    np.random.seed(42)
    n = 300
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.005,
        "low": close * 0.995, "close": close,
        "volume": np.random.randint(1000, 5000, n).astype(float),
    }, index=idx)


def test_calculate_indicators_all_disabled_has_fallback_columns(sample_df):
    """When all indicators disabled, legacy fallback columns are still present."""
    p = StrategyParams(
        pamrp_enabled=False, pamrp_exit_enabled=False,
        bbwp_enabled=False, bbwp_exit_enabled=False,
    )
    result = SignalGenerator(p).calculate_indicators(sample_df)
    for col in ("pamrp_entry", "pamrp_exit", "pamrp", "bbwp", "bbwp_sma"):
        assert col in result.columns, f"Missing fallback column: {col}"
    assert (result["pamrp_entry"] == 50.0).all()
    assert (result["bbwp"] == 50.0).all()


def test_calculate_indicators_all_enabled_has_expected_columns(sample_df):
    """When all indicators enabled, all output columns are present."""
    p = StrategyParams(
        pamrp_enabled=True, pamrp_exit_enabled=True,
        bbwp_enabled=True, bbwp_exit_enabled=True,
        adx_enabled=True, ma_trend_enabled=True,
        rsi_enabled=True, stoch_rsi_exit_enabled=True,
        volume_enabled=True, supertrend_enabled=True,
        vwap_enabled=True, macd_enabled=True,
        atr_trailing_enabled=True, ma_exit_enabled=True,
    )
    result = SignalGenerator(p).calculate_indicators(sample_df)
    expected = {
        "pamrp_entry", "pamrp_exit", "pamrp",
        "bbwp", "bbwp_sma",
        "adx", "di_plus", "di_minus",
        "ma_fast", "ma_slow",
        "rsi",
        "stoch_k", "stoch_d",
        "volume_ma",
        "supertrend", "st_direction",
        "vwap",
        "macd", "macd_signal", "macd_hist",
        "atr",
        "exit_ma_fast", "exit_ma_slow",
    }
    missing = expected - set(result.columns)
    assert not missing, f"Missing columns: {missing}"


def test_calculate_indicators_does_not_mutate_input(sample_df):
    """calculate_indicators must not modify the original DataFrame."""
    original_cols = set(sample_df.columns)
    p = StrategyParams(pamrp_enabled=True, bbwp_enabled=True)
    SignalGenerator(p).calculate_indicators(sample_df)
    assert set(sample_df.columns) == original_cols
