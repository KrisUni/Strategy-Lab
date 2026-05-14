"""
Registry schema and population tests (Phase 1, 2 & 3).

These tests verify that:
1. Every legacy StrategyParams field appears in the registry (or is a known strategy-level param)
2. Registry defaults match dataclass defaults exactly
3. No duplicate param names or indicator keys
4. validate_registry() passes
5. All compute/signal callables are named functions (not lambdas)
6. calculate_indicators produces the expected column set (Phase 2)
7. StrategyParams dict-backed shim behaves correctly (Phase 3)
"""
import numpy as np
import pandas as pd
import pytest
from src.strategy import StrategyParams, SignalGenerator, TradeDirection, _PROVISIONAL_EXIT_PARAMS
from src.indicators.registry import (
    INDICATOR_REGISTRY,
    STRATEGY_LEVEL_PARAMS,
    all_specs,
    build_defaults_from_registry,
    enabled_specs,
    get,
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
    """Every indicator param in StrategyParams defaults is covered by the registry."""
    sp_defaults = set(StrategyParams().to_dict().keys())
    indicator_fields = sp_defaults - STRATEGY_LEVEL_PARAMS - _PROVISIONAL_EXIT_PARAMS
    registry_params = set(build_defaults_from_registry().keys())
    missing = indicator_fields - registry_params
    assert not missing, (
        f"StrategyParams fields not found in registry: {sorted(missing)}\n"
        "Add a ParamSpec for each, or add to STRATEGY_LEVEL_PARAMS if they're not indicators."
    )


def test_registry_defaults_match_strategyparams_defaults():
    """Registry default for each param must equal the StrategyParams default."""
    sp = StrategyParams()
    defaults_sp = {k: v for k, v in sp.to_dict().items()
                   if k not in STRATEGY_LEVEL_PARAMS and k not in _PROVISIONAL_EXIT_PARAMS}
    defaults_reg = build_defaults_from_registry()

    mismatches = {}
    for name, sp_default in defaults_sp.items():
        if name in defaults_reg and defaults_reg[name] != sp_default:
            mismatches[name] = (sp_default, defaults_reg[name])

    assert not mismatches, (
        "Registry defaults differ from StrategyParams defaults:\n"
        + "\n".join(f"  {k}: sp={v[0]!r}, registry={v[1]!r}" for k, v in mismatches.items())
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
    assert "rsi_entry" not in enabled_keys
    assert "macd_entry" not in enabled_keys
    assert "supertrend_entry" not in enabled_keys


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
    assert len(INDICATOR_REGISTRY) == 24, (
        f"Expected 24 indicators, got {len(INDICATOR_REGISTRY)}. "
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
    }
    missing = expected - set(result.columns)
    assert not missing, f"Missing columns: {missing}"


def test_calculate_indicators_does_not_mutate_input(sample_df):
    """calculate_indicators must not modify the original DataFrame."""
    original_cols = set(sample_df.columns)
    p = StrategyParams(pamrp_enabled=True, bbwp_enabled=True)
    SignalGenerator(p).calculate_indicators(sample_df)
    assert set(sample_df.columns) == original_cols


# ─── Phase 3: StrategyParams dict-shim tests ──────────────────────────────────

def test_strategyparams_attribute_access_matches_dict():
    """Every key in to_dict() is reachable as an attribute (non-enum params match directly)."""
    from enum import Enum
    p = StrategyParams()
    d = p.to_dict()
    for key, val in d.items():
        attr = getattr(p, key)
        # Enum attrs store the enum object; to_dict serializes them to .value
        normalized = attr.value if isinstance(attr, Enum) else attr
        assert normalized == val, f"Attribute {key!r} mismatch: {normalized!r} != {val!r}"


def test_strategyparams_unknown_attr_raises():
    """Accessing an unknown attribute must raise AttributeError."""
    p = StrategyParams()
    with pytest.raises(AttributeError):
        _ = p.bogus_attr_xyz


def test_strategyparams_setattr_writes_through():
    """Setting an attribute must update to_dict()."""
    p = StrategyParams()
    p.rsi_length = 21
    assert p.to_dict()["rsi_length"] == 21
    assert p.rsi_length == 21


def test_strategyparams_overrides_applied():
    """Constructor keyword args override registry defaults."""
    p = StrategyParams(rsi_length=99, bbwp_enabled=False)
    assert p.rsi_length == 99
    assert p.bbwp_enabled is False


def test_strategyparams_legacy_pamrp_length_migration():
    """from_dict: pamrp_length (layer 1) → pamrp_entry_ma_length / pamrp_exit_ma_length."""
    p = StrategyParams.from_dict({"pamrp_length": 30})
    assert p.pamrp_entry_ma_length == 30
    assert p.pamrp_exit_ma_length == 30


def test_strategyparams_legacy_pamrp_entry_length_migration():
    """from_dict: pamrp_entry_length (layer 2) → pamrp_entry_ma_length."""
    p = StrategyParams.from_dict({"pamrp_entry_length": 42})
    assert p.pamrp_entry_ma_length == 42


def test_strategyparams_from_dict_direction_ui_strings():
    """from_dict: UI display strings ('Long Only', 'Short Only', 'Both') are coerced."""
    assert StrategyParams.from_dict({"trade_direction": "Long Only"}).trade_direction == TradeDirection.LONG_ONLY
    assert StrategyParams.from_dict({"trade_direction": "Short Only"}).trade_direction == TradeDirection.SHORT_ONLY
    assert StrategyParams.from_dict({"trade_direction": "Both"}).trade_direction == TradeDirection.BOTH


def test_strategyparams_from_dict_direction_storage_strings():
    """from_dict: storage-format strings ('long_only' etc.) are also coerced."""
    assert StrategyParams.from_dict({"trade_direction": "long_only"}).trade_direction == TradeDirection.LONG_ONLY
    assert StrategyParams.from_dict({"trade_direction": "both"}).trade_direction == TradeDirection.BOTH


def test_strategyparams_from_dict_unknown_keys_ignored():
    """from_dict: unrecognised keys are silently dropped."""
    p = StrategyParams.from_dict({"rsi_length": 10, "totally_unknown_key": 999})
    assert p.rsi_length == 10
    with pytest.raises(AttributeError):
        _ = p.totally_unknown_key


def test_strategyparams_to_dict_serializes_enums():
    """to_dict must return plain strings for enum values."""
    p = StrategyParams(trade_direction=TradeDirection.SHORT_ONLY)
    d = p.to_dict()
    assert isinstance(d["trade_direction"], str)
    assert d["trade_direction"] == "short_only"


def test_strategyparams_time_exit_bars_migration():
    """from_dict: time_exit_bars (legacy UI key) fans out to long/short split."""
    p = StrategyParams.from_dict({"time_exit_bars": 15})
    assert p.time_exit_bars_long == 15
    assert p.time_exit_bars_short == 15


def test_ma_exit_reuses_ma_trend_outputs():
    """ma_exit must not define its own fast/slow length params after refactor."""
    spec = get("ma_exit")
    param_names = {p.name for p in spec.params}
    assert "ma_exit_fast" not in param_names
    assert "ma_exit_slow" not in param_names
    assert spec.reuses_outputs_from == ["ma_trend"]


def test_stoch_rsi_exit_has_independent_thresholds():
    """stoch_rsi_exit must expose its own overbought/oversold params after refactor."""
    spec = get("stoch_rsi_exit")
    param_names = {p.name for p in spec.params}
    assert "stoch_rsi_exit_overbought" in param_names
    assert "stoch_rsi_exit_oversold" in param_names
    assert spec.reuses_outputs_from == ["stoch_rsi_entry"]


def test_ma_exit_topological_order():
    sorted_specs = topological_sort(all_specs())
    key_order = {s.key: i for i, s in enumerate(sorted_specs)}
    assert key_order["ma_trend"] < key_order["ma_exit"]


def test_strategyparams_from_dict_does_not_drop_ma_exit_lengths():
    """Forward-fix: ma_exit_fast and ma_exit_slow are valid params again (Issue A)."""
    p = StrategyParams.from_dict({"ma_exit_fast": 5, "ma_exit_slow": 15, "ma_exit_enabled": True})
    d = p.to_dict()
    assert d["ma_exit_fast"] == 5
    assert d["ma_exit_slow"] == 15
    assert d["ma_exit_enabled"] is True


def test_strategyparams_stoch_rsi_exit_seeded_from_entry():
    """from_dict: stoch_rsi_exit thresholds seeded from entry values when absent."""
    p = StrategyParams.from_dict({"stoch_rsi_overbought": 75, "stoch_rsi_oversold": 25})
    assert p.stoch_rsi_exit_overbought == 75
    assert p.stoch_rsi_exit_oversold == 25


def test_strategyparams_stoch_rsi_exit_not_overwritten():
    """from_dict: explicit exit thresholds are not overwritten by entry values."""
    p = StrategyParams.from_dict({
        "stoch_rsi_overbought": 75,
        "stoch_rsi_oversold": 25,
        "stoch_rsi_exit_overbought": 60,
        "stoch_rsi_exit_oversold": 40,
    })
    assert p.stoch_rsi_exit_overbought == 60
    assert p.stoch_rsi_exit_oversold == 40


def test_strategyparams_round_trips_provisional_exit_params():
    """Provisional exit params survive StrategyParams.from_dict() → to_dict()."""
    p = StrategyParams.from_dict({
        "rsi_length": 21,
        "rsi_exit_length": 7,
        "bbwp_length": 8,
    })
    d = p.to_dict()
    assert d["rsi_length"] == 21
    assert d["rsi_exit_length"] == 7        # not overwritten by seeding
    assert d["bbwp_length"] == 8
    assert d["bbwp_exit_length"] == 8       # seeded from entry
