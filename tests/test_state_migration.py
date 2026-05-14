import pytest

from ui.state_migration import (
    migrate_legacy_pamrp_params,
    migrate_legacy_pamrp_pins,
    migrate_legacy_ma_exit_params,
    migrate_legacy_ma_exit_pins,
    migrate_legacy_stoch_rsi_exit_params,
    migrate_exit_params_from_entry_defaults,
    migrate_exit_pins_from_entry_pins,
)


def test_migrate_legacy_pamrp_params_splits_shared_length():
    """pamrp_length flows through both migration layers to the new ma_length fields."""
    migrated = migrate_legacy_pamrp_params({
        'pamrp_length': 34,
        'pamrp_enabled': True,
    })

    assert 'pamrp_length' not in migrated
    assert 'pamrp_entry_length' not in migrated
    assert 'pamrp_exit_length' not in migrated
    assert migrated['pamrp_entry_ma_length'] == 34
    assert migrated['pamrp_exit_ma_length'] == 34


def test_migrate_legacy_pamrp_params_layer2_explicit_entry_exit_lengths():
    """pamrp_entry_length / pamrp_exit_length (layer-2 legacy) migrate to ma_length."""
    migrated = migrate_legacy_pamrp_params({
        'pamrp_entry_length': 11,
        'pamrp_exit_length': 27,
    })

    assert 'pamrp_entry_length' not in migrated
    assert 'pamrp_exit_length' not in migrated
    assert migrated['pamrp_entry_ma_length'] == 11
    assert migrated['pamrp_exit_ma_length'] == 27


def test_migrate_legacy_pamrp_params_does_not_overwrite_explicit_new_names():
    """Existing pamrp_entry_ma_length is not overwritten by legacy migration."""
    migrated = migrate_legacy_pamrp_params({
        'pamrp_length': 34,
        'pamrp_entry_ma_length': 99,
    })

    assert migrated['pamrp_entry_ma_length'] == 99
    assert migrated['pamrp_exit_ma_length'] == 34


def test_migrate_legacy_pamrp_pins_replaces_legacy_pin_with_new_names():
    migrated = migrate_legacy_pamrp_pins({
        'pamrp_length',
        'bbwp_length',
    })

    assert 'pamrp_length' not in migrated
    assert 'pamrp_entry_length' not in migrated
    assert 'pamrp_exit_length' not in migrated
    assert 'pamrp_entry_ma_length' in migrated
    assert 'pamrp_exit_ma_length' in migrated
    assert 'bbwp_length' in migrated


def test_migrate_legacy_pamrp_pins_layer2_entry_exit_length_pins():
    """pamrp_entry_length / pamrp_exit_length pins migrate to ma_length pins."""
    migrated = migrate_legacy_pamrp_pins({
        'pamrp_entry_length',
        'pamrp_exit_length',
        'stop_loss_pct_long',
    })

    assert 'pamrp_entry_length' not in migrated
    assert 'pamrp_exit_length' not in migrated
    assert 'pamrp_entry_ma_length' in migrated
    assert 'pamrp_exit_ma_length' in migrated
    assert 'stop_loss_pct_long' in migrated


def test_migrate_legacy_ma_exit_params_is_noop():
    out = migrate_legacy_ma_exit_params({"ma_exit_fast": 10, "ma_exit_slow": 20})
    assert out == {"ma_exit_fast": 10, "ma_exit_slow": 20}


def test_migrate_legacy_ma_exit_pins_is_noop():
    pins = migrate_legacy_ma_exit_pins({"ma_exit_fast", "ma_exit_slow", "rsi_length"})
    assert pins == {"ma_exit_fast", "ma_exit_slow", "rsi_length"}


def test_migrate_legacy_stoch_rsi_exit_params_seeds_from_entry():
    migrated = migrate_legacy_stoch_rsi_exit_params({
        "stoch_rsi_overbought": 75,
        "stoch_rsi_oversold": 25,
    })
    assert migrated["stoch_rsi_exit_overbought"] == 75
    assert migrated["stoch_rsi_exit_oversold"] == 25


def test_migrate_legacy_stoch_rsi_exit_params_does_not_overwrite_existing():
    migrated = migrate_legacy_stoch_rsi_exit_params({
        "stoch_rsi_overbought": 75,
        "stoch_rsi_oversold": 25,
        "stoch_rsi_exit_overbought": 60,
        "stoch_rsi_exit_oversold": 40,
    })
    assert migrated["stoch_rsi_exit_overbought"] == 60
    assert migrated["stoch_rsi_exit_oversold"] == 40


# ─── Independent-exit migration tests ────────────────────────────────────────

@pytest.mark.parametrize("entry_key,exit_key", [
    ("rsi_length",            "rsi_exit_length"),
    ("bbwp_length",           "bbwp_exit_length"),
    ("bbwp_lookback",         "bbwp_exit_lookback"),
    ("bbwp_sma_length",       "bbwp_exit_sma_length"),
    ("adx_length",            "adx_exit_length"),
    ("adx_smoothing",         "adx_exit_smoothing"),
    ("macd_fast",             "macd_exit_fast"),
    ("macd_slow",             "macd_exit_slow"),
    ("macd_signal",           "macd_exit_signal"),
    ("volume_ma_length",      "volume_exit_ma_length"),
    ("supertrend_period",     "supertrend_exit_period"),
    ("supertrend_multiplier", "supertrend_exit_multiplier"),
    ("stoch_rsi_length",      "stoch_rsi_exit_length"),
    ("stoch_rsi_k",           "stoch_rsi_exit_k"),
    ("stoch_rsi_d",           "stoch_rsi_exit_d"),
    ("stoch_rsi_overbought",  "stoch_rsi_exit_overbought"),
    ("stoch_rsi_oversold",    "stoch_rsi_exit_oversold"),
    ("ma_fast_length",        "ma_exit_fast"),
    ("ma_slow_length",        "ma_exit_slow"),
    ("ma_type",               "ma_exit_type"),
])
def test_migrate_exit_params_seeds_each_mapping(entry_key, exit_key):
    migrated = migrate_exit_params_from_entry_defaults({entry_key: "SENTINEL"})
    assert migrated[exit_key] == "SENTINEL"


def test_migrate_exit_params_does_not_overwrite_existing_exit_value():
    migrated = migrate_exit_params_from_entry_defaults({
        "rsi_length": 7,
        "rsi_exit_length": 21,
    })
    assert migrated["rsi_exit_length"] == 21


def test_migrate_exit_params_no_entry_value_no_seeding():
    migrated = migrate_exit_params_from_entry_defaults({"unrelated_key": 1})
    assert "rsi_exit_length" not in migrated
    assert "bbwp_exit_length" not in migrated


def test_migrate_exit_params_returns_copy_does_not_mutate():
    original = {"rsi_length": 14}
    migrated = migrate_exit_params_from_entry_defaults(original)
    assert "rsi_exit_length" in migrated
    assert "rsi_exit_length" not in original


def test_migrate_exit_pins_mirrors_entry_pins():
    migrated = migrate_exit_pins_from_entry_pins({"rsi_length", "macd_fast"})
    assert "rsi_exit_length" in migrated
    assert "macd_exit_fast" in migrated
    assert "rsi_length" in migrated  # entry pin not removed


def test_migrate_exit_pins_unrelated_pins_untouched():
    migrated = migrate_exit_pins_from_entry_pins({"stop_loss_pct_long", "trailing_stop_pct"})
    assert migrated == {"stop_loss_pct_long", "trailing_stop_pct"}


def test_migrate_exit_pins_idempotent():
    once = migrate_exit_pins_from_entry_pins({"rsi_length"})
    twice = migrate_exit_pins_from_entry_pins(once)
    assert once == twice
