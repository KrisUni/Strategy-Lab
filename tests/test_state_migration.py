from ui.state_migration import (
    migrate_legacy_pamrp_params,
    migrate_legacy_pamrp_pins,
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
