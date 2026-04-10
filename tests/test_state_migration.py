from ui.state_migration import (
    migrate_legacy_pamrp_params,
    migrate_legacy_pamrp_pins,
)


def test_migrate_legacy_pamrp_params_splits_shared_length():
    migrated = migrate_legacy_pamrp_params({
        'pamrp_length': 34,
        'pamrp_enabled': True,
    })

    assert 'pamrp_length' not in migrated
    assert migrated['pamrp_entry_length'] == 34
    assert migrated['pamrp_exit_length'] == 34


def test_migrate_legacy_pamrp_params_keeps_explicit_split_lengths():
    migrated = migrate_legacy_pamrp_params({
        'pamrp_length': 34,
        'pamrp_entry_length': 11,
        'pamrp_exit_length': 27,
    })

    assert 'pamrp_length' not in migrated
    assert migrated['pamrp_entry_length'] == 11
    assert migrated['pamrp_exit_length'] == 27


def test_migrate_legacy_pamrp_pins_replaces_legacy_pin_with_split_pins():
    migrated = migrate_legacy_pamrp_pins({
        'pamrp_length',
        'bbwp_length',
    })

    assert 'pamrp_length' not in migrated
    assert 'pamrp_entry_length' in migrated
    assert 'pamrp_exit_length' in migrated
    assert 'bbwp_length' in migrated
