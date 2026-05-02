"""
ui/state_migration.py
=====================
Helpers for migrating persisted UI/session state across renamed parameters.
"""

from typing import Any, Dict, Iterable, Set

LEGACY_PAMRP_LENGTH_KEY  = 'pamrp_length'
LEGACY_PAMRP_ENTRY_KEY   = 'pamrp_entry_length'
LEGACY_PAMRP_EXIT_KEY    = 'pamrp_exit_length'
PAMRP_MA_LENGTH_KEYS     = ('pamrp_entry_ma_length', 'pamrp_exit_ma_length')


def migrate_legacy_pamrp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate persisted PAMRP params across two rename generations.

    Layer 1 (oldest): pamrp_length → pamrp_entry_length / pamrp_exit_length
    Layer 2 (FIX-11): pamrp_entry_length → pamrp_entry_ma_length
                      pamrp_exit_length  → pamrp_exit_ma_length

    Returns a copy; does not mutate the original.
    """
    migrated = dict(params)

    # Layer 1
    legacy_length = migrated.pop(LEGACY_PAMRP_LENGTH_KEY, None)
    if legacy_length is not None:
        migrated.setdefault(LEGACY_PAMRP_ENTRY_KEY, legacy_length)
        migrated.setdefault(LEGACY_PAMRP_EXIT_KEY, legacy_length)

    # Layer 2
    legacy_entry = migrated.pop(LEGACY_PAMRP_ENTRY_KEY, None)
    if legacy_entry is not None:
        migrated.setdefault('pamrp_entry_ma_length', legacy_entry)
    legacy_exit = migrated.pop(LEGACY_PAMRP_EXIT_KEY, None)
    if legacy_exit is not None:
        migrated.setdefault('pamrp_exit_ma_length', legacy_exit)

    return migrated


def migrate_legacy_pamrp_pins(pinned_params: Iterable[str] | None) -> Set[str]:
    """
    Replace legacy PAMRP length pins with the current ma_length pin names.

    Handles both generations of legacy pin names.
    """
    migrated = set(pinned_params or ())

    # Layer 1: pamrp_length → entry/exit ma_length pins
    if LEGACY_PAMRP_LENGTH_KEY in migrated:
        migrated.discard(LEGACY_PAMRP_LENGTH_KEY)
        migrated.update(PAMRP_MA_LENGTH_KEYS)

    # Layer 2: pamrp_entry_length / pamrp_exit_length → ma_length pins
    if LEGACY_PAMRP_ENTRY_KEY in migrated:
        migrated.discard(LEGACY_PAMRP_ENTRY_KEY)
        migrated.add('pamrp_entry_ma_length')
    if LEGACY_PAMRP_EXIT_KEY in migrated:
        migrated.discard(LEGACY_PAMRP_EXIT_KEY)
        migrated.add('pamrp_exit_ma_length')

    return migrated
