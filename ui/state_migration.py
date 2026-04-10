"""
ui/state_migration.py
=====================
Helpers for migrating persisted UI/session state across renamed parameters.
"""

from typing import Any, Dict, Iterable, Set

LEGACY_PAMRP_LENGTH_KEY = 'pamrp_length'
PAMRP_LENGTH_KEYS = ('pamrp_entry_length', 'pamrp_exit_length')


def migrate_legacy_pamrp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand the legacy shared PAMRP length into the split entry/exit lengths.

    Returns a copy so callers can safely normalize session state, optimization
    results, or other persisted payloads without mutating the original object.
    """
    migrated = dict(params)
    legacy_length = migrated.pop(LEGACY_PAMRP_LENGTH_KEY, None)
    if legacy_length is not None:
        migrated.setdefault('pamrp_entry_length', legacy_length)
        migrated.setdefault('pamrp_exit_length', legacy_length)
    return migrated


def migrate_legacy_pamrp_pins(pinned_params: Iterable[str] | None) -> Set[str]:
    """
    Replace the legacy shared-length pin with the split entry/exit pins.

    The old `pamrp_length` controlled both paths, so the safest migration is to
    pin both renamed fields.
    """
    migrated = set(pinned_params or ())
    if LEGACY_PAMRP_LENGTH_KEY in migrated:
        migrated.discard(LEGACY_PAMRP_LENGTH_KEY)
        migrated.update(PAMRP_LENGTH_KEYS)
    return migrated
