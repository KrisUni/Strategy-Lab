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


# ─── ma_exit migration ────────────────────────────────────────────────────────

LEGACY_MA_EXIT_FAST = "ma_exit_fast"
LEGACY_MA_EXIT_SLOW = "ma_exit_slow"


def migrate_legacy_ma_exit_params(params: Dict[str, Any]) -> Dict[str, Any]:
    # DEPRECATED — kept as no-op for backward compatibility; can be removed once no caller exists.
    # ma_exit_fast / ma_exit_slow are valid params again (Issue A forward-fix).
    return dict(params)


def migrate_legacy_ma_exit_pins(pinned_params: Iterable[str] | None) -> Set[str]:
    # DEPRECATED — kept as no-op for backward compatibility; can be removed once no caller exists.
    # ma_exit_fast / ma_exit_slow are valid pins again (Issue A forward-fix).
    return set(pinned_params or ())


# ─── stoch_rsi_exit migration ─────────────────────────────────────────────────

def migrate_legacy_stoch_rsi_exit_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Seed new exit-specific overbought/oversold thresholds from the entry values
    if they're absent. This preserves the old behavior (entry and exit using
    the same thresholds) for users who don't customize.
    """
    migrated = dict(params)
    entry_ob = migrated.get("stoch_rsi_overbought")
    entry_os = migrated.get("stoch_rsi_oversold")
    if entry_ob is not None:
        migrated.setdefault("stoch_rsi_exit_overbought", entry_ob)
    if entry_os is not None:
        migrated.setdefault("stoch_rsi_exit_oversold", entry_os)
    return migrated


# ─── Independent-exit migration ──────────────────────────────────────────────

# Each tuple: (entry_param_name, exit_param_name)
# Used by migrate_exit_params_from_entry_defaults to seed user-customized
# entry values into the new exit params on first load post-refactor.
_ENTRY_TO_EXIT_PARAM_MAPPINGS: tuple = (
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
)


def migrate_exit_params_from_entry_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Seed missing *_exit_* computation params from their entry counterparts.

    For each (entry_key, exit_key) mapping, if the exit key is absent from
    params and the entry key is present, copy the entry value to the exit key.
    Never overwrites an existing exit value — explicit user customization wins.

    Returns a copy; does not mutate the original.

    Background: Issue B refactors every exit indicator to own its own set of
    computation params. Without this migration, users upgrading from a state
    with customized entry params would see exit params populated with spec
    defaults rather than their entry customizations.
    """
    migrated = dict(params)
    for entry_key, exit_key in _ENTRY_TO_EXIT_PARAM_MAPPINGS:
        if exit_key in migrated:
            continue  # explicit value already present, don't overwrite
        if entry_key in migrated:
            migrated[exit_key] = migrated[entry_key]
    return migrated


def migrate_exit_pins_from_entry_pins(pinned_params: Iterable[str] | None) -> Set[str]:
    """
    For every entry param that the user has pinned, also pin the corresponding
    exit param. Before the refactor, an entry pin like 'rsi_length' implicitly
    held the exit's length fixed too (because the exit read the entry's param).
    After the refactor, the exit has its own param. Without this migration,
    a user's pinned-param set silently stops constraining the exit side.

    Returns a new set; does not mutate the input.
    """
    migrated = set(pinned_params or ())
    for entry_key, exit_key in _ENTRY_TO_EXIT_PARAM_MAPPINGS:
        if entry_key in migrated:
            migrated.add(exit_key)
    return migrated
