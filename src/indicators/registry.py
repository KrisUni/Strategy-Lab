"""
Indicator Registry — single source of truth for all strategy indicators.

Each indicator is described by an IndicatorSpec that carries its params,
compute function, and signal callables. Consumers (calculate_indicators,
generate_entry_signals, _build_params_from_trial, UI sidebar) iterate this
registry instead of maintaining per-indicator if-else chains.

Phase 1: schema + population only. No consumer changes yet.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd

# ─── Type aliases ─────────────────────────────────────────────────────────────

ParamType = Literal["int", "float", "categorical", "bool"]
Direction = Literal["long", "short", "both"]
Group = Literal["entry", "exit", "risk", "core"]
SignalRole = Literal["trigger", "filter"]

# Strategy-level params that live outside the indicator registry.
# These are owned by StrategyParams itself, not by any IndicatorSpec.
STRATEGY_LEVEL_PARAMS: frozenset = frozenset({
    "trade_direction",
    "entry_operator",
    "exit_operator",
    "allow_same_bar_exit",
    "allow_same_bar_reversal",
    "entry_conflict_mode",
    "entry_exit_conflict_mode",
    "position_size_pct",
    "use_kelly",
    "kelly_fraction",
})


# ─── Schema ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter owned by an indicator."""
    name: str
    type: ParamType
    default: Any
    min: Optional[float] = None          # numeric only
    max: Optional[float] = None          # numeric only
    step: Optional[float] = None         # UI step; 1 for int, 0.5 for float if None
    choices: Optional[tuple] = None      # categorical only (use tuple for hashability)
    direction: Direction = "both"        # limits optimizer suggestion to this direction
    label: str = ""                      # UI label; falls back to name if empty
    help: str = ""                       # UI tooltip
    order: int = 0                       # UI ordering within indicator
    optimize: bool = True                # False → never appears in Optuna search


@dataclass
class Palette:
    """Color vocabulary for indicator renders. All render callables pull from here."""
    primary: str = "#3b82f6"                 # entry-side lines (blue)
    secondary: str = "#f59e0b"               # exit-side / signal lines (amber)
    bullish: str = "#10b981"                 # long direction (green)
    bearish: str = "#ef4444"                 # short direction (red)
    neutral_grid: str = "rgba(255,255,255,0.2)"
    ob_line: str = "rgba(239,68,68,0.6)"     # overbought hlines
    os_line: str = "rgba(16,185,129,0.6)"    # oversold hlines
    purple: str = "#a855f7"
    sky: str = "#60a5fa"
    exit_hline: str = "rgba(245,158,11,0.7)" # exit threshold hlines


PALETTE = Palette()


@dataclass
class PlotContext:
    """Passed to every render/contribute callable."""
    fig: Any               # go.Figure — plotly not imported in registry
    row: int               # subplot row (1 = price panel)
    col: int               # always 1
    df: pd.DataFrame       # OHLCV
    idf: pd.DataFrame      # indicator outputs
    params: Dict[str, Any]
    palette: Palette
    is_subplot: bool = True


@dataclass
class PlotSpec:
    """Declares how an indicator renders itself."""
    kind: Literal["overlay", "panel"]
    render: Callable[[PlotContext], None]
    panel_title: str = ""
    panel_y_range: Optional[Tuple[float, float]] = None
    panel_height_weight: float = 1.0
    owner_for_columns: List[str] = field(default_factory=list)
    # Contributor hook: called on the owner's panel after owner.render().
    # Define this to prevent getting a separate row when the owner is enabled.
    # Every spec with contribute must also have render capable of running standalone.
    contribute: Optional[Callable[[PlotContext], None]] = None


@dataclass
class IndicatorSpec:
    """One self-contained indicator declaration.

    Treat as immutable after registration. Using a regular dataclass (not
    frozen=True) because Callable fields are not hashable.
    """
    key: str                             # unique registry key, e.g. "rsi"
    name: str                            # display name, e.g. "RSI"
    group: Group                         # determines sidebar section + signal pipeline
    order: int                           # UI ordering within group
    enable_param: str                    # name of the bool ParamSpec that enables it
    params: List[ParamSpec]              # all tunable params owned by this indicator
    compute: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
        # Pure function: (df, params_dict) → df with new columns added.
        # Receives ALL params (registry-wide). Must not mutate df in-place on
        # columns it doesn't own (use df = df.copy() selectively).
    outputs: List[str]                   # column names this indicator writes to df

    long_signal: Optional[Callable[[pd.DataFrame, Dict[str, Any]], pd.Series]] = None
        # Returns boolean Series for long entry/exit. None = indicator doesn't
        # contribute to long signals.
    short_signal: Optional[Callable[[pd.DataFrame, Dict[str, Any]], pd.Series]] = None

    reuses_outputs_from: List[str] = field(default_factory=list)
        # Keys of other indicators whose outputs this one reads.
        # Used for topological sort (dependency comes first) and dedup
        # (if a dep is enabled, its compute already ran).
        # Example: bbwp_exit reads the "bbwp" column from bbwp_entry.

    signal_role: SignalRole = "trigger"
        # "trigger" — entry fires on False→True edge of long_signal/short_signal.
        #             Re-entry blocked while the same signal remains continuously True.
        # "filter"  — entry fires only when a trigger fires AND this signal is True.
        #             The filter describes regime/eligibility, not timing.
        # Exit specs: field is ignored (exits have no re-entry problem).

    signal_mode_param: Optional[str] = None
        # Name of a ParamSpec (categorical "trigger"|"filter") that lets the user
        # override signal_role at runtime. Set only for indicators that can
        # legitimately serve either role (e.g. PAMRP, RSI, Stoch RSI).

    plot: Optional[PlotSpec] = None
        # None = no visual. When set, the chart engine calls plot.render(ctx).
        # Contributors (plot.contribute defined) don't get their own panel row
        # when their owner is also enabled — contribute() runs on the owner's row.


# ─── Global registry ──────────────────────────────────────────────────────────

INDICATOR_REGISTRY: List[IndicatorSpec] = []
_REGISTRY_INDEX: Dict[str, IndicatorSpec] = {}
_PROVISIONAL_KEYS: set = set()  # keys of in-memory provisional indicators (restart clears)


def register(spec: IndicatorSpec) -> None:
    """Append a spec to the registry. Called at module import time by each spec file."""
    INDICATOR_REGISTRY.append(spec)
    _REGISTRY_INDEX[spec.key] = spec


def get(key: str) -> IndicatorSpec:
    """Return the spec for a given indicator key. Raises KeyError if not found."""
    return _REGISTRY_INDEX[key]


def all_specs() -> List[IndicatorSpec]:
    """Return all registered specs in registration order."""
    return list(INDICATOR_REGISTRY)


def enabled_specs(params: Dict[str, Any]) -> List[IndicatorSpec]:
    """Return specs whose enable_param is True in params."""
    return [s for s in INDICATOR_REGISTRY if params.get(s.enable_param, False)]


def build_defaults_from_registry() -> Dict[str, Any]:
    """Build a flat dict of all indicator param defaults from the registry.

    Does NOT include strategy-level params (trade_direction, etc.) — those are
    seeded separately in StrategyParams.__init__.
    """
    defaults: Dict[str, Any] = {}
    for spec in INDICATOR_REGISTRY:
        for p in spec.params:
            if p.name not in defaults:   # first registration wins
                defaults[p.name] = p.default
    return defaults


# ─── Topological sort ─────────────────────────────────────────────────────────

def topological_sort(specs: List[IndicatorSpec]) -> List[IndicatorSpec]:
    """Sort specs so that dependencies (reuses_outputs_from) come before dependents.

    Only specs present in the input list are considered; missing deps are skipped.
    """
    key_to_spec = {s.key: s for s in specs}
    visited: set = set()
    result: List[IndicatorSpec] = []

    def visit(spec: IndicatorSpec) -> None:
        if spec.key in visited:
            return
        for dep_key in spec.reuses_outputs_from:
            if dep_key in key_to_spec:
                visit(key_to_spec[dep_key])
        visited.add(spec.key)
        result.append(spec)

    for spec in specs:
        visit(spec)

    return result


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_registry() -> None:
    """Validate registry integrity. Called automatically at spec import time.

    Raises ValueError on any violation.
    """
    seen_keys: set = set()
    seen_param_names: set = set()

    for spec in INDICATOR_REGISTRY:
        # Duplicate indicator keys
        if spec.key in seen_keys:
            raise ValueError(f"Duplicate indicator key: '{spec.key}'")
        seen_keys.add(spec.key)

        # enable_param must exist in this spec's params as a bool
        enable_param_found = False
        for p in spec.params:
            if p.name == spec.enable_param:
                if p.type != "bool":
                    raise ValueError(
                        f"Indicator '{spec.key}': enable_param '{spec.enable_param}' "
                        f"must have type='bool', got '{p.type}'"
                    )
                enable_param_found = True
            # Duplicate param names across registry
            if p.name in seen_param_names:
                raise ValueError(
                    f"Duplicate param name '{p.name}' (in indicator '{spec.key}')"
                )
            seen_param_names.add(p.name)

            # Numeric params need min/max
            if p.type in ("int", "float"):
                if p.min is None or p.max is None:
                    raise ValueError(
                        f"Indicator '{spec.key}', param '{p.name}': "
                        f"numeric params must have min and max"
                    )
            # Categorical params need choices
            if p.type == "categorical":
                if not p.choices:
                    raise ValueError(
                        f"Indicator '{spec.key}', param '{p.name}': "
                        f"categorical params must have choices"
                    )

        if not enable_param_found:
            raise ValueError(
                f"Indicator '{spec.key}': enable_param '{spec.enable_param}' "
                f"not found in spec.params"
            )

    # Callable names must not be lambdas
    for spec in INDICATOR_REGISTRY:
        for fn_name, fn in [
            ("compute", spec.compute),
            ("long_signal", spec.long_signal),
            ("short_signal", spec.short_signal),
        ]:
            if fn is not None and fn.__name__.startswith("<lambda>"):
                raise ValueError(
                    f"Indicator '{spec.key}'.{fn_name} is a lambda — "
                    "use a named function for debuggability and picklability"
                )

    # PlotSpec validation
    owner_columns: Dict[str, str] = {}   # col → spec_key claiming ownership
    for spec in INDICATOR_REGISTRY:
        ps = spec.plot
        if ps is None:
            continue
        if ps.kind == "panel" and not ps.panel_title:
            raise ValueError(
                f"Indicator '{spec.key}': PlotSpec kind='panel' requires non-empty panel_title"
            )
        if ps.render.__name__.startswith("<lambda>"):
            raise ValueError(
                f"Indicator '{spec.key}': plot.render must be a named function, not a lambda"
            )
        if ps.contribute is not None and ps.contribute.__name__.startswith("<lambda>"):
            raise ValueError(
                f"Indicator '{spec.key}': plot.contribute must be a named function, not a lambda"
            )
        for col in ps.owner_for_columns:
            if col in owner_columns:
                raise ValueError(
                    f"Column '{col}' claimed as owner by both "
                    f"'{owner_columns[col]}' and '{spec.key}'"
                )
            owner_columns[col] = spec.key
