# Adding a New Indicator

Each indicator is a single file in `src/indicators/specs/`. Adding one means:

1. Create the spec file
2. Register it in `src/indicators/specs/__init__.py`
3. Add its default params to `ui/session.py` → `get_default_params()`

That's it. No if-else chains, no widget boilerplate, no optimizer blocks.

---

## Step 1 — Create the spec file

Use `src/indicators/specs/rsi.py` as a template. Minimal structure:

```python
# src/indicators/specs/my_indicator.py
"""My indicator entry spec."""
from typing import Any, Dict
import pandas as pd
from ..registry import IndicatorSpec, ParamSpec, register


def compute_my_indicator(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    # Write output column(s) onto df. Receives ALL params.
    df["my_col"] = ...
    return df


def long_signal_my_indicator(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["my_col"] > params["my_threshold"]


def short_signal_my_indicator(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    return df["my_col"] < params["my_threshold"]


register(IndicatorSpec(
    key="my_indicator",           # unique, snake_case
    name="My Indicator",          # UI display name
    group="entry",                # "entry", "exit", or "risk"
    order=99,                     # ordering within the sidebar group
    enable_param="my_indicator_enabled",
    params=[
        ParamSpec("my_indicator_enabled", "bool", False,
                  optimize=False, label="My Indicator enabled", order=0),
        ParamSpec("my_threshold", "int", 50, min=1, max=100,
                  label="Threshold", order=1),
    ],
    compute=compute_my_indicator,
    outputs=["my_col"],
    long_signal=long_signal_my_indicator,
    short_signal=short_signal_my_indicator,
))
```

### Key fields

| Field | Notes |
|-------|-------|
| `key` | Unique string. Used as the registry lookup key. |
| `group` | `"entry"` → entry filter. `"exit"` → signal-based exit. `"risk"` → stop/TP/trailing. |
| `enable_param` | Must match a `ParamSpec` with `type="bool"` and `optimize=False`. |
| `outputs` | Column names your `compute` function writes to `df`. |
| `long_signal` / `short_signal` | Return a boolean `pd.Series`. Set to `None` if the indicator doesn't generate that direction. |
| `reuses_outputs_from` | List of other spec `key`s whose columns you read. Ensures topological order. |

### `ParamSpec` fields

| Field | Notes |
|-------|-------|
| `type` | `"int"`, `"float"`, `"categorical"`, `"bool"`. |
| `min` / `max` | Required for numeric types. Also sets the Optuna search range. |
| `step` | Optional. UI slider step (defaults: 1 for int, 0.5 for float). |
| `choices` | Required for `"categorical"`. Use a `tuple` (hashable). |
| `direction` | `"long"`, `"short"`, or `"both"`. Optimizer skips the param when the run direction doesn't match. |
| `optimize` | `False` → param is never included in Optuna trials (use for enable toggles and display-only params). |
| `label` | UI widget label. Falls back to `name` if empty. |
| `order` | Widget ordering within the expander. |

---

## Step 2 — Register in `__init__.py`

Open `src/indicators/specs/__init__.py` and import your new file:

```python
from . import my_indicator  # noqa: F401
```

The `register()` call at module level runs on import and appends your spec to `INDICATOR_REGISTRY`.

---

## Step 3 — Add defaults to `get_default_params()`

Open `ui/session.py` and add your param defaults inside `get_default_params()`:

```python
'my_indicator_enabled': False,
'my_threshold': 50,
```

This keeps the session state initialiser in sync for users whose cached state predates your indicator.

---

## What you get for free

Once registered, the indicator is automatically wired into:

- **Sidebar**: expander with widgets rendered by `render_indicator_section()`
- **Signal pipeline**: `generate_entry_signals()` / `generate_exit_signals()` call your signal functions
- **Optimizer**: `_build_params_from_trial()` includes your params in the Optuna search space
- **Pin expander**: optimizable params appear in the "Pin Parameters" UI in the Optimize tab
- **Active filters display**: your indicator's name appears in the active-filters caption when enabled

---

## Validation

`validate_registry()` runs automatically at import time. It will raise `ValueError` if:

- Your `key` is already taken
- Any `param.name` conflicts with an existing param across the registry
- `enable_param` is missing from `params` or is not `type="bool"`
- A numeric param is missing `min` or `max`
- A categorical param is missing `choices`
- Any callable is a lambda (use named functions for picklability and debuggability)

Run `python -c "from src.indicators.specs import *"` to trigger validation without starting the app.
