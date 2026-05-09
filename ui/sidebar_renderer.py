"""
ui/sidebar_renderer.py
======================
Registry-driven sidebar widget rendering.

render_indicator_section() iterates IndicatorSpecs for a given group,
renders one expander per spec, and returns nothing — it mutates `p` in place.

Widget keys use the canonical pattern `widget_{param.name}`.
This is stable across reruns and unique by registry validation guarantee.
"""
from typing import Any, Dict

import streamlit as st

from src.indicators.registry import INDICATOR_REGISTRY, ParamSpec


def render_param_widget(param: ParamSpec, current_value: Any) -> Any:
    """Render a single Streamlit widget for a ParamSpec. Returns the new value."""
    key   = f"widget_{param.name}"
    label = param.label or param.name

    if param.type == "bool":
        return st.toggle(label, value=bool(current_value), key=key)

    if param.type == "int":
        step = int(param.step) if param.step is not None else 1
        return st.slider(
            label,
            min_value=int(param.min),
            max_value=int(param.max),
            value=int(current_value),
            step=step,
            key=key,
        )

    if param.type == "float":
        step = float(param.step) if param.step is not None else 0.5
        return st.slider(
            label,
            min_value=float(param.min),
            max_value=float(param.max),
            value=float(current_value),
            step=step,
            key=key,
        )

    if param.type == "categorical":
        choices = list(param.choices)
        try:
            idx = choices.index(current_value)
        except (ValueError, TypeError):
            idx = 0
        return st.selectbox(label, choices, index=idx, key=key)

    return current_value


def render_indicator_section(group: str, p: Dict[str, Any]) -> None:
    """
    Render all indicator expanders for a given group ("entry", "exit", "risk").
    Mutates `p` in place with updated param values.
    """
    specs = sorted(
        (s for s in INDICATOR_REGISTRY if s.group == group),
        key=lambda s: s.order,
    )

    for spec in specs:
        # Non-optimizable enable params (type=bool, optimize=False) double as the
        # expander toggle — render them first so the expander header is descriptive.
        with st.expander(spec.name, expanded=False):
            # Enable toggle (always the enable_param)
            enable_ps = next(ps for ps in spec.params if ps.name == spec.enable_param)
            p[spec.enable_param] = render_param_widget(
                enable_ps, p.get(spec.enable_param, enable_ps.default)
            )

            if p[spec.enable_param]:
                for param in sorted(spec.params, key=lambda ps: ps.order):
                    if param.name == spec.enable_param:
                        continue
                    p[param.name] = render_param_widget(
                        param, p.get(param.name, param.default)
                    )
