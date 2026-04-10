"""
ui/state_sync.py
================
Helpers for Streamlit widget state that should follow an upstream source value
until the user intentionally overrides the widget locally.
"""

from typing import Any

import streamlit as st


def sync_following_session_value(session_key: str, source_value: Any, last_source_key: str) -> None:
    """
    Keep a widget value synced to an upstream source until the widget is manually changed.

    Behavior:
    - first render: widget gets the source value
    - source changes and widget still mirrors the old source: update widget
    - source changes and widget was manually overridden: keep the override
    """
    previous_source_value = st.session_state.get(last_source_key)

    if session_key not in st.session_state:
        st.session_state[session_key] = source_value
    elif previous_source_value is not None and st.session_state[session_key] == previous_source_value:
        st.session_state[session_key] = source_value

    st.session_state[last_source_key] = source_value
