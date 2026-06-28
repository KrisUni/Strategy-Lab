"""
ui/advisor/panel.py
===================
Right-rail advisor panel scaffold.
Collects user input, displays conversation output, and drives streaming.
No business logic lives here — only input collection and output display.
"""
from __future__ import annotations

import streamlit as st

from ui.advisor.settings import is_advisor_enabled, build_system_prompt

_RAIL_KEY = "advisor_rail_open"
_MSGS_KEY = "advisor_messages"
_INPUT_KEY = "_advisor_input_text"
_PENDING_KEY = "advisor_pending_response"
_STOP_KEY = "advisor_stop_requested"


def init_advisor_rail_state() -> None:
    """Initialize rail session-state keys. Safe to call every rerun."""
    for k, v in [
        (_RAIL_KEY, False),
        (_MSGS_KEY, []),
        (_PENDING_KEY, False),
        (_STOP_KEY, False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v


def is_rail_open() -> bool:
    """True when the advisor rail is expanded."""
    return bool(st.session_state.get(_RAIL_KEY, False))


def get_column_ratio() -> list:
    """Return [main, rail] column ratio for the current rail state."""
    return [3, 1] if is_rail_open() else [20, 1]


def render_advisor_panel() -> None:
    """
    Render the advisor right-rail panel.

    Collapsed (default): one toggle button, nothing else.
    Expanded: conversation history, streaming, and input box.
    No advisor code path executes while the advisor is disabled.
    """
    is_open = is_rail_open()

    if not is_open:
        if st.button("🤖", key="_advisor_rail_open_btn", help="Open AI Advisor"):
            st.session_state[_RAIL_KEY] = True
            st.rerun()
        return

    # ── Header ─────────────────────────────────────────────────────────────────
    hcol_title, hcol_close = st.columns([5, 1])
    with hcol_title:
        st.markdown("**🤖 Advisor**")
    with hcol_close:
        if st.button("✕", key="_advisor_rail_close_btn", help="Close advisor"):
            st.session_state[_RAIL_KEY] = False
            st.rerun()

    st.divider()

    # ── Guard: no advisor code runs while disabled ─────────────────────────────
    if not is_advisor_enabled():
        st.info(
            "Advisor disabled.\n\n"
            "Add an API key in **⚙️ Settings** to enable.",
        )
        return

    # ── Conversation display ───────────────────────────────────────────────────
    messages: list = st.session_state.get(_MSGS_KEY, [])
    if not messages and not st.session_state.get(_PENDING_KEY, False):
        st.caption("Ask the advisor about your strategy…")

    for msg in messages:
        role = msg.get("role", "user")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

    # ── Streaming response (P1.4) ──────────────────────────────────────────────
    if st.session_state.get(_PENDING_KEY, False):
        _stream_next_response(messages)
        return  # rerun happens inside _stream_next_response

    # ── Input ──────────────────────────────────────────────────────────────────
    user_text = st.text_area(
        "Message",
        placeholder="Ask the advisor…",
        key=_INPUT_KEY,
        height=80,
        label_visibility="collapsed",
    )
    if st.button("Send ↑", key="_advisor_send_btn", use_container_width=True):
        text = (user_text or "").strip()
        if text:
            st.session_state[_MSGS_KEY].append({"role": "user", "content": text})
            if _INPUT_KEY in st.session_state:
                del st.session_state[_INPUT_KEY]
            st.session_state[_PENDING_KEY] = True
            st.rerun()


def _stream_next_response(messages: list[dict]) -> None:
    """
    Stream one assistant response and append it to history.

    Always sets advisor_pending_response = False on exit.
    On error: stores a formatted error message as the assistant turn.
    On partial stream interrupted: stores whatever was accumulated.
    """
    from ui.advisor import client as _client  # local to avoid circular at module load
    from ui.advisor.context import assemble_context

    model = st.session_state.get("advisor_model", "claude-sonnet-4-6")
    system = build_system_prompt()

    # Build the API message list from history (skip empty content defensively)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("content", "").strip()
    ]

    # Inject fresh context as a prefix to the first user message so the model
    # always has current session state. Not stored in advisor_messages.
    if api_messages:
        context_text = assemble_context()
        first = api_messages[0]
        api_messages[0] = {
            "role": first["role"],
            "content": f"[Current session context]\n{context_text}\n\n---\n\n{first['content']}",
        }

    full = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for chunk in _client.stream_response(api_messages, model, system):
                full += chunk
                placeholder.markdown(full + " ▌")
            placeholder.markdown(full)
        except Exception as exc:
            err = _client.format_api_error(exc)
            content = (full + f"\n\n*⚠️ {err}*") if full else f"⚠️ {err}"
            placeholder.markdown(content)
            full = content

    st.session_state[_MSGS_KEY].append({"role": "assistant", "content": full})
    st.session_state[_PENDING_KEY] = False
    st.rerun()
