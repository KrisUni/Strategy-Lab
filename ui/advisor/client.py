"""
ui/advisor/client.py
====================
Streaming Anthropic API client for the advisor rail.
No UI logic — only API communication and error formatting.
"""
from __future__ import annotations

from typing import Iterator

import anthropic
import streamlit as st

_STOP_KEY = "advisor_stop_requested"


def get_client() -> anthropic.Anthropic | None:
    """Return an Anthropic client if a key is present, else None."""
    key = st.session_state.get("advisor_api_key", "").strip()
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def stream_response(
    messages: list[dict],
    model: str,
    system_prompt: str,
) -> Iterator[str]:
    """
    Yield text chunks from a streaming Anthropic API call.

    Stops early when advisor_stop_requested is set in session state.
    Raises anthropic API exceptions on failure — caller must handle them.
    """
    client = get_client()
    if client is None:
        raise ValueError("No API key configured. Add a key in ⚙️ Settings.")

    st.session_state[_STOP_KEY] = False
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            if st.session_state.get(_STOP_KEY, False):
                break
            yield text


def format_api_error(exc: Exception) -> str:
    """Return a user-friendly error string without traceback."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "Invalid API key — check your key in ⚙️ Settings."
    if isinstance(exc, anthropic.RateLimitError):
        return "API rate limit reached. Wait a moment, then try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Connection failed — check your internet connection."
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}: {exc.message}"
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Unexpected error ({type(exc).__name__}). Try again."
