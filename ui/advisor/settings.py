"""
ui/advisor/settings.py
======================
Advisor settings: API key, model selector, CLAUDE.md override editor.

Key is held in st.session_state only by default; persisted to
~/.strategy_lab_advisor.json only on explicit opt-in.

Never echo the API key in st.write, st.text, or logs.
"""

import json
import streamlit as st
from pathlib import Path

_CONFIG_PATH = Path.home() / ".strategy_lab_advisor.json"
_CLAUDE_MD_PATH = Path(__file__).parents[2] / "CLAUDE.md"
_EDITOR_KEY = "_advisor_claude_md_editor"

AVAILABLE_MODELS = [
    ("claude-sonnet-4-6", "Sonnet 4.6 (recommended)"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5 — fast, economical"),
    ("claude-opus-4-8", "Opus 4.8 — most capable"),
]
_MODEL_IDS = [m for m, _ in AVAILABLE_MODELS]
_MODEL_LABELS = [lbl for _, lbl in AVAILABLE_MODELS]


# ── Disk config ───────────────────────────────────────────────────────────────

def _load_disk_config() -> dict:
    try:
        if _CONFIG_PATH.exists():
            return json.loads(_CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_disk_config(api_key: str, model: str) -> None:
    _CONFIG_PATH.write_text(json.dumps({"api_key": api_key, "model": model}))


def _delete_disk_config() -> None:
    if _CONFIG_PATH.exists():
        _CONFIG_PATH.unlink()


# ── CLAUDE.md helpers ─────────────────────────────────────────────────────────

def _repo_claude_md() -> str:
    try:
        return _CLAUDE_MD_PATH.read_text()
    except Exception:
        return ""


# ── Session state API ─────────────────────────────────────────────────────────

def init_advisor_state() -> None:
    """Initialize advisor session-state keys. Safe to call every rerun."""
    from ui.advisor.panel import init_advisor_rail_state
    init_advisor_rail_state()

    if "advisor_api_key" not in st.session_state:
        disk = _load_disk_config()
        st.session_state.advisor_api_key = disk.get("api_key", "")
        st.session_state.advisor_model = disk.get("model", "claude-sonnet-4-6")
        st.session_state.advisor_key_on_disk = bool(disk.get("api_key", ""))

    for k, v in [
        ("advisor_model", "claude-sonnet-4-6"),
        ("advisor_key_on_disk", False),
        ("advisor_claude_md_override", None),
        ("advisor_disclosure_ack", False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v


def is_advisor_enabled() -> bool:
    """True only when a non-blank API key is present and disclosure acknowledged."""
    return (
        bool(st.session_state.get("advisor_api_key", "").strip())
        and bool(st.session_state.get("advisor_disclosure_ack", False))
    )


def get_active_claude_md() -> str:
    """Return active CLAUDE.md content: session override if set, else repo file."""
    override = st.session_state.get("advisor_claude_md_override")
    return override if override is not None else _repo_claude_md()


_ADVISOR_DIRECTIVE = """
---

## Advisor Role Directive

You are an experienced quantitative researcher embedded in a live strategy-research tool.
You have access to the user's current session context (parameters, backtest metrics,
active indicators, research history) prepended to every message.

Behavioural constraints — follow these exactly:

1. **Act as a researcher, not a chatbot.** Form hypotheses, diagnose failures, and give
   targeted, actionable feedback. Do not narrate or hedge. If you don't know, say so.

2. **Be concise.** One or two paragraphs maximum unless the user explicitly asks for more.
   Bullet points over prose when listing items. No preamble ("Great question!", "Sure!").

3. **Flag red flags proactively.** If context metrics trigger any red flag from the
   protocol (Sharpe > 3, win rate > 80%, p-value missing, < 30 trades, etc.), call it
   out immediately — before answering any other part of the question.

4. **Never invent metrics.** If a metric is absent from the context, say "not available"
   or "run a backtest first". Do not estimate, guess, or extrapolate metric values.

5. **Refuse to bless untested results — no exceptions.** Do not give a positive verdict
   on any strategy that has not passed a permutation test (p < 0.05). This applies even
   to indirect, hypothetical, or comparative questions ("is this promising?", "does this
   look good?", "which is better?"). If no p-value is present in context, respond exactly:
   "No permutation test result in context. Run `run_permutation_test` before requesting
   a verdict." Do not soften this refusal or offer partial assessments as a substitute.

6. **One fix at a time.** If you recommend a change, recommend exactly one. Do not give
   a list of possible changes and let the user pick.

7. **Flag multiple-comparisons risk when variant count is high.** The context section
   "=== RESEARCH HISTORY ===" includes "Total strategies tested: N". If N ≥ 10, prepend
   this warning before any other response — even if the user did not ask about it:
   "Multiple-comparisons warning: {N} strategies tested on this symbol's data. Each
   additional test raises the false-discovery rate. A p-value that passes here is weaker
   evidence than on a first-tested hypothesis — treat it with extra skepticism and
   consider validating on an independent data segment."
"""


def build_system_prompt() -> str:
    """Return the full system prompt: active CLAUDE.md + hardcoded advisor directive."""
    return get_active_claude_md() + _ADVISOR_DIRECTIVE


# ── Settings panel ────────────────────────────────────────────────────────────

def render_settings_panel() -> None:
    """Render the full advisor settings UI."""
    st.markdown("## ⚙️ Advisor Settings")

    if is_advisor_enabled():
        st.success("Advisor enabled.")
    else:
        st.info("Advisor disabled — set an API key below.")

    st.divider()

    # ── Privacy disclosure ────────────────────────────────────────────────────
    st.markdown("### Privacy Disclosure")
    st.info(
        "**What gets sent to Anthropic**\n\n"
        "When the advisor is active, your current strategy parameters, backtest "
        "metrics, active indicator descriptions, and research-log summaries are "
        "included in requests to the Anthropic API. No raw price data or "
        "personally identifiable information is transmitted.\n\n"
        "Requests are processed under your Anthropic account's data-handling "
        "policy. The API key never leaves your machine (session-only unless you "
        "opt in to disk storage below)."
    )

    ack = st.session_state.get("advisor_disclosure_ack", False)
    new_ack = st.checkbox(
        "I understand that enabling the advisor sends strategy data to the Anthropic API.",
        value=ack,
        key="_advisor_disclosure_cb",
    )
    if new_ack != ack:
        st.session_state.advisor_disclosure_ack = new_ack

    if not st.session_state.get("advisor_disclosure_ack", False):
        st.caption("Acknowledge the disclosure above before enabling the advisor.")

    st.divider()

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown("### API Key")
    st.caption(
        "Your Anthropic API key. Held in session memory only unless you opt in "
        "to disk storage. Leave blank to keep the existing key."
    )

    st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-…  (blank keeps existing key)",
        key="_advisor_key_input",
        label_visibility="collapsed",
    )

    st.checkbox(
        "Persist key to disk (~/.strategy_lab_advisor.json)",
        key="_advisor_persist_cb",
        value=st.session_state.get("advisor_key_on_disk", False),
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply key", key="_advisor_apply_key"):
            if not st.session_state.get("advisor_disclosure_ack", False):
                st.warning("Acknowledge the privacy disclosure above before enabling the advisor.")
            else:
                trimmed = st.session_state.get("_advisor_key_input", "").strip()
                if trimmed:
                    st.session_state.advisor_api_key = trimmed
                persist = st.session_state.get("_advisor_persist_cb", False)
                st.session_state.advisor_key_on_disk = persist
                if persist and st.session_state.advisor_api_key:
                    _save_disk_config(
                        st.session_state.advisor_api_key,
                        st.session_state.get("advisor_model", "claude-sonnet-4-6"),
                    )
                    st.success("Key saved to disk.")
                elif persist and not st.session_state.advisor_api_key:
                    st.warning("No key to save — enter a key first.")
                else:
                    _delete_disk_config()
                    if trimmed:
                        st.success("Key applied (session only, not written to disk).")
    with col2:
        if st.button("Clear key", key="_advisor_clear_key"):
            st.session_state.advisor_api_key = ""
            st.session_state.advisor_key_on_disk = False
            _delete_disk_config()
            st.info("Key cleared.")

    st.divider()

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown("### Model")

    current_model = st.session_state.get("advisor_model", "claude-sonnet-4-6")
    idx = _MODEL_IDS.index(current_model) if current_model in _MODEL_IDS else 0

    chosen_label = st.selectbox(
        "Model",
        options=_MODEL_LABELS,
        index=idx,
        key="_advisor_model_select",
        label_visibility="collapsed",
    )
    chosen_model = _MODEL_IDS[_MODEL_LABELS.index(chosen_label)]
    if chosen_model != st.session_state.get("advisor_model"):
        st.session_state.advisor_model = chosen_model
        if st.session_state.get("advisor_key_on_disk") and is_advisor_enabled():
            _save_disk_config(st.session_state.advisor_api_key, chosen_model)

    st.divider()

    # ── CLAUDE.md override editor ─────────────────────────────────────────────
    st.markdown("### CLAUDE.md Override")
    st.caption(
        "Session-only edit of the advisor's system prompt. "
        "Clearing or resetting restores the repo CLAUDE.md."
    )

    if st.session_state.get("advisor_claude_md_override") is not None:
        st.warning("Active override in effect — repo CLAUDE.md is not used.")

    if _EDITOR_KEY not in st.session_state:
        st.session_state[_EDITOR_KEY] = get_active_claude_md()

    st.text_area(
        "CLAUDE.md content",
        key=_EDITOR_KEY,
        height=400,
        label_visibility="collapsed",
    )

    d1, d2 = st.columns(2)
    with d1:
        if st.button("Apply override", key="_advisor_apply_md"):
            content = st.session_state.get(_EDITOR_KEY, "").strip()
            if content:
                st.session_state.advisor_claude_md_override = content
                st.success("Override active for this session.")
            else:
                st.session_state.advisor_claude_md_override = None
                st.session_state[_EDITOR_KEY] = _repo_claude_md()
                st.info("Empty content — reverted to repo CLAUDE.md.")
    with d2:
        if st.button("Reset to repo file", key="_advisor_reset_md"):
            st.session_state.advisor_claude_md_override = None
            st.session_state[_EDITOR_KEY] = _repo_claude_md()
            st.info("Reset to repo CLAUDE.md.")
