"""ui.advisor — AI Advisor package."""
from ui.advisor.settings import (
    init_advisor_state,
    is_advisor_enabled,
    get_active_claude_md,
    build_system_prompt,
)
from ui.advisor.panel import (
    init_advisor_rail_state,
    is_rail_open,
    get_column_ratio,
    render_advisor_panel,
)
from ui.advisor.client import (
    get_client,
    stream_response,
    format_api_error,
)
from ui.advisor.context import assemble_context

__all__ = [
    "init_advisor_state",
    "is_advisor_enabled",
    "get_active_claude_md",
    "build_system_prompt",
    "init_advisor_rail_state",
    "is_rail_open",
    "get_column_ratio",
    "render_advisor_panel",
    "get_client",
    "stream_response",
    "format_api_error",
    "assemble_context",
]
