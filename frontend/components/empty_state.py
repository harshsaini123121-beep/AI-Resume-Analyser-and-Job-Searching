"""
Designed empty state component.
Provides guidance and next steps when no data is loaded.
"""
import html
from typing import Optional


def render_empty_state(
    title: str,
    description: str,
    icon: str = "📄",
    action_note: Optional[str] = None,
    action_label: Optional[str] = None,
    **kwargs
) -> str:
    """
    Renders a designed empty state placeholder.
    """
    safe_title = html.escape(title)
    safe_desc = html.escape(description)
    note_text = action_note or action_label or kwargs.get("action_text")
    note_html = (
        f'<div style="margin-top: 12px; font-size: var(--font-xs); color: var(--color-primary); font-weight: 600;">{html.escape(note_text)}</div>'
        if note_text
        else ""
    )

    markup = f"""
    <div class="empty-state-box">
        <div class="empty-state-icon" aria-hidden="true">{icon}</div>
        <div class="empty-state-title">{safe_title}</div>
        <div class="empty-state-desc">{safe_desc}</div>
        {note_html}
    </div>
    """
    return markup
