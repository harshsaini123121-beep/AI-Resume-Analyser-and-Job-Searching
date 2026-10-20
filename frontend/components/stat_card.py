"""
Stat / Metric card component with one-line explanation.
"""
import html


from typing import Optional


def render_stat_card(
    value: Optional[str] = None,
    label: Optional[str] = None,
    explanation: Optional[str] = None,
    icon: str = "📊",
    color_theme: str = "blue",
    title: Optional[str] = None,
    border_color: Optional[str] = None,
    **kwargs
) -> str:
    """
    Renders an accessible stat card with a metric, label, and one-line explanation.
    Supports title or label, border_color or color_theme, and arbitrary kwargs.
    """
    actual_label = title or label or kwargs.get("name") or ""
    actual_val = value or kwargs.get("stat") or ""
    actual_expl = explanation or kwargs.get("desc") or kwargs.get("description") or ""

    safe_val = html.escape(str(actual_val))
    safe_label = html.escape(str(actual_label))
    safe_expl = html.escape(str(actual_expl))

    border_style = f"border-left: 3px solid {html.escape(border_color)};" if border_color else ""
    icon_cls = f"stat-icon stat-icon-{html.escape(color_theme)}"

    markup = f"""
    <div class="stat-card" style="{border_style}" role="region" aria-label="{safe_label}: {safe_val}">
        <div class="{icon_cls}" aria-hidden="true">{icon}</div>
        <div>
            <div class="stat-val">{safe_val}</div>
            <div class="stat-label">{safe_label}</div>
            <div class="stat-expl">{safe_expl}</div>
        </div>
    </div>
    """
    return markup
