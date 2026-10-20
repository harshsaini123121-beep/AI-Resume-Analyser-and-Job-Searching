"""
Stat / Metric card component with one-line explanation.
"""
import html


def render_stat_card(
    value: str,
    label: str,
    explanation: str,
    icon: str = "📊",
    color_theme: str = "blue"
) -> str:
    """
    Renders an accessible stat card with a metric, label, and one-line explanation.
    """
    safe_val = html.escape(str(value))
    safe_label = html.escape(str(label))
    safe_expl = html.escape(str(explanation))

    icon_cls = f"stat-icon stat-icon-{color_theme}"

    markup = f"""
    <div class="stat-card" role="region" aria-label="{safe_label}: {safe_val}">
        <div class="{icon_cls}" aria-hidden="true">{icon}</div>
        <div>
            <div class="stat-val">{safe_val}</div>
            <div class="stat-label">{safe_label}</div>
            <div class="stat-expl">{safe_expl}</div>
        </div>
    </div>
    """
    return markup
