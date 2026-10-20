"""
Accessible circular SVG score ring component.
Always pairs color with textual verdict and icon for accessibility.
"""
import html


def render_score_ring(score: float, max_score: float = 100.0, size: int = 140, **kwargs) -> str:
    """
    Renders an accessible SVG score ring with high contrast and explicit text label.
    """
    score = max(0.0, min(float(score), max_score))
    radius = 44
    circ = 2 * 3.14159265 * radius  # ~276.46
    offset = circ * (1.0 - (score / max_score))

    if score >= 80:
        bar_color = "#10b981"  # Emerald
        verdict = "✓ Good Match"
        pill_class = "badge-tag badge-tag-green"
    elif score >= 60:
        bar_color = "#2563eb"  # Royal blue
        verdict = "ℹ Competitive"
        pill_class = "badge-tag badge-tag-blue"
    elif score >= 40:
        bar_color = "#f59e0b"  # Amber
        verdict = "⚠️ Needs Work"
        pill_class = "badge-tag badge-tag-amber"
    else:
        bar_color = "#ef4444"  # Red
        verdict = "✗ Low Alignment"
        pill_class = "badge-tag badge-tag-red"

    svg_markup = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 4px;" role="img" aria-label="ATS Score {int(round(score))} out of 100, verdict: {html.escape(verdict)}">
        <svg width="{size}" height="{size}" viewBox="0 0 110 110" aria-hidden="true">
            <circle cx="55" cy="55" r="{radius}" fill="none" stroke="var(--color-border, #e2e8f0)" stroke-width="9" />
            <circle cx="55" cy="55" r="{radius}" fill="none" stroke="{bar_color}" stroke-width="9"
                    stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
                    stroke-linecap="round" transform="rotate(-90 55 55)" />
            <text x="55" y="52" text-anchor="middle" font-family="var(--font-sans, sans-serif)" font-size="26" font-weight="800" fill="var(--color-text, #0f172a)">{int(round(score))}</text>
            <text x="55" y="68" text-anchor="middle" font-family="var(--font-sans, sans-serif)" font-size="11" font-weight="600" fill="var(--color-muted, #64748b)">/100</text>
        </svg>
        <div style="margin-top: 6px;">
            <span class="{pill_class}" style="font-weight: 700;">{html.escape(verdict)}</span>
        </div>
    </div>
    """
    return svg_markup
