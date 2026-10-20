"""
Accessible error banner component with recovery recommendations.
"""
import html


def render_error_banner(
    title: str,
    message: str,
    recovery_tip: str = ""
) -> str:
    """
    Renders an accessible error banner.
    """
    safe_title = html.escape(title)
    safe_msg = html.escape(message)
    safe_tip = html.escape(recovery_tip)

    tip_html = f'<div class="error-desc">💡 <b>Tip:</b> {safe_tip}</div>' if safe_tip else ""

    markup = f"""
    <div class="error-banner" role="alert">
        <div class="error-title">⚠️ {safe_title}</div>
        <div class="error-desc">{safe_msg}</div>
        {tip_html}
    </div>
    """
    return markup
