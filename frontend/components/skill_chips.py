"""
Skill chips component for matched, missing, and bonus skills.
Always escapes user input and pairs color with clear icons.
"""
import html
from typing import List, Optional


def render_skill_chips(
    skills: Optional[List[str]] = None,
    chip_type: str = "matched",
    max_display: Optional[int] = None,
    matched: Optional[List[str]] = None,
    missing: Optional[List[str]] = None,
    bonus: Optional[List[str]] = None,
    **kwargs
) -> str:
    """
    Renders HTML skill chips with escaping and accessibility icons.
    chip_type: 'matched' (✓), 'missing' (✗), 'bonus' (★), or 'neutral' (•)
    Also supports keyword arguments: matched, missing, bonus for multi-group rendering.
    """
    if matched is not None or missing is not None or bonus is not None:
        parts = []
        if matched:
            parts.append(render_skill_chips(skills=matched, chip_type="matched", max_display=max_display))
        if missing:
            parts.append(render_skill_chips(skills=missing, chip_type="missing", max_display=max_display))
        if bonus:
            parts.append(render_skill_chips(skills=bonus, chip_type="bonus", max_display=max_display))
        return "".join(parts)

    if not skills:
        return ""

    displayed = skills[:max_display] if max_display else skills
    remaining = len(skills) - len(displayed)

    if chip_type == "matched":
        cls = "skill-chip skill-chip-matched"
        icon = "✓ "
    elif chip_type == "missing":
        cls = "skill-chip skill-chip-missing"
        icon = "✗ "
    elif chip_type == "bonus":
        cls = "skill-chip skill-chip-bonus"
        icon = "★ "
    else:
        cls = "skill-chip"
        icon = "• "

    chips_html = "".join([
        f'<span class="{cls}">{icon}{html.escape(str(s).strip())}</span>'
        for s in displayed if str(s).strip()
    ])

    if remaining > 0:
        chips_html += f'<span class="skill-chip" style="color: var(--color-muted);">+{remaining} more</span>'

    return f'<div style="display: flex; flex-wrap: wrap; gap: 4px; margin: 4px 0;">{chips_html}</div>'
