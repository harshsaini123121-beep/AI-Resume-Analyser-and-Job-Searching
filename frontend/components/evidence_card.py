"""
Sentence-level evidence mapping card component.
Renders each job requirement alongside the candidate resume proof.
"""
import html
from typing import Dict, Any, Optional


def render_evidence_card(
    evidence_item: Optional[Dict[str, Any]] = None,
    requirement: Optional[str] = None,
    candidate_proof: Optional[str] = None,
    status: Optional[str] = None
) -> str:
    """
    Renders a side-by-side evidence comparison card.
    Accepts either an evidence dictionary or individual requirement/proof kwargs.
    """
    if isinstance(evidence_item, dict):
        job_req = html.escape(str(evidence_item.get("job_requirement", "")))
        resume_proof = html.escape(str(evidence_item.get("resume_evidence", "")))
        item_status = evidence_item.get("status", "Moderate Alignment")
    else:
        job_req = html.escape(str(requirement or ""))
        resume_proof = html.escape(str(candidate_proof or ""))
        item_status = status or "Moderate Alignment"

    is_strong = item_status == "Strong Evidence"
    card_cls = "evidence-card strong" if is_strong else "evidence-card"
    status_pill = (
        '<span class="badge-tag" style="background-color: var(--color-success-bg); color: var(--color-success-text); border: 1px solid var(--color-success-border);">✓ Strong Match</span>'
        if is_strong
        else '<span class="badge-tag" style="background-color: var(--color-warning-bg); color: var(--color-warning-text); border: 1px solid var(--color-warning-border);">ℹ Inferred Match</span>'
    )

    markup = f"""
    <div class="{card_cls}">
        <div class="evidence-header">
            <span class="evidence-label">Requirement</span>
            {status_pill}
        </div>
        <div class="evidence-req-text">"{job_req}"</div>
        <div class="evidence-header" style="margin-top: 6px;">
            <span class="evidence-label" style="color: var(--color-primary);">Resume Proof</span>
        </div>
        <div class="evidence-proof-box">"{resume_proof}"</div>
    </div>
    """
    return markup
