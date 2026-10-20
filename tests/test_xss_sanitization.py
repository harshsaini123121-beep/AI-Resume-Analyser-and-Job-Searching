"""
Security regression test verifying that no user-controlled string reaches HTML unescaped.
Tests malicious filenames, script injections, and HTML tag sanitization across all UI components.
"""
from frontend.components.score_ring import render_score_ring
from frontend.components.skill_chips import render_skill_chips
from frontend.components.evidence_card import render_evidence_card
from frontend.components.stat_card import render_stat_card
from frontend.components.empty_state import render_empty_state
from frontend.components.error_banner import render_error_banner


def test_malicious_filename_and_script_injection_sanitization():
    """Verifies that malicious filenames and script tags are strictly escaped in HTML components."""
    malicious_filename = "<script>alert('pwned')</script>.pdf"
    malicious_skill = "<img src=x onerror=alert(1)>"
    malicious_proof = "<b>bold</b> <svg onload=alert(2)>"

    # 1. Skill Chips Component
    chips_html = render_skill_chips([malicious_skill, "python"], chip_type="matched")
    assert "<img" not in chips_html
    assert "&lt;img src=x onerror=alert(1)&gt;" in chips_html

    # 2. Evidence Card Component
    ev_item = {
        "job_requirement": malicious_filename,
        "resume_evidence": malicious_proof,
        "status": "Strong Evidence"
    }
    card_html = render_evidence_card(ev_item)
    assert "<script>" not in card_html
    assert "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;.pdf" in card_html or "&lt;script&gt;alert('pwned')&lt;/script&gt;.pdf" in card_html
    assert "<svg onload=" not in card_html
    assert "&lt;svg onload=alert(2)&gt;" in card_html

    # 3. Stat Card Component
    stat_html = render_stat_card(
        value="88%",
        label=malicious_filename,
        explanation="<script>evil()</script>"
    )
    assert "<script>" not in stat_html
    assert "&lt;script&gt;" in stat_html

    # 4. Empty State Component
    empty_html = render_empty_state(
        title=malicious_filename,
        description="<iframe src='malicious.com'></iframe>",
        action_note="<script>evil()</script>"
    )
    assert "<iframe" not in empty_html
    assert "<script>" not in empty_html
    assert "&lt;iframe" in empty_html

    # 5. Error Banner Component
    banner_html = render_error_banner(
        title=malicious_filename,
        message="<b onmouseover=alert(1)>error</b>",
        recovery_tip="<a href='javascript:alert(1)'>click</a>"
    )
    assert "<b onmouseover" not in banner_html
    assert "<a href=" not in banner_html
    assert "&lt;b onmouseover" in banner_html
