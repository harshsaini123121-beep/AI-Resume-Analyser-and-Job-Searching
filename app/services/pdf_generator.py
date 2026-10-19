"""
Branded PDF ATS Audit Report Generator using fpdf2.
Compiles candidate scores, category breakdown, identified issues, and skills into an executive PDF.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from fpdf import FPDF


class ResumeAuditPDF(FPDF):
    def header(self):
        self.set_fill_color(30, 27, 75)  # Dark Indigo
        self.rect(0, 0, 210, 22, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(12, 6)
        self.cell(0, 10, "AI RESUME ANALYSER & ATS AUDIT REPORT", ln=False)
        self.set_font("Helvetica", size=9)
        self.set_xy(145, 6)
        self.cell(0, 10, datetime.now().strftime("%B %d, %Y"), ln=True, align="R")
        self.set_text_color(30, 41, 59)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | AI Resume Analyser & Job Matching Suite", align="C")


def generate_pdf_report(
    candidate_name: str,
    email: Optional[str],
    phone: Optional[str],
    ats_score: int,
    ats_grade: str,
    category_breakdown: Dict[str, Dict[str, Any]],
    strengths: List[str],
    issues: List[str],
    suggestions: List[str],
    skills: List[str]
) -> bytes:
    """
    Builds a professional 2-page PDF ATS Audit Report and returns binary bytes.
    """
    pdf = ResumeAuditPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    epw = pdf.epw

    # 1. Candidate Info Card
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, f"Candidate: {candidate_name or 'Applicant'}", ln=True)

    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(71, 85, 105)
    contact_line = f"Email: {email or 'Not provided'}   |   Phone: {phone or 'Not provided'}"
    pdf.cell(0, 6, contact_line, ln=True)
    pdf.ln(4)

    # 2. Score Banner
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(pdf.get_x(), pdf.get_y(), epw, 24, "F")
    pdf.set_xy(pdf.get_x() + 4, pdf.get_y() + 4)

    pdf.set_font("Helvetica", "B", 20)
    if ats_score >= 80:
        pdf.set_text_color(16, 185, 129)  # Green
    elif ats_score >= 60:
        pdf.set_text_color(59, 130, 246)  # Blue
    else:
        pdf.set_text_color(239, 68, 68)   # Red

    pdf.cell(50, 14, f"ATS Score: {ats_score}/100", ln=False)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 14, f"Grade: {ats_grade}", ln=True, align="R")
    pdf.ln(8)

    # 3. Category Breakdown Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "ATS Dimension Performance Breakdown", ln=True)

    # Table Header
    pdf.set_fill_color(226, 232, 240)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(65, 7, "Dimension", border=1, fill=True)
    pdf.cell(30, 7, "Score", border=1, fill=True, align="C")
    pdf.cell(epw - 95, 7, "Inspection Details", border=1, fill=True)
    pdf.ln()

    # Table Rows
    pdf.set_font("Helvetica", size=8.5)
    for cat_name, data in category_breakdown.items():
        s = data.get("score", 0)
        m = data.get("max_score", 10)
        details = str(data.get("details", ""))
        # Clean non-ascii if any
        details_clean = details.encode("latin-1", "replace").decode("latin-1")

        pdf.cell(65, 6, cat_name, border=1)
        pdf.cell(30, 6, f"{s} / {m} pts", border=1, align="C")
        pdf.cell(epw - 95, 6, details_clean[:55], border=1)
        pdf.ln()

    pdf.ln(6)

    # 4. Detected Skills
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, f"Detected Technical & Professional Skills ({len(skills)})", ln=True)

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(71, 85, 105)
    skills_text = ", ".join(skills) if skills else "No standard technical skills detected."
    pdf.multi_cell(epw, 5, skills_text[:600])
    pdf.ln(6)

    # 5. Identified Strengths
    if strengths:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(22, 101, 52)
        pdf.cell(0, 8, "Key Resume Strengths", ln=True)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(51, 65, 85)
        for st_item in strengths[:5]:
            clean_item = st_item.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(epw, 5, f"+ {clean_item}")
        pdf.ln(4)

    # 6. Critical Issues & Suggestions (Page 2 if needed)
    if issues or suggestions:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(185, 28, 28)
        pdf.cell(0, 8, "ATS Compliance Issues & Fixes", ln=True)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(51, 65, 85)
        for iss in issues[:5]:
            clean_iss = iss.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(epw, 5, f"- Issue: {clean_iss}")
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 8, "Strategic Next Steps & Recommendations", ln=True)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(51, 65, 85)
        for sug in suggestions[:5]:
            clean_sug = sug.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(epw, 5, f"> {clean_sug}")

    return bytes(pdf.output())
