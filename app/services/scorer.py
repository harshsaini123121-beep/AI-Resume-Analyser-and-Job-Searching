"""
ATS (Applicant Tracking System) Scoring Engine for Resumes.
Evaluates resume quality on a comprehensive 100-point rubric.
"""

import re
from typing import Dict, List, Any


# Standard action verbs favored by technical and corporate recruiters
ACTION_VERBS = {
    "accelerated", "achieved", "architected", "authored", "automated", "boosted",
    "built", "championed", "collaborated", "conceived", "constructed", "created",
    "decreased", "delivered", "deployed", "designed", "developed", "devised",
    "directed", "drove", "eliminated", "engineered", "enhanced", "established",
    "executed", "expanded", "expedited", "fine-tuned", "formulated", "founded",
    "generated", "guided", "implemented", "improved", "increased", "initiated",
    "innovated", "instituted", "integrated", "launched", "lead", "led", "leveraged",
    "managed", "mentored", "migrated", "modernized", "negotiated", "optimized",
    "orchestrated", "overhauled", "pioneered", "produced", "reduced", "refactored",
    "resolved", "restructured", "revamped", "saved", "scaled", "solved",
    "spearheaded", "standardized", "streamlined", "strengthened", "surpassed",
    "trained", "transformed", "unified", "upgraded", "validated"
}

# Weak passive phrases to discourage
WEAK_PHRASES = [
    "responsible for", "duties included", "helped with", "assisted in",
    "worked on", "handled", "participated in"
]


def calculate_ats_score(resume_text: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes an objective ATS score (0-100) and produces itemized category breakdowns,
    identified issues, detected strengths, and actionable suggestions.
    """
    category_scores: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    strengths: List[str] = []
    suggestions: List[str] = []

    # -------------------------------------------------------------
    # 1. Contact Information (Max: 10 pts)
    # -------------------------------------------------------------
    contact_pts = 0
    email = extracted_data.get("email")
    phone = extracted_data.get("phone")
    links = extracted_data.get("links", {})
    linkedin = links.get("linkedin")
    github = links.get("github")
    portfolio = links.get("portfolio")

    if email:
        contact_pts += 4
    else:
        issues.append("Missing email address in resume header.")
        suggestions.append("Add a professional email address at the very top.")

    if phone:
        contact_pts += 3
    else:
        issues.append("Missing telephone / phone contact number.")
        suggestions.append("Include an accessible phone number with country/area code.")

    if linkedin or github or portfolio:
        contact_pts += 3
        strengths.append("Professional profile link (LinkedIn/GitHub/Portfolio) is present.")
    else:
        issues.append("No professional profile links (LinkedIn, GitHub, or Portfolio) detected.")
        suggestions.append("Add a customized LinkedIn or GitHub URL to showcase your work.")

    category_scores["Contact Info"] = {
        "score": contact_pts,
        "max_score": 10,
        "details": f"Email: {'✅' if email else '❌'}, Phone: {'✅' if phone else '❌'}, Profiles: {'✅' if (linkedin or github or portfolio) else '❌'}"
    }

    # -------------------------------------------------------------
    # 2. Standard Resume Sections (Max: 20 pts)
    # -------------------------------------------------------------
    section_pts = 0
    detected_sections = extracted_data.get("sections_detected", [])
    expected_sections = [
        ("summary", "Professional Summary / Objective", 4),
        ("skills", "Core Skills & Technologies", 4),
        ("experience", "Work Experience", 4),
        ("education", "Education & Academics", 4),
        ("projects", "Projects / Portfolio / Certifications", 4)
    ]

    missing_sections = []
    for key, label, pts in expected_sections:
        # Check if project or certification is present
        if key == "projects":
            if "projects" in detected_sections or "certifications" in detected_sections:
                section_pts += pts
            else:
                missing_sections.append(label)
        else:
            if key in detected_sections:
                section_pts += pts
            else:
                missing_sections.append(label)

    if missing_sections:
        issues.append(f"Missing recommended resume sections: {', '.join(missing_sections)}.")
        suggestions.append(f"Organize your resume with explicit standard headers: {', '.join(missing_sections)}.")
    else:
        strengths.append("Complete standard section structure (Summary, Skills, Experience, Education, Projects).")

    category_scores["Sections Structure"] = {
        "score": section_pts,
        "max_score": 20,
        "details": f"Found {len(detected_sections)} standard sections: {', '.join(detected_sections)}"
    }

    # -------------------------------------------------------------
    # 3. Skills Count & Density (Max: 15 pts)
    # -------------------------------------------------------------
    skills = extracted_data.get("skills", [])
    skill_count = len(skills)
    skill_pts = 0

    if skill_count >= 12:
        skill_pts = 15
        strengths.append(f"Strong technical skill breadth: {skill_count} relevant skills recognized.")
    elif skill_count >= 8:
        skill_pts = 12
        strengths.append(f"Good skills foundation: {skill_count} skills recognized.")
    elif skill_count >= 5:
        skill_pts = 8
        suggestions.append("Enrich your technical stack with at least 8-12 industry-standard skills and tools.")
    elif skill_count >= 1:
        skill_pts = 4
        issues.append(f"Very few skills identified ({skill_count} skills). ATS scanners might filter your resume out.")
        suggestions.append("Add a dedicated 'Skills' section listing languages, frameworks, databases, and tools.")
    else:
        skill_pts = 0
        issues.append("No recognizable technical or soft skills were detected.")
        suggestions.append("Create a prominent Skills section with standard industry keywords.")

    category_scores["Skills Density"] = {
        "score": skill_pts,
        "max_score": 15,
        "details": f"{skill_count} skills detected"
    }

    # -------------------------------------------------------------
    # 4. Quantified Achievements & Metrics (Max: 15 pts)
    # Look for metrics: percentages (%), currencies ($/€/£/₹), multipliers (2x, 10x), scale (100k, 2M, 500+ users, 45 ms)
    # Strictly require metric units, percentages, currency symbols, or multipliers (prevents matching bare years/numbers)
    metric_pattern = (
        r"(?:\d+(?:\.\d+)?%"
        r"|[\$€£₹]\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:k|m|b|million|billion))?"
        r"|\b\d+(?:\.\d+)?x\b"
        r"|\b\d+(?:,\d+)*(?:\.\d+)?\+?\s*(?:k|m|million|billion|users|clients|customers|requests|queries|qps|rps|tps|transactions|dau|mau|employees|engineers|members|tickets|hours|days|weeks|months|years|ms|seconds|tb|gb|pb|pts|points)\b"
        r"|\b\d{2,}\+\b)"
    )
    quantified_matches = re.findall(metric_pattern, resume_text, re.IGNORECASE)
    quant_count = len(quantified_matches)
    quant_pts = 0

    if quant_count >= 5:
        quant_pts = 15
        strengths.append(f"Excellent quantifiable metrics: {quant_count}+ data points demonstrating business impact.")
    elif quant_count >= 3:
        quant_pts = 11
        strengths.append("Measurable outcomes included in bullet points.")
    elif quant_count >= 1:
        quant_pts = 6
        suggestions.append("Add more quantitative metrics (e.g., '% improved', '₹ saved / cost reduced', 'number of users served').")
    else:
        quant_pts = 0
        issues.append("No quantified achievements found (percentages, Rupee (₹) amounts, scale).")
        suggestions.append("Quantify your impact using the XYZ formula: 'Accomplished [X] as measured by [Y], by doing [Z]'.")

    category_scores["Quantified Impact"] = {
        "score": quant_pts,
        "max_score": 15,
        "details": f"{quant_count} quantified data points identified"
    }

    # -------------------------------------------------------------
    # 5. Action Verbs (Max: 10 pts)
    # -------------------------------------------------------------
    words_in_text = set(re.findall(r"\b[a-z]{3,}\b", resume_text.lower()))
    found_action_verbs = sorted(list(words_in_text.intersection(ACTION_VERBS)))
    action_count = len(found_action_verbs)
    action_pts = 0

    if action_count >= 8:
        action_pts = 10
        strengths.append(f"Strong active voice: {action_count} dynamic action verbs utilized.")
    elif action_count >= 5:
        action_pts = 7
    elif action_count >= 2:
        action_pts = 4
        suggestions.append("Begin accomplishment bullet points with dynamic action verbs (e.g. Architected, Spearheaded, Accelerated).")
    else:
        action_pts = 1
        issues.append("Lacks strong action verbs; language may sound passive.")
        suggestions.append("Replace passive phrases with strong verbs: engineered, deployed, reduced, led, optimized.")

    category_scores["Action Verbs"] = {
        "score": action_pts,
        "max_score": 10,
        "details": f"{action_count} distinct action verbs: {', '.join(found_action_verbs[:5])}..." if found_action_verbs else "0 verbs"
    }

    # -------------------------------------------------------------
    # 6. Length & Word Count (Max: 10 pts)
    # -------------------------------------------------------------
    word_count = extracted_data.get("word_count", len(resume_text.split()))
    length_pts = 0

    if 350 <= word_count <= 850:
        length_pts = 10
        strengths.append(f"Ideal document length ({word_count} words; fits 1-2 clean pages).")
    elif 250 <= word_count < 350:
        length_pts = 7
        suggestions.append("Resume is somewhat brief. Elaborate on project specifics and technical responsibilities.")
    elif 850 < word_count <= 1200:
        length_pts = 7
        suggestions.append("Resume is on the longer side (over 850 words). Consider trimming to keep it focused.")
    elif word_count > 1200:
        length_pts = 4
        issues.append(f"Resume is excessively long ({word_count} words). Recruiters spend only 6-8 seconds per resume.")
        suggestions.append("Condense resume to under 800 words, prioritizing the most recent and impactful 3-5 years.")
    else:
        length_pts = 3
        issues.append(f"Resume is too short ({word_count} words). It may be missing vital technical details.")

    category_scores["Word Count & Length"] = {
        "score": length_pts,
        "max_score": 10,
        "details": f"{word_count} words"
    }

    # -------------------------------------------------------------
    # 7. Formatting & Bullet Point Structure (Max: 10 pts)
    # -------------------------------------------------------------
    bullet_count = extracted_data.get("bullet_count", 0)
    format_pts = 0

    if bullet_count >= 6:
        format_pts += 6
        strengths.append("Well-structured with clear bullet points for rapid recruiter scanning.")
    elif bullet_count >= 2:
        format_pts += 4
        suggestions.append("Convert paragraph blocks into concise, bulleted accomplishment statements.")
    else:
        format_pts += 1
        issues.append("Few or no bullet points detected. Dense text paragraphs harm recruiter readability.")
        suggestions.append("Use bullet points for all work experience and project descriptions.")

    # Check for excessive capitalization yelling
    all_caps_words = re.findall(r"\b[A-Z]{4,}\b", resume_text)
    legit_acronyms = {"HTML", "REST", "JSON", "HTTP", "HTTPS", "SaaS", "IEEE", "STEM", "BCA", "MCA", "DBMS"}
    noisy_caps = [w for w in all_caps_words if w not in legit_acronyms]
    if len(noisy_caps) < 8:
        format_pts += 4
    else:
        format_pts += 2
        suggestions.append("Avoid excessive ALL-CAPS text outside of standard acronyms and section titles.")

    category_scores["Formatting & Bullets"] = {
        "score": format_pts,
        "max_score": 10,
        "details": f"{bullet_count} bullet points formatted"
    }

    # -------------------------------------------------------------
    # 8. Clarity, Style & Passive Language Check (Max: 10 pts)
    # -------------------------------------------------------------
    clarity_pts = 10
    detected_weak_phrases = []
    text_lower = resume_text.lower()
    for phrase in WEAK_PHRASES:
        if phrase in text_lower:
            detected_weak_phrases.append(phrase)

    if detected_weak_phrases:
        penalty = min(5, len(detected_weak_phrases) * 2)
        clarity_pts -= penalty
        issues.append(f"Found passive filler phrases: {', '.join(detected_weak_phrases)}.")
        suggestions.append("Replace passive phrases like 'responsible for' with direct verbs like 'Engineered' or 'Led'.")
    else:
        strengths.append("Professional tone free of passive filler phrases.")

    category_scores["Tone & Clarity"] = {
        "score": clarity_pts,
        "max_score": 10,
        "details": f"Passive phrases: {len(detected_weak_phrases)}"
    }

    # -------------------------------------------------------------
    # Calculate Total Score & Final Verdict
    # -------------------------------------------------------------
    total_score = sum(cat["score"] for cat in category_scores.values())
    total_score = max(0, min(100, total_score))

    if total_score >= 85:
        grade = "A+ (ATS Ready - Excellent)"
    elif total_score >= 75:
        grade = "A (Competitive & Strong)"
    elif total_score >= 60:
        grade = "B (Good - Needs Targeted Polish)"
    elif total_score >= 45:
        grade = "C (Average - Missing Key Components)"
    else:
        grade = "D (Below Average - Urgent Revision Required)"

    return {
        "total_score": total_score,
        "grade": grade,
        "category_breakdown": category_scores,
        "issues": issues,
        "strengths": strengths,
        "suggestions": suggestions
    }
