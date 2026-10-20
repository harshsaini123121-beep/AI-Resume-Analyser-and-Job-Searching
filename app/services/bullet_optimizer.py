"""
Interactive Resume Bullet Point Optimizer based on Google's X-Y-Z formula:
'Accomplished [X] as measured by [Y], by doing [Z]'.
Evaluates strength, detects action verbs and metrics, and produces instant impact rewrites.
"""

import re
from typing import Dict, List, Any

# Prominent technical and managerial action verbs
STRONG_ACTION_VERBS = {
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

WEAK_START_PHRASES = [
    "responsible for", "duties included", "helped with", "assisted in",
    "worked on", "handled", "participated in", "contributed to", "tasked with"
]


def analyze_bullet_point(bullet_text: str) -> Dict[str, Any]:
    """
    Analyzes an individual resume bullet point against the Google X-Y-Z rubric:
    - Checks for strong initial action verbs
    - Checks for measurable quantitative outcomes (%, $, scale)
    - Checks for technical tooling / method context ('by doing Z')
    - Flags passive filler phrases
    - Returns score (0-100), diagnosis, and improved rewrite.
    """
    text = bullet_text.strip().lstrip("-•*▪–— ")
    if not text:
        return {
            "score": 0,
            "status": "Empty",
            "has_action_verb": False,
            "has_metrics": False,
            "has_context": False,
            "feedback": ["Please enter a bullet point to analyze."],
            "improved_rewrite": ""
        }

    words = text.split()
    first_word = words[0].lower().rstrip(",:;.") if words else ""

    feedback = []
    score = 0

    # 1. Action Verb Check (35 pts)
    is_weak_start = any(text.lower().startswith(p) for p in WEAK_START_PHRASES)
    has_action_verb = first_word in STRONG_ACTION_VERBS

    if is_weak_start:
        feedback.append("⚠️ Starts with passive filler phrase (e.g. 'Responsible for'). Replace with a direct action verb.")
        score += 5
    elif has_action_verb:
        feedback.append(f"✅ Strong action verb detected: '{words[0]}'.")
        score += 35
    else:
        # Check if verb appears anywhere in the first 3 words
        first_few = [w.lower().rstrip(",:;.") for w in words[:3]]
        found_verb = set(first_few).intersection(STRONG_ACTION_VERBS)
        if found_verb:
            feedback.append(f"💡 Action verb '{list(found_verb)[0]}' found, but ideally start the bullet directly with it.")
            score += 25
            has_action_verb = True
        else:
            feedback.append("⚠️ Missing a dynamic action verb at the start. Begin with verbs like 'Architected', 'Spearheaded', or 'Optimized'.")
            score += 10

    # 2. Measurable Metric Check [Y] (35 pts)
    metric_pattern = r"(?:\d+(?:\.\d+)?%|[\$₹]\s*\d+(?:,\d+)*(?:\.\d+)?|\b\d+x\b|\b\d+\s*(?:k|m|lakh|crore|lpa|inr|million|billion|users|clients|queries|hours|days|engineers|repositories)\b|\b\d{2,}\b)"
    metrics_found = re.findall(metric_pattern, text, re.IGNORECASE)
    has_metrics = len(metrics_found) > 0

    if has_metrics:
        feedback.append(f"✅ Quantifiable impact detected: {', '.join(metrics_found)}.")
        score += 35
    else:
        feedback.append("❌ Missing measurable metrics [Y]. Quantify outcomes (e.g., 'by 35%', 'saving 15 hours weekly', 'serving 50k users').")
        score += 5

    # 3. Method & Tooling Context [Z] (20 pts)
    method_cues = ["using", "utilizing", "via", "through", "by implementing", "with", "leveraging", "built with"]
    has_context = any(cue in text.lower() for cue in method_cues) or any(
        t in text.lower() for t in ["python", "react", "aws", "docker", "sql", "api", "pipeline", "kubernetes"]
    )

    if has_context:
        feedback.append("✅ Tooling / implementation context [Z] included.")
        score += 20
    else:
        feedback.append("💡 Mention specific technologies or methods [Z] used to achieve the result.")
        score += 8

    # 4. Length & Conciseness (10 pts)
    word_len = len(words)
    if 12 <= word_len <= 30:
        score += 10
        feedback.append("✅ Ideal bullet length (12-30 words).")
    elif word_len < 12:
        score += 4
        feedback.append("⚠️ Bullet is too brief. Elaborate on the business impact and tooling.")
    else:
        score += 5
        feedback.append("⚠️ Bullet is over 30 words. Condense to keep recruiter focus.")

    score = min(100, max(0, score))

    # Generate Google X-Y-Z rewrite
    # Strip passive opener
    core_content = text
    for p in WEAK_START_PHRASES:
        if core_content.lower().startswith(p):
            core_content = core_content[len(p):].strip()

    # Formulate rewrite
    suggested_verb = words[0].capitalize() if has_action_verb else "Spearheaded"
    improved_rewrite = (
        f"{suggested_verb} {core_content.rstrip('.')} utilizing industry-standard best practices, "
        f"improving team delivery speed by 30% and eliminating operational bottlenecks."
    )

    return {
        "score": score,
        "grade": "Strong (X-Y-Z Ready)" if score >= 80 else ("Moderate (Add Metrics)" if score >= 55 else "Needs Revision"),
        "has_action_verb": has_action_verb,
        "has_metrics": has_metrics,
        "has_context": has_context,
        "feedback": feedback,
        "metrics_found": metrics_found,
        "improved_rewrite": improved_rewrite
    }
