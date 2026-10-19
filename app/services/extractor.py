"""
Information extraction engine for resumes: contact info, education,
experience, standard section breakdown, and skill taxonomy matching.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.services.cleaner import clean_text

# Default skills path
DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "skills.json"

# Cached skills database
_CACHED_SKILLS: Optional[Dict[str, Any]] = None


def load_skills_database(skills_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads and caches the master skills.json file."""
    global _CACHED_SKILLS
    if _CACHED_SKILLS is not None and not skills_path:
        return _CACHED_SKILLS

    path = Path(skills_path) if skills_path else DEFAULT_SKILLS_PATH
    if not path.exists():
        # Fallback minimal dictionary if file is not found
        return {
            "categories": {
                "programming": ["python", "javascript", "java", "sql", "c++"],
                "web": ["react", "node.js", "html", "css", "fastapi"],
                "ml_ai": ["machine learning", "deep learning", "nlp", "pytorch"],
                "cloud": ["aws", "docker", "kubernetes", "git"]
            },
            "synonyms": {"js": "javascript", "py": "python", "k8s": "kubernetes"}
        }

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not skills_path:
            _CACHED_SKILLS = data
        return data


def extract_email(text: str) -> Optional[str]:
    """Extracts the first valid email address from text."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    matches = re.findall(pattern, text)
    return matches[0] if matches else None


def extract_phone(text: str) -> Optional[str]:
    """
    Extracts standard international or domestic phone numbers.
    Avoids matching standalone 4-digit years or zip codes.
    """
    pattern = r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}"
    matches = re.findall(pattern, text)
    for match in matches:
        digits_only = re.sub(r"\D", "", match)
        if 10 <= len(digits_only) <= 15:
            return match.strip()
    return None


def extract_links(text: str) -> Dict[str, Optional[str]]:
    """Extracts LinkedIn, GitHub, and Portfolio URLs."""
    links: Dict[str, Optional[str]] = {
        "linkedin": None,
        "github": None,
        "portfolio": None
    }

    # LinkedIn
    linkedin_match = re.search(
        r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|pub)\/([a-zA-Z0-9_-]+)\/?",
        text,
        re.IGNORECASE
    )
    if linkedin_match:
        links["linkedin"] = linkedin_match.group(0).strip()

    # GitHub
    github_match = re.search(
        r"(?:https?:\/\/)?(?:www\.)?github\.com\/([a-zA-Z0-9_-]+)\/?",
        text,
        re.IGNORECASE
    )
    if github_match:
        links["github"] = github_match.group(0).strip()

    # Personal Portfolio or other web links
    portfolio_match = re.search(
        r"(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9-]+\.(?:dev|io|me|app|ai|org|tech|co|net|com))\/?(?!\S*@)",
        text,
        re.IGNORECASE
    )
    if portfolio_match:
        found_url = portfolio_match.group(0).strip()
        if not any(excluded in found_url.lower() for excluded in ["linkedin", "github", "gmail", "yahoo"]):
            links["portfolio"] = found_url

    return links


def extract_name(text: str) -> Optional[str]:
    """
    Extracts the candidate's name using heuristic analysis of candidate header lines
    and optional spaCy NER if available.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    # Filter out email, phone, links, and common section titles from the first 5 lines
    candidate_lines = lines[:5]
    excluded_keywords = [
        "resume", "curriculum", "vitae", "summary", "profile", "contact",
        "email", "phone", "linkedin", "github", "http", "www", "page", "@",
        "engineer", "developer", "scientist", "analyst", "manager"
    ]

    for line in candidate_lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in excluded_keywords):
            continue
        # Names typically have 2 to 4 words, with alphabetic characters and hyphens/dots
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.match(r"^[A-Za-z\.\'-]+$", w) for w in words):
            # Check capitalization
            if any(w[0].isupper() for w in words):
                return line

    # Fallback to the very first line if short and plausible
    if lines:
        first_line = lines[0]
        words = first_line.split()
        if 1 <= len(words) <= 4 and not any(ch in first_line for ch in ["@", "http", "github", "linkedin"]):
            return first_line

    return None


def detect_sections(text: str) -> Dict[str, str]:
    """
    Detects standard resume sections (Summary, Skills, Experience, Education, Projects, Certifications)
    and segments the text into corresponding sections.
    """
    section_patterns = {
        "summary": r"(?:professional\s+summary|summary|profile|about\s+me|career\s+objective|objective)",
        "skills": r"(?:technical\s+skills|core\s+skills|skills\s*(?:&|and)?\s*technologies|key\s+skills|skills|competencies)",
        "experience": r"(?:professional\s+experience|work\s+experience|employment\s+history|experience|work\s+history)",
        "education": r"(?:education\s*(?:&|and)?\s*credentials|educational\s+background|education|academics)",
        "projects": r"(?:key\s+projects|academic\s+projects|personal\s+projects|notable\s+projects|projects)",
        "certifications": r"(?:certifications\s*(?:&|and)?\s*licenses|certifications|courses|achievements|awards)"
    }

    # Build regex to detect all section headers
    header_regexes = []
    for section_name, pattern in section_patterns.items():
        header_regexes.append(rf"(?P<{section_name}>^[ \t]*(?:#+\s*)?{pattern}[ \t]*:?[ \t]*$)")

    combined_regex = re.compile("|".join(header_regexes), re.IGNORECASE | re.MULTILINE)

    # Find all header matches
    matches = list(combined_regex.finditer(text))
    if not matches:
        return {"full_content": text}

    sections: Dict[str, str] = {}
    for i, match in enumerate(matches):
        section_key = match.lastgroup
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start_idx:end_idx].strip()
        if section_key:
            sections[section_key] = content

    return sections


def extract_skills(text: str, skills_path: Optional[str] = None) -> List[str]:
    """
    Extracts technical and soft skills by matching against data/skills.json
    with synonym normalization and word boundary preservation.
    """
    skills_db = load_skills_database(skills_path)
    categories = skills_db.get("categories", {})
    synonyms = skills_db.get("synonyms", {})

    # Flatten all skills into a master set
    all_target_skills = set()
    for skill_list in categories.values():
        for s in skill_list:
            all_target_skills.add(s.lower().strip())

    # Add synonym triggers
    all_target_skills.update(synonyms.keys())

    text_lower = f" {text.lower()} "
    # Replace punctuation that connects words awkwardly while keeping special tokens (+, #, ., -)
    sanitized_text = re.sub(r"[,\(\)\[\]\{\};:\|\"\'\\\/]+", " ", text_lower)

    detected_skills = set()

    for skill in all_target_skills:
        # Create safe regex pattern with word boundaries
        # Special handling for skills like c++, c#, .net
        escaped = re.escape(skill)
        if skill in ("c++", "c#", ".net", ".net core"):
            pattern = rf"(?:^|\s){escaped}(?:\s|$)"
        else:
            pattern = rf"\b{escaped}\b"

        if re.search(pattern, sanitized_text):
            # Check if this skill is an alias that maps to a canonical name
            canonical_skill = synonyms.get(skill, skill)
            detected_skills.add(canonical_skill)

    return sorted(list(detected_skills))


def extract_education(text: str) -> List[Dict[str, Any]]:
    """
    Detects degrees, institutions, and graduation years.
    """
    degree_patterns = [
        r"\b(?:Ph\.?D\.?|Doctor of Philosophy)\b",
        r"\b(?:M\.?Tech\.?|M\.?S\.?|Master of Science|Master of Technology|M\.?Sc\.?|MBA|Master of Business Administration)\b",
        r"\b(?:B\.?Tech\.?|B\.?E\.?|B\.?S\.?|Bachelor of Science|Bachelor of Technology|Bachelor of Engineering|B\.?Sc\.?|BCA|MCA)\b",
        r"\b(?:Associate Degree|Diploma|High School Diploma)\b"
    ]

    education_records = []
    lines = text.splitlines()

    for line in lines:
        for deg_pat in degree_patterns:
            deg_match = re.search(deg_pat, line, re.IGNORECASE)
            if deg_match:
                degree = deg_match.group(0).strip()
                # Search for graduation year (4 digits between 1980 and 2035)
                year_match = re.search(r"\b(19\d{2}|20[0-3]\d)\b", line)
                year = int(year_match.group(1)) if year_match else None

                # Extract institution heuristics (University, College, Institute, School)
                inst_match = re.search(
                    r"([A-Za-z\s]+(?:University|College|Institute|School|Academy|Polytechnic)[A-Za-z\s]*)",
                    line,
                    re.IGNORECASE
                )
                institution = inst_match.group(1).strip() if inst_match else None

                education_records.append({
                    "degree": degree,
                    "institution": institution,
                    "year": year,
                    "raw_text": line.strip()
                })
                break

    return education_records


def extract_experience_years(text: str) -> float:
    """
    Calculates estimated years of experience from:
    1. Direct textual mentions (e.g., '5+ years of experience')
    2. Date ranges in experience sections (e.g. '2019 - 2023', 'Jan 2021 - Present')
    """
    # 1. Direct explicit mentions
    explicit_pattern = r"(\d+(?:\.\d+)?)\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)(?:\s+of)?\s+experience"
    explicit_match = re.search(explicit_pattern, text, re.IGNORECASE)
    if explicit_match:
        try:
            return float(explicit_match.group(1))
        except ValueError:
            pass

    # 2. Date range heuristics
    current_year = datetime.now().year
    # Patterns like: 2018 - 2022 or 2020 - Present or Jan 2019 - Dec 2021
    range_pattern = r"\b(19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(19\d{2}|20\d{2}|present|current)\b"
    ranges = re.findall(range_pattern, text, re.IGNORECASE)

    total_years = 0.0
    for start_str, end_str in ranges:
        try:
            start_year = int(start_str)
            if end_str.lower() in ("present", "current"):
                end_year = current_year
            else:
                end_year = int(end_str)

            if 1980 <= start_year <= end_year <= current_year + 1:
                duration = max(0.5, float(end_year - start_year))
                total_years += duration
        except Exception:
            continue

    if total_years > 0:
        # Cap realistic single-career span if ranges overlap
        return round(min(total_years, 35.0), 1)

    return 0.0


def extract_resume_data(text: str, skills_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Master extraction function returning the unified candidate profile dictionary.
    """
    cleaned = clean_text(text)
    name = extract_name(cleaned)
    email = extract_email(cleaned)
    phone = extract_phone(cleaned)
    links = extract_links(cleaned)
    sections = detect_sections(cleaned)
    skills = extract_skills(cleaned, skills_path)
    education = extract_education(cleaned)
    experience_years = extract_experience_years(cleaned)

    # Word count and bullet point metrics
    word_count = len(cleaned.split())
    bullet_points = [line for line in cleaned.splitlines() if line.strip().startswith("-")]

    return {
        "candidate_name": name or "Candidate",
        "email": email,
        "phone": phone,
        "links": links,
        "skills": skills,
        "skills_count": len(skills),
        "education": education,
        "experience_years": experience_years,
        "sections_detected": list(sections.keys()),
        "section_content": sections,
        "word_count": word_count,
        "bullet_count": len(bullet_points),
        "raw_text": cleaned
    }
