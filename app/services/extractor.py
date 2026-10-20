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


# Disambiguation constants for ambiguous short tokens and everyday verbs/nouns
SHORT_AMBIGUOUS_SKILLS = {"r", "go", "c"}
COMMON_AMBIGUOUS_SKILLS = {"react", "spark", "excel", "word", "spring"}

MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12
}

STOP_CONTEXT_PATTERNS = {
    "go": [
        r"\bgo\s+to\b", r"\bgo\s+through\b", r"\bready\s+to\s+go\b", r"\bon\s+the\s+go\b",
        r"\blet\s+go\b", r"\bgo\s+ahead\b", r"\bgo\s+forward\b", r"\bgo\s+beyond\b",
        r"\bway\s+to\s+go\b", r"\bgo\s+live\b", r"\bgo\s+with\b", r"\bgo\s+back\b"
    ],
    "react": [
        r"\breact\s+quickly\b", r"\breact\s+to\b", r"\bability\s+to\s+react\b",
        r"\bhow\s+to\s+react\b", r"\bfailed\s+to\s+react\b", r"\bquick\s+to\s+react\b",
        r"\bslow\s+to\s+react\b", r"\breacted\s+to\b"
    ],
    "spark": [
        r"\bspark\s+joy\b", r"\bspark\s+interest\b", r"\bspark\s+curiosity\b",
        r"\bspark\s+a\s+discussion\b", r"\bspark\s+innovation\b", r"\bspark\s+creativity\b",
        r"\bspark\s+change\b", r"\bbright\s+spark\b"
    ],
    "excel": [
        r"\bexcel\s+at\b", r"\bexcel\s+in\b", r"\bexcelled\s+at\b", r"\bexcelled\s+in\b",
        r"\bability\s+to\s+excel\b", r"\bstrive\s+to\s+excel\b", r"\bhelping\s+you\s+excel\b"
    ],
    "word": [
        r"\bword\s+of\s+mouth\b", r"\bword\s+for\s+word\b", r"\bin\s+other\s+words\b",
        r"\bspread\s+the\s+word\b", r"\bwritten\s+word\b", r"\bspoken\s+word\b"
    ],
    "spring": [
        r"\bspring\s+(?:20\d\d|19\d\d)\b", r"\bspring\s+semester\b", r"\bspring\s+break\b",
        r"\bspring\s+quarter\b"
    ]
}


def _has_technical_context_for_short_skill(skill: str, full_text: str, skills_section_text: str) -> bool:
    """Verifies that short ambiguous tokens (R, Go, C) appear in a genuine technical context."""
    # 1. Direct match in dedicated skills section
    if skills_section_text:
        if skill == "r" and re.search(r"\b[Rr]\b", skills_section_text):
            return True
        if skill == "go" and re.search(r"\b(?:[Gg]o|Golang)\b", skills_section_text):
            return True
        if skill == "c" and re.search(r"\b[Cc]\b(?:\s*[/,]\s*[Cc]\+\+)?", skills_section_text):
            return True

    # 2. Case-sensitive or contextual match in full document
    if skill == "r":
        r_patterns = [
            r"\b[Rr]\s+(?:programming|language|scripting|packages|studio|data|scripts)\b",
            r"\b(?:programming\s+in|code\s+in|using)\s+[Rr]\b",
            r"\b[Rr]\s*[/,]\s*(?:Python|SQL|MATLAB|SAS|Julia)\b",
            r"\b(?:Python|SQL|MATLAB|SAS|Julia)\s*[/,]\s*[Rr]\b"
        ]
        return any(re.search(pat, full_text) for pat in r_patterns)

    elif skill == "go":
        if re.search(r"\bGolang\b", full_text, re.IGNORECASE):
            return True
        go_patterns = [
            r"\bGo\s+(?:programming|language|developer|engineer|backend|microservices|goroutines)\b",
            r"\b(?:programming\s+in|code\s+in|using|skilled\s+in|proficient\s+in)\s+[^.\n]*?\bGo\b",
            r"\bGo\s*[/,]\s*(?:and\s+)?(?:Python|Java|Rust|C\+\+|C|R|Kubernetes|Docker)\b",
            r"\b(?:Python|Java|Rust|C\+\+|C|R|Kubernetes|Docker)\s*[/,]\s*(?:and\s+)?Go\b"
        ]
        return any(re.search(pat, full_text) for pat in go_patterns)

    elif skill == "c":
        c_patterns = [
            r"\bC\s*[/,]\s*C\+\+",
            r"\bC\s+and\s+C\+\+",
            r"\b(?:Embedded\s+C|ANSI\s+C|C\s+programming|C\s+language|C\s+developer)\b",
            r"\b(?:programming\s+in|code\s+in|using)\s+C\b",
            r"\b(?:Python|Java|Assembly|Rust)\s*[/,]\s*C\b"
        ]
        return any(re.search(pat, full_text) for pat in c_patterns)

    return False


def _is_blacklisted_usage(skill: str, full_text: str, skills_section_text: str) -> bool:
    """Returns True if the only occurrences of an ambiguous skill match common stop-contexts."""
    if skills_section_text and re.search(rf"\b{re.escape(skill)}\b", skills_section_text, re.IGNORECASE):
        return False

    patterns = STOP_CONTEXT_PATTERNS.get(skill.lower())
    if not patterns:
        return False

    has_stop_context = any(re.search(pat, full_text, re.IGNORECASE) for pat in patterns)
    if has_stop_context:
        tech_indicators = {
            "react": r"\b(?:react(?:\.js|js| native)?\s*(?:frontend|components?|developer|app|library|hooks|redux)?|ui\s+with\s+react)\b",
            "spark": r"\b(?:apache\s+spark|spark\s+streaming|spark\s+sql|pyspark|spark\s+cluster)\b",
            "excel": r"\b(?:ms\s+excel|microsoft\s+excel|advanced\s+excel|excel\s+vba|excel\s+spreadsheets?|formulas|vlookup|pivot)\b",
            "word": r"\b(?:ms\s+word|microsoft\s+word|word\s+processing)\b",
            "spring": r"\b(?:spring\s+boot|spring\s+mvc|spring\s+framework|spring\s+security|java\s+spring)\b"
        }
        tech_pat = tech_indicators.get(skill.lower())
        if tech_pat and re.search(tech_pat, full_text, re.IGNORECASE):
            return False
        return True

    return False


def extract_skills(text: str, skills_path: Optional[str] = None) -> List[str]:
    """
    Extracts technical and soft skills by matching against data/skills.json
    with synonym normalization, ambiguous token disambiguation, and stop-context filtering.
    """
    skills_db = load_skills_database(skills_path)
    categories = skills_db.get("categories", {})
    synonyms = skills_db.get("synonyms", {})

    sections = detect_sections(text)
    skills_section_text = sections.get("skills", "")

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
        skill_lower = skill.lower()

        # Handle short tokens with strict casing & technical context checks
        if skill_lower in SHORT_AMBIGUOUS_SKILLS:
            if _has_technical_context_for_short_skill(skill_lower, text, skills_section_text):
                canonical_skill = synonyms.get(skill, skill)
                detected_skills.add(canonical_skill)
            continue

        # Handle common ambiguous English words with stop-context blacklist
        if skill_lower in COMMON_AMBIGUOUS_SKILLS:
            if _is_blacklisted_usage(skill_lower, text, skills_section_text):
                continue

        # Create safe regex pattern with word boundaries
        escaped = re.escape(skill)
        if skill in ("c++", "c#", ".net", ".net core"):
            pattern = rf"(?:^|\s){escaped}(?:\s|$)"
        else:
            pattern = rf"\b{escaped}\b"

        if re.search(pattern, sanitized_text):
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
    Calculates estimated years of experience:
    - Excludes education and certification dates from inflating experience
    - Supports month-level resolution (e.g. 'Jan 2021 - Dec 2021' = 1.0 yr)
    - Merges overlapping or concurrent employment date intervals
    - Reconciles explicit phrases ('X years of experience') with date spans
    """
    sections = detect_sections(text)

    # 1. Direct explicit mention check
    explicit_years = None
    explicit_pattern = r"(\d+(?:\.\d+)?)\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)(?:\s+of)?\s+experience"
    explicit_match = re.search(explicit_pattern, text, re.IGNORECASE)
    if explicit_match:
        try:
            explicit_years = float(explicit_match.group(1))
        except ValueError:
            pass

    # 2. Isolate experience text (exclude education and certification sections)
    exp_text = sections.get("experience", "")
    if not exp_text or len(exp_text.strip()) < 40:
        edu_text = sections.get("education", "")
        cert_text = sections.get("certifications", "")
        filtered = text
        if edu_text:
            filtered = filtered.replace(edu_text, " ")
        if cert_text:
            filtered = filtered.replace(cert_text, " ")
        exp_text = filtered

    current_year = datetime.now().year
    current_month = datetime.now().month

    def parse_month(m_str: Optional[str]) -> Optional[int]:
        if not m_str:
            return None
        clean = m_str.lower().strip("., ")
        return MONTH_MAP.get(clean)

    intervals = []

    # Pattern A: Named Month + Year to Named Month + Year / Present
    named_range_pat = re.compile(
        r"\b(?P<s_m>[A-Za-z]{3,9}\.?)?\s*(?P<s_y>19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(?:(?P<e_m>[A-Za-z]{3,9}\.?)?\s*(?P<e_y>19\d{2}|20\d{2})|(?P<present>present|current))\b",
        re.IGNORECASE
    )
    for m in named_range_pat.finditer(exp_text):
        s_y = int(m.group("s_y"))
        parsed_sm = parse_month(m.group("s_m"))
        s_m = parsed_sm if parsed_sm else 1
        parsed_em = None

        if m.group("present"):
            e_y = current_year
            e_m = current_month
        else:
            e_y = int(m.group("e_y"))
            parsed_em = parse_month(m.group("e_m"))
            # If months were provided on either side, use 12 for end month, otherwise 1
            if parsed_em:
                e_m = parsed_em
            elif parsed_sm:
                e_m = 12
            else:
                e_m = 1

        if 1980 <= s_y <= e_y <= current_year + 1:
            start_val = s_y + (s_m - 1) / 12.0
            end_val = e_y + (e_m / 12.0 if (parsed_em or parsed_sm or m.group("present")) else 0.0)
            if end_val <= start_val:
                end_val = start_val + max(0.5, float(e_y - s_y))
            intervals.append((start_val, end_val))

    # Pattern B: Numeric Month/Year (e.g., 01/2019 - 06/2021)
    numeric_range_pat = re.compile(
        r"\b(?P<s_m>0?[1-9]|1[0-2])\s*[/\-]\s*(?P<s_y>19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(?:(?P<e_m>0?[1-9]|1[0-2])\s*[/\-]\s*(?P<e_y>19\d{2}|20\d{2})|(?P<present>present|current))\b",
        re.IGNORECASE
    )
    for m in numeric_range_pat.finditer(exp_text):
        s_y = int(m.group("s_y"))
        s_m = int(m.group("s_m"))

        if m.group("present"):
            e_y = current_year
            e_m = current_month
        else:
            e_y = int(m.group("e_y"))
            e_m = int(m.group("e_m"))

        if 1980 <= s_y <= e_y <= current_year + 1:
            start_val = s_y + (s_m - 1) / 12.0
            end_val = e_y + e_m / 12.0
            if end_val > start_val:
                intervals.append((start_val, end_val))

    # 3. Merge overlapping date intervals
    if intervals:
        intervals.sort(key=lambda x: (x[0], x[1]))
        merged = []
        for s, e in intervals:
            if not merged:
                merged.append([s, e])
            else:
                prev_s, prev_e = merged[-1]
                if s <= prev_e:
                    merged[-1][1] = max(prev_e, e)
                else:
                    merged.append([s, e])

        date_calculated_years = sum(e - s for s, e in merged)
    else:
        date_calculated_years = 0.0

    # 4. Reconcile explicit mentions vs calculated date intervals
    if explicit_years is not None:
        if date_calculated_years == 0.0:
            return round(min(explicit_years, 35.0), 1)
        if abs(explicit_years - date_calculated_years) <= 2.5:
            return round(min(max(explicit_years, date_calculated_years), 35.0), 1)
        return round(min(date_calculated_years, 35.0), 1)

    return round(min(date_calculated_years, 35.0), 1)


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
