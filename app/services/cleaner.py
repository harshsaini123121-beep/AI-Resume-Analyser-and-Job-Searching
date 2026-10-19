"""
Text cleaning and normalization utilities for resumes and job descriptions.
"""

import re
import unicodedata
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw text extracted from documents:
    - Normalizes Unicode characters
    - Standardizes bullet points to standard hyphens
    - Removes non-printable and non-standard control characters
    - Compresses excessive blank lines and spaces while preserving paragraph structure
    """
    if not text:
        return ""

    # Normalize unicode (compatibility decomposition)
    text = unicodedata.normalize("NFKD", text)

    # Standardize bullet symbols to a uniform hyphen
    bullet_patterns = r"[\u2022\u2023\u25E6\u2043\u2219\u25CB\u25CF\u25AA\u25AB\u25B6\u25C6\u27A4\u27A2\u2714\u2713▪●○◆➢]"
    text = re.sub(bullet_patterns, "- ", text)

    # Standardize dashes and quotation marks
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)
    text = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', text)

    # Remove non-printable control characters, preserving \n and \t
    text = "".join(ch for ch in text if ch in ("\n", "\t") or ch.isprintable())

    # Replace tabs and irregular spaces with single spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Clean lines individually
    lines = [line.strip() for line in text.splitlines()]

    # Collapse multiple consecutive empty lines to a single empty line
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False

    return "\n".join(cleaned_lines).strip()


def mask_pii(text: str) -> str:
    """
    Masks Personally Identifiable Information (email and phone)
    prior to sending text to third-party LLMs for privacy and compliance.
    """
    if not text:
        return ""

    # Mask email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    masked = re.sub(email_pattern, "[EMAIL REDACTED]", text)

    # Mask phone numbers (US, international, 10-digit formats)
    phone_pattern = r"(\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,5}"
    masked = re.sub(phone_pattern, "[PHONE REDACTED]", masked)

    return masked


def normalize_for_matching(text: str) -> str:
    """
    Converts text to lowercase, removes special characters, and strips extra spaces
    for robust keyword and token matching.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\+\#\.\/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
