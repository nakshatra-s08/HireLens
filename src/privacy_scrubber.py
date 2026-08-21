"""
Privacy & Anti-Bias Scrubber for HireLens.
Filters out non-job-relevant demographic attributes (Name, Age, Gender, Religion, Caste, Location, University)
to ensure fair, non-discriminatory candidate matching.
"""

import re
from typing import Dict, Tuple

DEMOGRAPHIC_PATTERNS = [
    # Age patterns
    (r'\b(?:\d{1,2}|thirty|twenty|forty|fifty)\s*(?:years?\s*old|yo|year-old)\b', '[REDACTED_AGE]'),
    (r'\bborn\s*in\s*(?:19|20)\d{2}\b', '[REDACTED_AGE]'),
    
    # Gender patterns
    (r'\b(?:male|female|man|woman|guy|girl|he/him|she/her|they/them)\b', '[REDACTED_GENDER]'),
    (r'\b(?:he|she|his|her|hers)\b', '[REDACTED_PRONOUN]'),

    # Religion & Caste patterns
    (r'\b(?:christian|muslim|hindu|sikh|buddhist|jewish|catholic|protestant|brahmin|kshatriya|vaishya|shudra|caste)\b', '[REDACTED_DEMOGRAPHIC]'),

    # University / College patterns
    (r'\b(?:university\s+of\s+[A-Za-z\s]+|stanford|mit|harvard|berkeley|oxford|cambridge|iit\s+[A-Za-z]+|nit\s+[A-Za-z]+|college|university)\b', '[REDACTED_UNIVERSITY]'),

    # Location patterns
    (r'\bbased\s+in\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)?\b', '[REDACTED_LOCATION]'),
    (r'\bliving\s+in\s+[A-Z][a-z]+\b', '[REDACTED_LOCATION]'),
    (r'\bfrom\s+(?:india|usa|uk|canada|germany|france|australia|singapore|china|japan|london|new york|san francisco|bangalore)\b', '[REDACTED_LOCATION]'),

    # Name introduction patterns
    (r'\bmy\s+name\s+is\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', 'My name is [REDACTED_NAME]'),
    (r'\bi\s+am\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?=\s*,|\s*\.|\s+a|\s+an)\b', 'I am [REDACTED_NAME]')
]

def scrub_demographics(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Scrub demographic markers from input text to ensure fairness in downstream matching.
    Returns:
        sanitized_text (str): Scrubbed text with placeholders.
        redactions_summary (Dict[str, int]): Count of redactions by category.
    """
    sanitized_text = text
    redactions_summary: Dict[str, int] = {}

    for pattern, replacement in DEMOGRAPHIC_PATTERNS:
        matches = re.findall(pattern, sanitized_text, flags=re.IGNORECASE)
        if matches:
            redactions_summary[replacement] = redactions_summary.get(replacement, 0) + len(matches)
            sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=re.IGNORECASE)

    return sanitized_text, redactions_summary
