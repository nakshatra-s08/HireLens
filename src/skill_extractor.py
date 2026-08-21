"""
Skill Extractor Engine for HireLens.
Extracts canonical skills, technologies, programming languages, and experience
from conversational input using a hybrid Regex/Taxonomy + LLM structured parsing approach.
"""

import re
import json
from typing import Dict, List, Any, Set, Tuple, Optional
from src.taxonomy import SKILL_TAXONOMY, ALIAS_TO_CANONICAL, CANONICAL_TO_CATEGORY
from src.llm_client import LLMClient
from src.privacy_scrubber import scrub_demographics

class SkillExtractor:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self._compile_regexes()

    def _compile_regexes(self):
        """Compile regex patterns for alias lookup with strict word boundaries."""
        self.alias_patterns: List[Tuple[str, str, re.Pattern]] = []
        # Sort aliases by length descending so longer phrases match first (e.g. "React Native" before "React")
        sorted_aliases = sorted(ALIAS_TO_CANONICAL.items(), key=lambda x: len(x[0]), reverse=True)
        
        for alias, canonical in sorted_aliases:
            # Escape regex special characters in alias except spaces
            escaped = re.escape(alias)
            # Handle boundary for C++, C#, .NET, etc.
            pattern = re.compile(rf'(?<!\w){escaped}(?!\w)', re.IGNORECASE)
            self.alias_patterns.append((alias, canonical, pattern))

    def extract_from_text(self, text: str, scrub_privacy: bool = True) -> Dict[str, Any]:
        """
        Extract structured skills, technologies, languages, experience, and evidence from conversational text.
        """
        sanitized_text = text
        redactions = {}
        if scrub_privacy:
            sanitized_text, redactions = scrub_demographics(text)

        # 1. Regex & Taxonomy Skill Extraction
        found_skills: Set[str] = set()
        skill_evidence: Dict[str, List[str]] = {}

        # Split into sentences for evidence extraction
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', sanitized_text) if s.strip()]

        for sentence in sentences:
            for alias, canonical, pattern in self.alias_patterns:
                if pattern.search(sentence):
                    found_skills.add(canonical)
                    if canonical not in skill_evidence:
                        skill_evidence[canonical] = []
                    if sentence not in skill_evidence[canonical]:
                        skill_evidence[canonical].append(sentence)

        # 2. Extract Experience Years
        experience_years = self._extract_experience_years(sanitized_text)

        # 3. Organize Skills into Categories
        categorized_skills: Dict[str, List[str]] = {}
        languages: List[str] = []
        technologies: List[str] = []

        for skill in sorted(found_skills):
            category = CANONICAL_TO_CATEGORY.get(skill, "Other")
            if category not in categorized_skills:
                categorized_skills[category] = []
            categorized_skills[category].append(skill)

            if category == "Programming Languages":
                languages.append(skill)
            elif category in ["Frameworks & Libraries", "Cloud & DevOps", "Databases", "Data & AI / ML", "Tools & Practices"]:
                technologies.append(skill)

        # 4. Optional LLM Refinement for implicit skills / domain summary
        llm_analysis = self._llm_refine(sanitized_text, sorted(found_skills))

        return {
            "sanitized_text": sanitized_text,
            "extracted_skills": sorted(list(found_skills)),
            "categorized_skills": categorized_skills,
            "languages": sorted(languages),
            "technologies": sorted(technologies),
            "experience_years": experience_years,
            "evidence": skill_evidence,
            "privacy_redactions": redactions,
            "llm_summary": llm_analysis.get("summary", "Extraction completed successfully."),
            "domain_background": llm_analysis.get("domain", "General Engineering")
        }

    def _extract_experience_years(self, text: str) -> float:
        """Extract total numeric years of experience mentioned in profile."""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'experience\s+of\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)',
            r'working\s+for\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)',
            r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+in\s+software'
        ]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        return 0.0

    def _llm_refine(self, text: str, initial_skills: List[str]) -> Dict[str, Any]:
        """Call LLM to get structured summary and refine implicit skills."""
        prompt = f"""Analyze the candidate profile below and provide a concise JSON object:
Candidate Profile:
{text[:1000]}

Known extracted skills: {', '.join(initial_skills)}

Return JSON with keys:
- "summary": 1-2 sentence professional summary of candidate
- "domain": primary technical domain (e.g. Frontend, Backend, Machine Learning, DevOps, Mobile)
- "additional_skills": list of implicit soft skills or technical skills not in known list
"""
        response = self.llm_client.generate(prompt, system="You are an expert AI recruiter parsing candidate profiles into JSON.")
        try:
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass
        return {
            "summary": "Candidate profile processed by HireLens AI Engine.",
            "domain": "Software Engineering",
            "additional_skills": []
        }
