"""
Explainable Candidate-Role Matcher and Ranking Engine for HireLens.
Uses scikit-learn TF-IDF, Cosine Similarity, Skill Overlap analysis, and multi-factor weighted scoring.
"""

from typing import Dict, List, Any, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_extractor import SkillExtractor
from src.privacy_scrubber import scrub_demographics
from src.config import (
    SKILL_MATCH_WEIGHT,
    SEMANTIC_MATCH_WEIGHT,
    EXP_MATCH_WEIGHT,
    DOMAIN_MATCH_WEIGHT,
    EXCELLENT_MATCH_THRESHOLD,
    GOOD_MATCH_THRESHOLD,
    FAIR_MATCH_THRESHOLD
)

class RoleMatcher:
    def __init__(self, skill_extractor: Optional[SkillExtractor] = None):
        self.skill_extractor = skill_extractor or SkillExtractor()

    def match_candidate_to_role(
        self, candidate_profile: str, job_description: str, job_title: str = "Target Role"
    ) -> Dict[str, Any]:
        """
        Evaluate candidate profile against a job description and generate an explainable match breakdown.
        """
        # 1. Scrub demographic bias from candidate profile
        candidate_clean, redactions = scrub_demographics(candidate_profile)
        job_clean, _ = scrub_demographics(job_description)

        # 2. Extract candidate skills and job required skills
        candidate_ext = self.skill_extractor.extract_from_text(candidate_clean, scrub_privacy=False)
        job_ext = self.skill_extractor.extract_from_text(job_clean, scrub_privacy=False)

        cand_skills = set(candidate_ext["extracted_skills"])
        job_skills = set(job_ext["extracted_skills"])

        # Matched and missing skills
        matched_skills = sorted(list(cand_skills.intersection(job_skills)))
        missing_skills = sorted(list(job_skills.difference(cand_skills)))
        extra_skills = sorted(list(cand_skills.difference(job_skills)))

        # 3. Calculate Skill Coverage Score
        if len(job_skills) > 0:
            skill_score = (len(matched_skills) / len(job_skills)) * 100.0
        else:
            skill_score = 50.0  # Default if no explicit skills identified in JD

        # 4. Calculate Semantic Similarity via scikit-learn TF-IDF + Cosine Similarity
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform([candidate_clean, job_clean])
            semantic_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]) * 100.0
        except Exception:
            semantic_score = 0.0

        # 5. Experience Level Match Score
        cand_exp = candidate_ext.get("experience_years", 0.0)
        req_exp = self.skill_extractor._extract_experience_years(job_clean)
        
        if req_exp == 0.0:
            exp_score = 100.0
        elif cand_exp >= req_exp:
            exp_score = 100.0
        else:
            exp_score = max(0.0, (cand_exp / req_exp) * 100.0)

        # 6. Domain Alignment Score
        cand_domain = candidate_ext.get("domain_background", "").lower()
        domain_keywords = ["backend", "frontend", "fullstack", "data", "machine learning", "devops", "cloud", "security", "mobile", "qa"]
        matched_domains = [d for d in domain_keywords if d in candidate_clean.lower() and d in job_clean.lower()]
        domain_score = 100.0 if matched_domains else 50.0

        # 7. Final Weighted Match Score Calculation
        final_score = (
            (skill_score * SKILL_MATCH_WEIGHT) +
            (semantic_score * SEMANTIC_MATCH_WEIGHT) +
            (exp_score * EXP_MATCH_WEIGHT) +
            (domain_score * DOMAIN_MATCH_WEIGHT)
        )
        final_score = round(min(100.0, max(0.0, final_score)), 1)

        # Match Rating Label
        if final_score >= EXCELLENT_MATCH_THRESHOLD:
            match_rating = "Excellent Match"
        elif final_score >= GOOD_MATCH_THRESHOLD:
            match_rating = "Good Match"
        elif final_score >= FAIR_MATCH_THRESHOLD:
            match_rating = "Fair Match"
        else:
            match_rating = "Low Match"

        # Evidence gathering for matched skills
        evidence_matched: Dict[str, Dict[str, List[str]]] = {}
        for skill in matched_skills:
            evidence_matched[skill] = {
                "candidate_evidence": candidate_ext["evidence"].get(skill, [f"Mentioned in candidate profile."]),
                "job_requirement": job_ext["evidence"].get(skill, [f"Required by job description."])
            }

        # Readable Explanation Synthesis
        explanation_lines = [
            f"**Overall Score**: {final_score}% ({match_rating})",
            f"- **Skill Coverage ({int(SKILL_MATCH_WEIGHT*100)}% weight)**: Candidate matches {len(matched_skills)} of {len(job_skills)} required skills ({round(skill_score, 1)}%).",
            f"- **Semantic Relevance ({int(SEMANTIC_MATCH_WEIGHT*100)}% weight)**: {round(semantic_score, 1)}% textual similarity based on TF-IDF profile analysis.",
            f"- **Experience Alignment ({int(EXP_MATCH_WEIGHT*100)}% weight)**: Candidate has {cand_exp} years vs. {req_exp} years required ({round(exp_score, 1)}%).",
            f"- **Domain Alignment ({int(DOMAIN_MATCH_WEIGHT*100)}% weight)**: {round(domain_score, 1)}% alignment across technical domains ({', '.join(matched_domains) if matched_domains else 'General Software Engineering'})."
        ]
        readable_explanation = "\n".join(explanation_lines)

        return {
            "job_title": job_title,
            "match_score": final_score,
            "match_rating": match_rating,
            "score_breakdown": {
                "skill_coverage_score": round(skill_score, 1),
                "semantic_similarity_score": round(semantic_score, 1),
                "experience_score": round(exp_score, 1),
                "domain_score": round(domain_score, 1),
            },
            "weights": {
                "skill": SKILL_MATCH_WEIGHT,
                "semantic": SEMANTIC_MATCH_WEIGHT,
                "experience": EXP_MATCH_WEIGHT,
                "domain": DOMAIN_MATCH_WEIGHT
            },
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "candidate_extracted": candidate_ext,
            "job_extracted": job_ext,
            "skill_evidence": evidence_matched,
            "privacy_redactions": redactions,
            "readable_explanation": readable_explanation
        }

    def rank_roles_for_candidate(
        self, candidate_profile: str, roles: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Rank a candidate against multiple available job roles.
        """
        rankings = []
        for role in roles:
            title = role.get("title", "Untitled Role")
            jd = role.get("description", "")
            match_res = self.match_candidate_to_role(candidate_profile, jd, job_title=title)
            match_res["role_id"] = role.get("id", title.lower().replace(" ", "_"))
            rankings.append(match_res)

        # Sort by match score descending
        rankings.sort(key=lambda x: x["match_score"], reverse=True)
        return rankings
