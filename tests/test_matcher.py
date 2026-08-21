"""
Unit tests for Matcher and Ranking Engine.
"""

from src.matcher import RoleMatcher

def test_matcher_breakdown():
    matcher = RoleMatcher()
    candidate = "I have 5 years experience with Python, FastAPI, PostgreSQL, Docker, and REST API."
    job_desc = "Looking for Senior Backend Engineer with Python, FastAPI, PostgreSQL, Docker, Redis, and REST API."
    
    result = matcher.match_candidate_to_role(candidate, job_desc)
    assert result["match_score"] > 60.0
    assert "Python" in result["matched_skills"]
    assert "Redis" in result["missing_skills"]
    assert "score_breakdown" in result
    assert result["readable_explanation"] != ""

if __name__ == "__main__":
    test_matcher_breakdown()
    print("Matcher tests passed!")
