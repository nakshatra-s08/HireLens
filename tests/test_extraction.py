"""
Unit tests for Skill Extraction engine.
"""

from src.skill_extractor import SkillExtractor

def test_skill_extraction():
    extractor = SkillExtractor()
    sample = "I have 5 years experience with Python, Django, PostgreSQL, Docker, and AWS."
    res = extractor.extract_from_text(sample)

    assert "Python" in res["extracted_skills"]
    assert "Django" in res["extracted_skills"]
    assert "PostgreSQL" in res["extracted_skills"]
    assert "Docker" in res["extracted_skills"]
    assert "AWS" in res["extracted_skills"]
    assert res["experience_years"] == 5.0

def test_alias_canonicalization():
    extractor = SkillExtractor()
    sample = "Worked with JS, TS, K8s, Postgres, and Py."
    res = extractor.extract_from_text(sample)

    extracted = set(res["extracted_skills"])
    assert "JavaScript" in extracted
    assert "TypeScript" in extracted
    assert "Kubernetes" in extracted
    assert "PostgreSQL" in extracted
    assert "Python" in extracted

if __name__ == "__main__":
    test_skill_extraction()
    test_alias_canonicalization()
    print("Extraction tests passed!")
