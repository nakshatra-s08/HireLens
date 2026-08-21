"""
Unit tests for Local RAG Engine.
"""

from src.rag_engine import LocalRAGEngine
from src.matcher import RoleMatcher

def test_rag_retrieval_and_answer():
    matcher = RoleMatcher()
    candidate = "I have 6 years of experience in Python, FastAPI, and Docker."
    job_desc = "Seeking Backend Dev with Python, FastAPI, Docker, and Redis."
    match_res = matcher.match_candidate_to_role(candidate, job_desc)

    rag = LocalRAGEngine()
    rag.build_index(match_res, candidate, job_desc)
    
    contexts = rag.retrieve_context("What skills are missing?")
    assert len(contexts) > 0
    
    qa_res = rag.answer_question("Does the candidate know Python?")
    assert qa_res["answer"] != ""

if __name__ == "__main__":
    test_rag_retrieval_and_answer()
    print("RAG tests passed!")
