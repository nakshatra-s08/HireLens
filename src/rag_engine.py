"""
Local RAG Engine for HireLens Recruiter.
Provides grounded Q&A chatbot capabilities over candidate profiles, job descriptions, and match evidence.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.llm_client import LLMClient

class LocalRAGEngine:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.documents: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None

    def build_index(self, match_result: Dict[str, Any], candidate_text: str, job_text: str):
        """
        Build an in-memory TF-IDF vector index over candidate text, job description, and match breakdown.
        """
        raw_docs = []
        
        # 1. Candidate text sentences
        cand_sents = [s.strip() for s in re.split(r'[.!?\n]+', candidate_text) if len(s.strip()) > 10]
        for s in cand_sents:
            raw_docs.append(f"[Candidate Profile]: {s}")

        # 2. Job description sentences
        job_sents = [s.strip() for s in re.split(r'[.!?\n]+', job_text) if len(s.strip()) > 10]
        for s in job_sents:
            raw_docs.append(f"[Job Description]: {s}")

        # 3. Match breakdown evidence
        matched = match_result.get("matched_skills", [])
        missing = match_result.get("missing_skills", [])
        raw_docs.append(f"[Skill Match Summary]: Candidate matched skills: {', '.join(matched)}. Missing skills: {', '.join(missing)}.")
        raw_docs.append(f"[Score Explanation]: {match_result.get('readable_explanation', '')}")

        for skill, ev in match_result.get("skill_evidence", {}).items():
            cand_ev = " ".join(ev.get("candidate_evidence", []))
            job_ev = " ".join(ev.get("job_requirement", []))
            raw_docs.append(f"[Evidence for {skill}]: Candidate evidence: {cand_ev}. Required by job: {job_ev}.")

        self.documents = raw_docs
        if self.documents:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top k most relevant context snippets for a query."""
        if not self.documents or not self.vectorizer or self.doc_vectors is None:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self.doc_vectors)[0]
            top_indices = sims.argsort()[-top_k:][::-1]
            retrieved = []
            for idx in top_indices:
                if sims[idx] > 0.05:  # Relevance threshold
                    retrieved.append(self.documents[idx])
            return retrieved
        except Exception:
            return self.documents[:top_k]

    def answer_question(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Answer question grounded strictly in retrieved context.
        """
        context_snippets = self.retrieve_context(query, top_k=top_k)
        
        if not context_snippets:
            context_str = "No specific profile or job data indexed."
        else:
            context_str = "\n".join(context_snippets)

        system_prompt = (
            "You are HireLens Explainable AI Assistant. Answer the user's question using ONLY the provided context snippets. "
            "If the answer cannot be answered from the context, state clearly: 'Based on the candidate profile and job data, this information is not mentioned.' "
            "Do NOT make up facts or assumptions."
        )

        prompt = f"""CONTEXT SNIPPETS:
{context_str}

USER QUESTION:
{query}

ANSWER (grounded strictly in context):"""

        response = self.llm_client.generate(prompt, system=system_prompt)

        return {
            "query": query,
            "answer": response,
            "retrieved_context": context_snippets
        }
