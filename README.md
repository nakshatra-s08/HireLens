# 🔍 HireLens: Explainable AI Recruiter

**HireLens** is a local, privacy-preserving, and explainable AI recruitment assistant. It extracts candidate skills from conversational profiles and resume text, matches candidates against target job roles using multi-factor weighted scoring (scikit-learn TF-IDF + Cosine Similarity + Skill Coverage), provides verifiable textual evidence snippets for every match, and powers grounded local Q&A via RAG with Ollama + Llama 8B.

Site link to run : https://mic-ai-recruiter-vibecoded-gv7wpwzo9vdlfuoey9bx7c.streamlit.app/
---

## 🛡️ Fairness & Decision-Support Notice
HireLens is explicitly designed as a **decision-support tool** for recruiters and candidates.
- **It does NOT make final hiring or rejection decisions.**
- **Anti-Bias Privacy Scrubber**: Demographic markers (Name, Age, Gender, Religion, Caste, Location, Photo, College) are automatically redacted prior to evaluation to enforce fair, non-discriminatory matching.

---

## ✨ Core Features
1. **Hybrid Skill Extraction**:
   - Compiles canonical skill names and categories (Programming Languages, Frameworks, Cloud & DevOps, Databases, Data & AI/ML, Soft Skills) using regex taxonomy matching and Ollama / Llama 8B JSON parsing.
   - Computes total extracted experience years and technical domain background.
2. **Explainable Candidate-Role Matching & Ranking**:
   - Multi-factor weighted match score (Skill Coverage %, TF-IDF Semantic Relevance %, Experience Alignment %, Domain Fit %).
   - Human-readable breakdown for every match score.
   - Skill Matrix: Matched Skills vs Missing Skills.
   - Textual Evidence Inspector: Displays exact candidate quotes and job requirements supporting each match.
   - Candidate Role Ranking: Ranks suitable roles across preset and custom job descriptions.
3. **Candidate Strengths & Skill Gap Analysis**:
   - Highlights candidate core competencies.
   - Provides clear, actionable recommendations on missing skills needed to increase role match percentage.
4. **Grounded Local RAG Q&A Chatbot**:
   - In-memory TF-IDF vector index over candidate profile, job description, and match evidence.
   - Answers recruiter & candidate questions grounded strictly in supplied data, complete with context citations.
5. **Ollama + Llama 8B & Fast Offline Fallback**:
   - Integrates seamlessly with Ollama running local `llama3:8b` or `llama3.1:8b`.
   - Includes a smart offline fallback engine so the application runs 100% reliably out of the box even if Ollama is not running.

---

## 📊 Benchmark Evaluation Metrics (15 Test Profiles)
HireLens includes an automated evaluation suite (`tests/run_benchmarks.py`) tested across 15 diverse candidate profiles:

| Metric | Result | Target |
| :--- | :---: | :---: |
| **Skill Extraction Recall** | **97.0%** | > 90% |
| **Top-1 Role Ranking Accuracy** | **86.7%** | High |
| **Top-2 Role Ranking Accuracy** | **93.3%** | High |
| **Score Explanation Coverage** | **100.0%** | 100% |
| **Grounded RAG Q&A Functionality** | **100.0%** | 100% |

---

## 🚀 Quickstart Guide

### 1. Installation & Environment Setup
```bash
# Navigate to project directory
cd /Users/nakshatrasharma/HireLens

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Running the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Running Unit Tests & Benchmarks
```bash
# Run unit tests
python -m tests.test_privacy
python -m tests.test_extraction
python -m tests.test_matcher
python -m tests.test_rag

# Run full 15-profile evaluation benchmark suite
python -m tests.run_benchmarks
```

---

## 🔌 Local Ollama Setup (Optional)
To use local Llama 8B via Ollama:
1. Install Ollama: `brew install ollama` or download from [ollama.com](https://ollama.com).
2. Start Ollama daemon: `ollama serve`
3. Pull Llama 8B model: `ollama pull llama3:8b`
HireLens will automatically detect Ollama at `http://localhost:11434` and use `llama3:8b` for RAG generation and skill refinement. If Ollama is offline, HireLens automatically falls back to its built-in offline engine.

---

## 📁 Repository Structure
```
/Users/nakshatrasharma/HireLens/
├── app.py                      # Main Streamlit Web Application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── src/
│   ├── taxonomy.py             # Canonical skill taxonomy & alias mapping
│   ├── privacy_scrubber.py     # PII & Demographic Scrubbing module
│   ├── skill_extractor.py      # Hybrid Regex & LLM Skill Extractor
│   ├── matcher.py              # Candidate-Role Matcher & Ranking Engine
│   ├── rag_engine.py           # Local Grounded RAG Chatbot Engine
│   ├── llm_client.py           # Ollama / Llama 8B client + Offline fallback
│   └── config.py               # Application configuration parameters
├── data/
│   ├── sample_jobs.json        # Benchmark job roles
│   └── test_profiles.json      # 15 realistic test candidate profiles
└── tests/
    ├── test_privacy.py         # Privacy & anti-bias unit tests
    ├── test_extraction.py      # Extraction unit tests
    ├── test_matcher.py         # Matcher unit tests
    ├── test_rag.py             # RAG unit tests
    └── run_benchmarks.py       # Benchmark runner for 15 test profiles
```
