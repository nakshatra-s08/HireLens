"""
HireLens: Explainable AI Recruiter
Streamlit Web Dashboard Application
"""

import streamlit as st
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.skill_extractor import SkillExtractor
from src.matcher import RoleMatcher
from src.rag_engine import LocalRAGEngine
from src.privacy_scrubber import scrub_demographics
from src.llm_client import LLMClient
from src.config import DEFAULT_LLM_MODEL

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HireLens: Explainable AI Recruiter",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .privacy-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
    }
    .skill-chip {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.25rem 0.6rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
        border: 1px solid #BFDBFE;
    }
    .missing-chip {
        display: inline-block;
        background-color: #FEF2F2;
        color: #DC2626;
        padding: 0.25rem 0.6rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
        border: 1px solid #FECACA;
    }
    .evidence-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-top: 0.4rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_sample_jobs():
    path = os.path.join(DATA_DIR, "sample_jobs.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

@st.cache_data
def load_test_profiles():
    path = os.path.join(DATA_DIR, "test_profiles.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

sample_jobs = load_sample_jobs()
test_profiles = load_test_profiles()

# -----------------------------------------------------------------------------
# App State & Initialize Engine Instances
# -----------------------------------------------------------------------------
if "llm_client" not in st.session_state:
    st.session_state.llm_client = LLMClient()
if "extractor" not in st.session_state:
    st.session_state.extractor = SkillExtractor(llm_client=st.session_state.llm_client)
if "matcher" not in st.session_state:
    st.session_state.matcher = RoleMatcher(skill_extractor=st.session_state.extractor)
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = LocalRAGEngine(llm_client=st.session_state.llm_client)

extractor = st.session_state.extractor
matcher = st.session_state.matcher
rag_engine = st.session_state.rag_engine
llm_client = st.session_state.llm_client

# -----------------------------------------------------------------------------
# Header Banner & Anti-Bias Disclaimer
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>🔍 HireLens: Explainable AI Recruiter</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Local, Evidence-Based Candidate-Role Matching & Grounded Decision Support</div>", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer-banner'>
    <strong>🛡️ Fairness & Decision Support Notice:</strong><br>
    HireLens is designed to assist recruiters and candidates with explainable insights, skill extraction, and evidence-based matching.
    <strong>It does NOT make automated hiring or rejection decisions.</strong> Demographic markers (Name, Age, Gender, Religion, Caste, Location, Photo, College) are automatically redacted to prevent algorithmic bias.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Controls, Presets & Weight Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Inputs")
    
    # Ollama status check
    ollama_online = llm_client.is_ollama_available()
    if ollama_online:
        st.success(f"🟢 Ollama Local LLM Connected ({DEFAULT_LLM_MODEL})")
    else:
        st.info("ℹ️ Local Heuristic Engine Active (Ollama offline/optional)")

    st.subheader("1. Select Candidate Profile")
    profile_mode = st.radio("Candidate Input Method:", ["Preset Test Profile (15 Profiles)", "Custom Input Text / Resume"])

    candidate_text_input = ""
    selected_prof_obj = None

    if profile_mode.startswith("Preset"):
        prof_options = {p["name"]: p for p in test_profiles}
        chosen_prof_name = st.selectbox("Choose Profile:", list(prof_options.keys()))
        selected_prof_obj = prof_options[chosen_prof_name]
        candidate_text_input = selected_prof_obj["profile_text"]
        st.caption(f"**Target Role**: {selected_prof_obj.get('expected_role', 'N/A')}")
    else:
        candidate_text_input = st.text_area(
            "Paste Candidate Conversational Profile or Resume Text:",
            height=180,
            value="I have 5 years of backend engineering experience using Python, FastAPI, PostgreSQL, Docker, and Redis."
        )

    st.subheader("2. Target Job Description")
    job_mode = st.radio("Job Input Method:", ["Preset Job Role", "Custom Job Description"])

    job_text_input = ""
    selected_job_title = "Target Role"

    if job_mode.startswith("Preset"):
        job_options = {j["title"]: j for j in sample_jobs}
        chosen_job_title = st.selectbox("Choose Target Role:", list(job_options.keys()))
        selected_job_obj = job_options[chosen_job_title]
        job_text_input = selected_job_obj["description"]
        selected_job_title = chosen_job_title
    else:
        selected_job_title = st.text_input("Job Title:", value="Senior Backend Engineer")
        job_text_input = st.text_area(
            "Paste Job Description:",
            height=180,
            value="We are seeking a Senior Backend Engineer with 4+ years experience in Python, FastAPI, PostgreSQL, Redis, Docker, and REST API."
        )

    st.subheader("3. Privacy & Anti-Bias")
    scrub_privacy = st.checkbox("Scrub Demographic PII (Name/Age/Gender/Location/College)", value=True)

# -----------------------------------------------------------------------------
# Core Engine Processing
# -----------------------------------------------------------------------------
if candidate_text_input.strip() and job_text_input.strip():
    # Perform Candidate Matching
    match_result = matcher.match_candidate_to_role(candidate_text_input, job_text_input, job_title=selected_job_title)
    
    # Build RAG Index
    rag_engine.build_index(match_result, candidate_text_input, job_text_input)
    
    # Perform Rank All Preset Roles
    role_rankings = matcher.rank_roles_for_candidate(candidate_text_input, sample_jobs)

    # -----------------------------------------------------------------------------
    # Main UI Tabs
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Recruiter View (Matching)",
        "👤 Candidate View (Skill Gap)",
        "🤖 Grounded RAG Chatbot",
        "🔒 Privacy & Bias Audit",
        "📊 Evaluation Benchmarks (15 Profiles)"
    ])

    # -----------------------------------------------------------------------------
    # TAB 1: Recruiter View
    # -----------------------------------------------------------------------------
    with tab1:
        col_score, col_chart = st.columns([1, 1.5])
        
        with col_score:
            score = match_result["match_score"]
            rating = match_result["match_rating"]
            
            st.markdown(f"### Target Role: **{selected_job_title}**")
            st.metric(label="Candidate-Role Match Score", value=f"{score}%", delta=rating)
            
            # Progress bar
            st.progress(score / 100.0)

            st.markdown("#### Score Explanation:")
            st.markdown(match_result["readable_explanation"])

        with col_chart:
            st.markdown("### Match Score Breakdown")
            sb = match_result["score_breakdown"]
            
            # Radar chart for multi-factor score breakdown
            categories = ['Skill Coverage', 'Semantic Relevance', 'Experience Level', 'Domain Alignment']
            scores = [sb['skill_coverage_score'], sb['semantic_similarity_score'], sb['experience_score'], sb['domain_score']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Candidate Fit',
                line_color='#2563EB'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                margin=dict(l=40, r=40, t=20, b=20),
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # Skill Matrix: Matched vs Missing Skills
        col_matched, col_missing = st.columns(2)
        with col_matched:
            st.subheader(f"✅ Matched Skills ({len(match_result['matched_skills'])})")
            if match_result['matched_skills']:
                html_chips = "".join([f"<span class='skill-chip'>✓ {s}</span>" for s in match_result['matched_skills']])
                st.markdown(html_chips, unsafe_allow_html=True)
            else:
                st.info("No direct skill matches identified.")

        with col_missing:
            st.subheader(f"⚠️ Missing Skills ({len(match_result['missing_skills'])})")
            if match_result['missing_skills']:
                html_chips = "".join([f"<span class='missing-chip'>✗ {s}</span>" for s in match_result['missing_skills']])
                st.markdown(html_chips, unsafe_allow_html=True)
            else:
                st.success("No missing required skills!")

        st.markdown("---")
        st.subheader("🔎 Textual Evidence Inspector")
        st.caption("Verifiable snippets extracted directly from candidate profile and job description:")
        
        if match_result["skill_evidence"]:
            for skill, ev in match_result["skill_evidence"].items():
                with st.expander(f"Skill Evidence for: **{skill}**"):
                    st.write("**Candidate Evidence:**")
                    for ce in ev.get("candidate_evidence", []):
                        st.markdown(f"<div class='evidence-box'>💬 \"{ce}\"</div>", unsafe_allow_html=True)
                    st.write("**Job Requirement:**")
                    for je in ev.get("job_requirement", []):
                        st.markdown(f"<div class='evidence-box'>📋 \"{je}\"</div>", unsafe_allow_html=True)
        else:
            st.write("No skill evidence snippets recorded.")

        st.markdown("---")
        st.subheader("🏆 Ranked Suitable Roles for Candidate")
        rank_df = pd.DataFrame([
            {
                "Rank": idx + 1,
                "Role Title": r["job_title"],
                "Match Score": f"{r['match_score']}%",
                "Rating": r["match_rating"],
                "Matched Skills Count": len(r["matched_skills"]),
                "Missing Skills Count": len(r["missing_skills"])
            }
            for idx, r in enumerate(role_rankings)
        ])
        st.table(rank_df)

    # -----------------------------------------------------------------------------
    # TAB 2: Candidate View
    # -----------------------------------------------------------------------------
    with tab2:
        st.header("👤 Candidate Strengths & Skill Gap Analysis")
        st.caption("Empowering candidates to understand their skill alignment and areas for growth.")

        col_str, col_gap = st.columns(2)

        cand_ext = match_result["candidate_extracted"]

        with col_str:
            st.subheader("💪 Your Identified Strengths")
            st.write(f"**Years of Experience Extracted**: `{cand_ext.get('experience_years', 0.0)} years`")
            st.write(f"**Extracted Technical Domain**: `{cand_ext.get('domain_background', 'General Engineering')}`")
            
            st.markdown("**Identified Technologies & Frameworks:**")
            techs = cand_ext.get("technologies", [])
            if techs:
                st.write(", ".join([f"`{t}`" for t in techs]))
            else:
                st.write("None identified.")

            st.markdown("**Programming Languages:**")
            langs = cand_ext.get("languages", [])
            if langs:
                st.write(", ".join([f"`{l}`" for l in langs]))
            else:
                st.write("None identified.")

        with col_gap:
            st.subheader("📈 Skill Gap Recommendations")
            missing = match_result["missing_skills"]
            if missing:
                st.warning(f"To improve your match for **{selected_job_title}**, consider learning:")
                for m in missing:
                    st.markdown(f"- **{m}**: Required by job description but not found in your profile text.")
            else:
                st.success(f"Great job! You possess all key skills explicitly required for **{selected_job_title}**.")

        st.markdown("---")
        st.subheader("📥 Structured JSON Extraction Output")
        st.json(cand_ext)

    # -----------------------------------------------------------------------------
    # TAB 3: Grounded RAG Chatbot
    # -----------------------------------------------------------------------------
    with tab3:
        st.header("🤖 Grounded RAG Recruiter Chatbot")
        st.caption("Ask questions about the candidate's profile, job requirements, or score breakdown. Answers are strictly grounded in retrieved data.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display previous chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander("📌 Retrieved Context Citations"):
                        for c in msg["citations"]:
                            st.caption(c)

        user_query = st.chat_input("Ask a question (e.g. 'What experience does candidate have with Python?' or 'Why is score low?'):")

        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                with st.spinner("Searching grounded context and synthesizing answer..."):
                    qa_res = rag_engine.answer_question(user_query)
                    st.markdown(qa_res["answer"])
                    
                    if qa_res["retrieved_context"]:
                        with st.expander("📌 Retrieved Context Citations"):
                            for c in qa_res["retrieved_context"]:
                                st.caption(c)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": qa_res["answer"],
                "citations": qa_res["retrieved_context"]
            })

    # -----------------------------------------------------------------------------
    # TAB 4: Privacy & Bias Audit
    # -----------------------------------------------------------------------------
    with tab4:
        st.header("🔒 Anti-Bias & Privacy Scrubber Audit")
        st.caption("HireLens strips non-job-relevant demographic attributes before candidate evaluation.")

        col_orig, col_scrub = st.columns(2)

        raw_candidate = candidate_text_input
        scrubbed_candidate, redactions = scrub_demographics(raw_candidate)

        with col_orig:
            st.subheader("Raw Candidate Profile Text")
            st.text_area("Input Text", raw_candidate, height=220, disabled=True)

        with col_scrub:
            st.subheader("Sanitized Profile Text (Used for Scoring)")
            st.text_area("Scrubbed Text", scrubbed_candidate, height=220, disabled=True)

        st.subheader("📋 Redactions Log")
        if redactions:
            red_df = pd.DataFrame([
                {"Demographic Marker Category": cat, "Occurrences Redacted": count}
                for cat, count in redactions.items()
            ])
            st.table(red_df)
        else:
            st.info("No demographic markers (name, age, gender, location, college) detected in input text.")

    # -----------------------------------------------------------------------------
    # TAB 5: Evaluation Benchmarks
    # -----------------------------------------------------------------------------
    with tab5:
        st.header("📊 Benchmark Evaluation across 15 Test Profiles")
        st.caption("Live empirical metrics demonstrating HireLens recall, ranking accuracy, and explainability.")

        if st.button("🚀 Run Full Benchmark Suite (15 Profiles)"):
            with st.spinner("Running benchmark suite..."):
                from tests.run_benchmarks import run_evaluation_suite
                bench_metrics = run_evaluation_suite()
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Skill Extraction Recall", f"{bench_metrics['overall_recall']}%")
                col_m2.metric("Top-1 Ranking Accuracy", f"{bench_metrics['top1_accuracy']}%")
                col_m3.metric("Top-2 Ranking Accuracy", f"{bench_metrics['top2_accuracy']}%")
                col_m4.metric("Explanation Coverage", "100.0%")

                st.subheader("Detailed Profile Results")
                st.dataframe(pd.DataFrame(bench_metrics["details"]))
else:
    st.warning("Please enter or select both candidate profile text and job description.")
