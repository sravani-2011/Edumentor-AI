"""
app.py – EduMentor AI: Adaptive Learning Chatbot (Streamlit Entry Point)

Run with:  streamlit run app.py

A 4-tab application providing:
  Tab 1 (Setup)        – API key, PDF upload, vector store management, learner profile
  Tab 2 (Chat Tutor)   – RAG-powered chat with citations and follow-ups
  Tab 3 (Practice Quiz) – MCQ/short-answer quiz with auto-grading
  Tab 4 (Insights)     – Evaluation dashboard with charts and export
"""

import streamlit as st
import os, sys, time

# ---------------------------------------------------------------------------
# Ensure project root is on the Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.ingest import ingest_pdfs, ingest_wikipedia_stub, clear_vector_store
from rag.retriever import retrieve
from rag.chain import get_rag_answer
from tutor.personalize import LearnerProfile
from tutor.quiz import generate_quiz
from tutor.grader import grade_quiz
from tutor.flashcards import generate_flashcards
from tutor.summarizer import generate_summary
from eval.metrics import compute_rouge_l, compute_bleu
from eval.logger import create_log_entry, export_logs_csv, export_logs_json
from utils.config import get_gemini_api_key, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from auth import register_user, login_user, update_score, get_leaderboard
from tools.video_summarizer import summarize_video
from tools.blog_summarizer import summarize_blog
from tools.mindmap import generate_mindmap, generate_concept_tree
from tools.audio_summarizer import summarize_audio
from tools.coding_challenge import generate_coding_challenge, evaluate_solution
from tools.visual_explainer import generate_visual_explanation
from tools.expert_mentor import get_mentor_guidance
from tools.problem_bank import get_curated_problems
import json as json_lib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduMentor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# ✨ PREMIUM MIDNIGHT GOLD CSS — Minor Project Showcase Edition
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, .hero-title {
        font-family: 'Fraunces', serif;
    }

    /* ── Deep Charcoal Base ────────────────────────────── */
    .stApp {
        background: #0a0a0c;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(212, 175, 55, 0.03) 0%, transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(212, 175, 55, 0.02) 0%, transparent 40%);
        color: #e0e0e0;
    }
    .stApp > header { background: transparent !important; }

    /* ── Classy Hero Header ───────────────────────────── */
    @keyframes subtleReveal {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hero {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(212, 175, 55, 0.15);
        backdrop-filter: blur(20px);
        padding: 3rem 2.8rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        color: #f8f8f8;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
        position: relative;
        animation: subtleReveal 1.2s ease-out;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 150px; height: 150px;
        background: radial-gradient(circle, rgba(212,175,55,0.05) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: inline-block;
        filter: drop-shadow(0 0 10px rgba(212,175,55,0.2));
    }
    .hero h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        background: linear-gradient(135deg, #f8f8f8 30%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero .tagline {
        margin: 0.8rem 0 0 0;
        color: #b0b0b0;
        font-size: 1.1rem;
        font-weight: 400;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .hero .badges {
        display: flex;
        gap: 0.8rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    .hero .badge {
        background: rgba(212, 175, 55, 0.08);
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #d4af37;
        border: 1px solid rgba(212, 175, 55, 0.2);
        letter-spacing: 0.02em;
    }

    /* ── Section cards ────────────────────────────────── */
    .section-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    .section-card:hover {
        border-color: rgba(212, 175, 55, 0.2);
        background: rgba(255, 255, 255, 0.03);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    .section-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* ── Tab Bar (Minimalist) ─────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255,255,255,0.01);
        padding: 10px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.95rem;
        color: #808080 !important;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(212, 175, 55, 0.12) !important;
        color: #d4af37 !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
    }

    /* ── Classy Buttons ───────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1a1a1c 0%, #0a0a0c 100%) !important;
        color: #d4af37 !important;
        border: 1px solid rgba(212, 175, 55, 0.4) !important;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 0.7rem 1.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .stButton > button:hover {
        background: #d4af37 !important;
        color: #0a0a0c !important;
        box-shadow: 0 0 25px rgba(212, 175, 55, 0.3);
    }

    /* ── Inputs ───────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #121214 !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 10px;
        color: #e0e0e0 !important;
        padding: 12px;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #d4af37 !important;
        box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.1) !important;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.01);
        border: 1px solid rgba(212,175,55,0.1);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
    }
    .metric-card h3 {
        font-size: 2.2rem;
        color: #d4af37;
        margin: 0;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.78rem;
        font-weight: 600;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Expanders ─────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #d4af37 !important;
    }
    details {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        transition: all 0.2s ease;
    }
    details:hover {
        border-color: rgba(102, 126, 234, 0.35);
    }
    details summary {
        cursor: pointer;
        font-weight: 600;
        color: #b8c5ff;
    }

    /* ── Quiz result colors ───────────────────────────── */
    .quiz-correct { color: #43e97b; font-weight: 700; }
    .quiz-wrong   { color: #ff6b6b; font-weight: 700; }
    .quiz-partial { color: #ffd93d; font-weight: 700; }

    /* ── Feature grid (Setup welcome) ─────────────────── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    .feature-item {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.2rem;
        transition: all 0.25s ease;
    }
    .feature-item:hover {
        border-color: rgba(102, 126, 234, 0.25);
        transform: translateY(-2px);
    }
    .feature-item .f-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
    .feature-item h4 {
        margin: 0.3rem 0 0.2rem 0;
        font-size: 0.92rem;
        font-weight: 700;
        color: #c5cafe;
    }
    .feature-item p {
        margin: 0;
        font-size: 0.8rem;
        color: #888888;
        line-height: 1.4;
    }

    /* ── Status pill ──────────────────────────────────── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.8rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(67, 233, 123, 0.12);
        color: #43e97b;
        border: 1px solid rgba(67, 233, 123, 0.25);
    }
    .status-waiting {
        background: rgba(255, 107, 107, 0.12);
        color: #ff6b6b;
        border: 1px solid rgba(255, 107, 107, 0.25);
    }

    /* ── Sidebar ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0c 0%, #16161a 100%) !important;
        border-right: 1px solid rgba(212,175,55,0.1);
    }
    section[data-testid="stSidebar"] * {
        color: #d1d1d1 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] span {
        color: #d1d1d1 !important;
    }

    /* ── Streamlit metrics override ───────────────────── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(212,175,55,0.1);
        border-radius: 14px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: #888888 !important; }
    [data-testid="stMetricValue"] { color: #d4af37 !important; }

    /* ── Dividers ──────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212,175,55,0.15), transparent);
    }

    /* ── Alerts / info boxes ──────────────────────────── */
    .stAlert {
        border-radius: 14px;
        border: none;
    }

    /* ── Subheaders ───────────────────────────────────── */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: #e0e0e0 !important;
    }

    /* ── Dataframe ────────────────────────────────────── */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
    }

    /* ── Toggle ───────────────────────────────────────── */
    .stToggle > label > span { color: #c5cafe !important; }

    /* ── Hide Streamlit furniture ──────────────────────── */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ── Download buttons ─────────────────────────────── */
    .stDownloadButton > button {
        background: rgba(212,175,55,0.08) !important;
        border: 1px solid rgba(212,175,55,0.2) !important;
        color: #d4af37 !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(212,175,55,0.15) !important;
        box-shadow: 0 4px 16px rgba(212, 175, 55, 0.15);
    }

    /* ── Radio buttons ────────────────────────────────── */
    .stRadio > div { color: #d1d1d1 !important; }
    .stRadio label span { color: #d1d1d1 !important; }

    /* ── Caption text ─────────────────────────────────── */
    .stCaption, small { color: #888888 !important; }

    /* ── Chat input ───────────────────────────────────── */
    .stChatInput > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px;
    }
    .stChatInput textarea {
        color: #e0e0e0 !important;
    }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideLeft { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
.step-card {
    background: #16161a;
    padding: 15px;
    border-left: 4px solid #d4af37;
    margin-bottom: 10px;
    border-radius: 4px;
}
.fade-in { animation: fadeIn 0.8s ease-out forwards; }
.slide-left { animation: slideLeft 0.8s ease-out forwards; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "api_key": "",
    "learner_profile": LearnerProfile(),
    "chat_history": [],         # [{role, content}]
    "last_chunks": [],          # last retrieved chunks for quiz
    "quiz_data": None,          # current quiz questions
    "quiz_answers": {},         # user answers keyed by question id
    "quiz_results": None,       # grading results
    "logs": [],                 # evaluation logs
    "course_id": "General",
    "language": "English",      # response language
    "flashcards": [],           # generated flashcards
    "summary": "",              # generated summary
    "logged_in": False,         # auth state
    "username": "",             # current user
    "current_challenge": None,  # active coding challenge
    "challenge_start": None,    # challenge timer start
    "mentor_history": [],       # Expert Mentor chat history
    "visual_explain": None,     # current visual explanation data
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# 🔐 LOGIN / REGISTER PAGE
# ---------------------------------------------------------------------------
if not st.session_state.logged_in:
    st.markdown("""
    <div class="hero">
        <span class="hero-icon">🎓</span>
        <h1>EduMentor AI</h1>
        <p class="tagline">Your Personal AI Tutor — Adaptive Learning Powered by RAG Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "📝 Register"])
    with auth_tab1:
        st.markdown("### Welcome Back!")
        login_user_input = st.text_input("Username", key="login_username")
        login_pass_input = st.text_input("Password", type="password", key="login_password")
        if st.button("🔓 Login", use_container_width=True):
            result = login_user(login_user_input, login_pass_input)
            if result.get("ok"):
                st.session_state.logged_in = True
                st.session_state.username = login_user_input
                st.rerun()
            else:
                st.error(result.get("error", "Login failed."))

    with auth_tab2:
        st.markdown("### Create an Account")
        reg_user = st.text_input("Choose a Username", key="reg_username")
        reg_pass = st.text_input("Choose a Password", type="password", key="reg_password")
        reg_pass2 = st.text_input("Confirm Password", type="password", key="reg_password2")
        if st.button("✅ Register", use_container_width=True):
            if reg_pass != reg_pass2:
                st.error("Passwords do not match.")
            else:
                result = register_user(reg_user, reg_pass)
                if result.get("ok"):
                    st.success("Account created! Please log in.")
                else:
                    st.error(result.get("error", "Registration failed."))

    st.stop()  # Don't show main app until logged in

# ---------------------------------------------------------------------------
# ✨ Hero Header (logged in)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <span class="hero-icon">🎓</span>
    <h1>EduMentor AI</h1>
    <p class="tagline">Welcome, {st.session_state.username}! — Adaptive Learning Powered by RAG</p>
    <div class="badges">
        <span class="badge">🤖 Gemini 2.0 Flash</span>
        <span class="badge">🔍 ChromaDB RAG</span>
        <span class="badge">📊 Auto-Evaluation</span>
        <span class="badge">🧠 Adaptive Quizzes</span>
        <span class="badge">🏆 Competitive Coding</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
(tab_setup, tab_chat, tab_quiz, tab_code, tab_flash, tab_summary,
 tab_video, tab_blog, tab_mindmap, tab_audio, tab_visual, tab_leader, tab_insights) = st.tabs([
    "⚙️ Setup", "💬 Chat Tutor", "📝 Quiz", "🏆 CP Hub",
    "📇 Flashcards", "📋 Summary", "🎥 Video", "📰 Blog",
    "🧠 Mind Map", "🎧 Audio", "🎨 Visual Explainer", "🏅 Leaderboard", "📊 Insights",
])


# =====================================================================
# TAB 1 – SETUP
# =====================================================================
with tab_setup:
    col_left, col_right = st.columns([1, 1], gap="large")

    # --- Left column: API key + PDF upload ---
    with col_left:
        # API Key Section
        st.markdown("""
        <div class="section-card">
            <h3>🔑 Google Gemini Configuration</h3>
        </div>
        """, unsafe_allow_html=True)

        api_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Get your free key at aistudio.google.com/apikey — stored in session only.",
        )
        if api_input:
            st.session_state.api_key = api_input

        # Resolve key (session → .env fallback)
        resolved_key = get_gemini_api_key(st.session_state.api_key)
        if resolved_key:
            st.markdown('<span class="status-pill status-ready">● API Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill status-waiting">● Awaiting API Key</span>', unsafe_allow_html=True)

        st.divider()

        # PDF Upload Section
        st.markdown("""
        <div class="section-card">
            <h3>📂 Course Material Upload</h3>
        </div>
        """, unsafe_allow_html=True)

        course_id = st.text_input("Course / Subject", value=st.session_state.course_id)
        st.session_state.course_id = course_id

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload course notes, slides (exported to PDF), or handouts.",
        )

        col_build, col_clear = st.columns(2)
        with col_build:
            if st.button("🔨 Build / Update Knowledge Base", use_container_width=True):
                if not resolved_key:
                    st.error("Please enter your Gemini API key first.")
                elif not uploaded_files:
                    st.warning("Please upload at least one PDF file.")
                else:
                    with st.spinner("Ingesting PDFs and building vector store…"):
                        try:
                            result = ingest_pdfs(
                                uploaded_files=uploaded_files,
                                api_key=resolved_key,
                                course_id=course_id,
                            )
                            st.success(
                                f"✅ Done! Ingested **{result['ingested']}** file(s), "
                                f"skipped **{result['skipped']}** (unchanged), "
                                f"created **{result['total_chunks']}** chunks."
                            )
                        except Exception as e:
                            st.error(f"⚠️ Ingestion Error: {e}")
                            st.info("💡 Check your Gemini API key and try again.")

        with col_clear:
            if st.button("🗑️ Clear Vector Store", use_container_width=True):
                clear_vector_store()
                st.info("Vector store cleared.")

        st.divider()

        # Wikipedia Section
        st.markdown("""
        <div class="section-card">
            <h3>🌐 Wikipedia EDU (Optional)</h3>
        </div>
        """, unsafe_allow_html=True)

        wiki_toggle = st.toggle("Enable Wikipedia EDU dump ingestion", value=False)
        if wiki_toggle:
            st.info(
                "**Stub feature**: Paste extracted Wikipedia text below, or "
                "point to a `.txt` file. For production, implement a proper "
                "article parser."
            )
            wiki_text = st.text_area("Paste Wikipedia text content", height=120)
            if st.button("Ingest Wikipedia Text") and wiki_text and resolved_key:
                with st.spinner("Ingesting Wikipedia content…"):
                    w_result = ingest_wikipedia_stub(wiki_text, resolved_key, course_id)
                st.success(f"Ingested {w_result['ingested_chunks']} chunks from Wikipedia.")

    # --- Right column: Learner Profile + Feature Showcase ---
    with col_right:
        st.markdown("""
        <div class="section-card">
            <h3>👤 Learner Profile</h3>
        </div>
        """, unsafe_allow_html=True)

        lp = st.session_state.learner_profile
        lp.name = st.text_input("Your Name", value=lp.name)
        lp.course = st.text_input("Course", value=lp.course)
        lp.skill_level = st.selectbox(
            "Skill Level",
            ["Beginner", "Intermediate", "Advanced"],
            index=["Beginner", "Intermediate", "Advanced"].index(lp.skill_level),
            help="This controls how EduMentor adapts its explanations.",
        )
        lp.goals = st.text_area("Learning Goals", value=lp.goals, height=80)
        st.session_state.learner_profile = lp

        # Language selector
        languages = ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam",
                      "Bengali", "Marathi", "Gujarati", "Spanish", "French", "German"]
        lang_idx = languages.index(st.session_state.language) if st.session_state.language in languages else 0
        st.session_state.language = st.selectbox(
            "🌍 Response Language", languages, index=lang_idx,
            help="AI will respond in this language."
        )

        # Profile summary metrics
        st.divider()
        st.markdown("##### 🎯 Profile Summary")
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Skill Level", lp.skill_level)
        pcol2.metric("Concepts Asked", len(lp.concepts_asked))

        if lp.weak_concepts:
            st.warning(f"⚠️ Weak areas detected: {', '.join(lp.weak_concepts)}")

        if lp.quiz_scores:
            latest = lp.quiz_scores[-1]
            st.info(f"📊 Last quiz: {latest['percentage']}% on *{latest['concept']}*")

        # Progress tracking (save / load)
        st.divider()
        st.markdown("##### 💾 Progress Tracking")
        prog_col1, prog_col2 = st.columns(2)
        with prog_col1:
            progress_data = {
                "name": lp.name, "course": lp.course,
                "skill_level": lp.skill_level, "goals": lp.goals,
                "concepts_asked": lp.concepts_asked,
                "quiz_scores": lp.quiz_scores,
                "language": st.session_state.language,
            }
            st.download_button(
                "📥 Download Progress",
                data=json_lib.dumps(progress_data, indent=2),
                file_name="edumentor_progress.json",
                mime="application/json",
                use_container_width=True,
            )
        with prog_col2:
            uploaded_progress = st.file_uploader("📤 Upload Progress", type=["json"], key="progress_upload")
            if uploaded_progress:
                try:
                    loaded = json_lib.loads(uploaded_progress.read())
                    lp.name = loaded.get("name", lp.name)
                    lp.course = loaded.get("course", lp.course)
                    lp.skill_level = loaded.get("skill_level", lp.skill_level)
                    lp.goals = loaded.get("goals", lp.goals)
                    lp.concepts_asked = loaded.get("concepts_asked", [])
                    lp.quiz_scores = loaded.get("quiz_scores", [])
                    st.session_state.language = loaded.get("language", "English")
                    st.session_state.learner_profile = lp
                    st.success("✅ Progress restored!")
                except Exception:
                    st.error("Invalid progress file.")

        # Feature showcase
        st.divider()
        st.markdown("""
        <div class="section-card">
            <h3>✨ What EduMentor Can Do</h3>
        </div>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="f-icon">📄</div>
                <h4>Smart PDF Ingestion</h4>
                <p>Upload course PDFs with auto-chunking, deduplication, and semantic indexing.</p>
            </div>
            <div class="feature-item">
                <div class="f-icon">🧠</div>
                <h4>Adaptive Tutoring</h4>
                <p>Explanations adapt to your skill level — beginner to advanced.</p>
            </div>
            <div class="feature-item">
                <div class="f-icon">📝</div>
                <h4>Auto-Generated Quizzes</h4>
                <p>MCQ and short-answer questions generated from your study material.</p>
            </div>
            <div class="feature-item">
                <div class="f-icon">📊</div>
                <h4>Learning Analytics</h4>
                <p>Track progress, identify weak areas, and export detailed reports.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# TAB 2 – CHAT TUTOR
# =====================================================================
with tab_chat:
    resolved_key = get_gemini_api_key(st.session_state.api_key)

    # Sidebar controls (rendered in sidebar for this tab)
    with st.sidebar:
        st.markdown("### 🎛️ Chat Controls")
        top_k = st.slider("Top-K Results", 1, 15, TOP_K, help="Number of chunks to retrieve.")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05, help="LLM creativity.")
        explain_simply = st.toggle("🧒 Explain Like I'm 12", value=False)
        verbosity = st.slider("Verbosity", 1, 10, 5, help="1 = concise, 10 = detailed.")

        # Multi-modal: image upload
        st.markdown("---")
        st.markdown("### 📷 Image Input")
        uploaded_img = st.file_uploader("Upload image to ask about", type=["png", "jpg", "jpeg", "webp"],
                                         key="sidebar_img")
        if uploaded_img:
            st.image(uploaded_img, caption="Uploaded image", use_container_width=True)

        # Learner profile summary
        st.markdown("---")
        lp = st.session_state.learner_profile
        st.markdown(
            f"👤 **{lp.name}** | 📚 **{lp.skill_level}** | 🎓 {lp.course}"
        )

        # Clear chat history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_chunks = []
            st.rerun()

        # Logout button
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 🎤 Voice Input (copy-paste) ---
    voice_html = """
    <div id="voice-container" style="text-align:center;">
        <button id="voice-btn" onclick="startVoice()" style="
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white; border: none; padding: 10px 20px;
            border-radius: 10px; cursor: pointer; font-size: 15px;
            transition: all 0.3s; display:inline-flex; align-items:center; gap:6px;">
            🎤 Tap to Speak
        </button>
        <span id="voice-status" style="margin-left:10px; font-size:0.85em; color:#aaa;"></span>
        <textarea id="voice-result" readonly style="
            width:100%; min-height:50px; margin-top:8px;
            background: #1a1a2e; color: #e0e0e0; border: 1px solid #333;
            border-radius: 8px; padding: 8px; font-size: 14px;
            display:none; resize:vertical;
        "></textarea>
    </div>
    <script>
    function startVoice() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            document.getElementById('voice-status').textContent = '❌ Not supported in this browser';
            return;
        }
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        const btn = document.getElementById('voice-btn');
        const status = document.getElementById('voice-status');
        const result = document.getElementById('voice-result');
        btn.innerHTML = '🔴 Listening...';
        btn.style.background = 'linear-gradient(135deg, #ef4444, #b91c1c)';
        status.textContent = 'Speak now...';
        recognition.start();
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            result.style.display = 'block';
            result.value = transcript;
            result.select();
            status.textContent = '✅ Copy the text above and paste into the chat!';
            btn.innerHTML = '🎤 Tap to Speak';
            btn.style.background = '#d4af37';
            btn.style.color = '#0a0a0c';
        };
        recognition.onerror = function(event) {
            status.textContent = '❌ Error: ' + event.error;
            btn.innerHTML = '🎤 Tap to Speak';
            btn.style.background = '#d4af37';
            btn.style.color = '#0a0a0c';
        };
        recognition.onend = function() {
            btn.innerHTML = '🎤 Tap to Speak';
            btn.style.background = '#d4af37';
            btn.style.color = '#0a0a0c';
        };
    }
    </script>
    """
    st.components.v1.html(voice_html, height=120)

    # Chat input — also check for pending follow-up questions
    chat_input = st.chat_input("Ask EduMentor anything about your course…")
    prompt = chat_input or st.session_state.pop("pending_followup", None)

    if prompt:
        if not resolved_key:
            st.error("Please set your Gemini API key in the Setup tab first.")
        else:
            # Show user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Retrieve + Generate
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    # Retrieve
                    try:
                        retrieval = retrieve(prompt, resolved_key, top_k=top_k)
                        chunks = retrieval["chunks"]
                        is_confident = retrieval["is_confident"]
                    except Exception as e:
                        st.error(f"⚠️ Retrieval Error: {e}")
                        st.info("💡 Make sure you've uploaded PDFs first and your API key is valid.")
                        chunks = []
                        is_confident = False
                    st.session_state.last_chunks = chunks

                    # Record concept
                    lp = st.session_state.learner_profile
                    lp.record_concept(prompt[:60])

                    # Generate answer
                    try:
                        # Handle multi-modal image input if available
                        img_bytes = None
                        img_type = "image/jpeg"
                        if uploaded_img:
                            img_bytes = uploaded_img.getvalue()
                            img_type = uploaded_img.type

                        answer = get_rag_answer(
                            question=prompt,
                            chunks=chunks,
                            is_confident=is_confident,
                            learner_profile=lp.to_dict(),
                            api_key=resolved_key,
                            temperature=temperature,
                            explain_simply=explain_simply,
                            verbosity=verbosity,
                            language=st.session_state.language,
                            image_bytes=img_bytes,
                            image_mime=img_type,
                        )
                    except Exception as e:
                        answer = None
                        st.error(f"⚠️ Gemini API Error: {e}")
                        st.info("💡 **Troubleshooting tips:**\n"
                                "1. Check that your Gemini API key is valid at [aistudio.google.com](https://aistudio.google.com)\n"
                                "2. Make sure you haven't exceeded the free tier rate limit (15 req/min)\n"
                                "3. Try again in a few seconds")

                if answer:
                    st.markdown(answer)

                # Citations section
                if chunks:
                    st.markdown("---")
                    st.markdown("##### 📚 Source Citations")
                    for i, chunk in enumerate(chunks, 1):
                        meta = chunk.get("metadata", {})
                        with st.expander(
                            f"📄 {meta.get('source', 'Unknown')} – Page {meta.get('page', '?')} "
                            f"(Score: {chunk.get('score', 'N/A')})"
                        ):
                            st.markdown(chunk["content"][:500])

            # Save assistant message and compute metrics (only if answer succeeded)
            if answer:
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                # Generate dynamic follow-up questions using the AI
                try:
                    genai.configure(api_key=resolved_key)
                    followup_model = genai.GenerativeModel("gemini-2.0-flash")
                    followup_resp = followup_model.generate_content(
                        f"Based on this Q&A, suggest exactly 3 short follow-up questions a student might ask next. "
                        f"Return ONLY the 3 questions, one per line, no numbering, no bullets.\n\n"
                        f"Question: {prompt}\nAnswer summary: {answer[:300]}",
                        generation_config={"max_output_tokens": 150, "temperature": 0.7},
                    )
                    lines = [l.strip() for l in followup_resp.text.strip().split("\n") if l.strip()]
                    st.session_state.last_follow_ups = lines[:3] if len(lines) >= 3 else [
                        f"Can you explain more about {prompt[:40]}?",
                        f"What are the practical applications of this?",
                        f"What should I study next after this topic?",
                    ]
                except Exception:
                    st.session_state.last_follow_ups = [
                        f"Can you explain more about {prompt[:40]}?",
                        f"What are the practical applications of this?",
                        f"What should I study next after this topic?",
                    ]

    # --- Persistent follow-up buttons (always show for last response) ---
    def _set_followup(question: str):
        st.session_state.pending_followup = question

    if st.session_state.get("last_follow_ups"):
        st.markdown("---")
        st.markdown("##### 🚀 Follow-up Questions")
        fcols = st.columns(len(st.session_state.last_follow_ups))
        for idx, (fc, fu) in enumerate(zip(fcols, st.session_state.last_follow_ups)):
            with fc:
                st.button(
                    fu,
                    key=f"followup_{idx}_{len(st.session_state.chat_history)}",
                    on_click=_set_followup,
                    args=(fu,),
                )

    # --- Compute evaluation metrics and log (only when a new answer was just generated) ---
    if prompt and answer:
        context_text = " ".join([c["content"] for c in chunks])
        rouge = compute_rouge_l(answer, context_text) if context_text else {"f1": 0}
        bleu_result = compute_bleu(answer, context_text) if context_text else {"bleu": 0}

        log_entry = create_log_entry(
            query=prompt,
            answer=answer,
            retrieval_scores=[c.get("score", 0) for c in chunks],
            rouge_l=rouge["f1"],
            bleu=bleu_result["bleu"],
            is_confident=is_confident,
        )
        st.session_state.logs.append(log_entry)


# =====================================================================
# TAB 3 – PRACTICE QUIZ
# =====================================================================
with tab_quiz:
    resolved_key = get_gemini_api_key(st.session_state.api_key)

    st.markdown("""
    <div class="section-card">
        <h3>📝 Practice Quiz</h3>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Generate a quiz from your last chat topic to reinforce learning.")

    if not resolved_key:
        st.warning("Set your Gemini API key in the Setup tab first.")
    elif not st.session_state.last_chunks:
        st.info("💡 Ask a question in the Chat Tutor tab first, then come here to quiz yourself!")
    else:
        # Quiz controls: two columns like the screenshot
        quiz_left, quiz_right = st.columns([2, 1])
        with quiz_left:
            if st.button("🎲 Generate Quiz from Last Chat Topic", use_container_width=True):
                st.session_state._generate_quiz = True
        with quiz_right:
            num_q = st.number_input("Number of questions", min_value=3, max_value=10, value=5, step=1)
            difficulty = st.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"])

        if getattr(st.session_state, "_generate_quiz", False):
            st.session_state._generate_quiz = False
            with st.spinner("Generating quiz from your last topic…"):
                lp = st.session_state.learner_profile
                # Map difficulty to skill_level override for the prompt
                diff_map = {"Easy": "Beginner", "Medium": "Intermediate", "Hard": "Advanced", "Mixed": lp.skill_level}
                quiz = generate_quiz(
                    chunks=st.session_state.last_chunks,
                    api_key=resolved_key,
                    skill_level=diff_map[difficulty],
                    num_questions=num_q,
                )
            if "error" in quiz:
                st.error(f"Quiz generation failed: {quiz['error']}")
            else:
                st.session_state.quiz_data = quiz
                st.session_state.quiz_answers = {}
                st.session_state.quiz_results = None

        # Display quiz questions
        if st.session_state.quiz_data and "questions" in st.session_state.quiz_data:
            quiz = st.session_state.quiz_data
            st.markdown(f"**Topic:** {quiz.get('quiz_topic', 'General')}")
            st.divider()

            questions = quiz["questions"]
            for q in questions:
                qid = q["id"]
                st.markdown(f"**Q{qid}** ({q.get('difficulty', '?')}) – {q['question']}")

                if q["type"] == "MCQ" and q.get("options"):
                    options = q["options"]
                    answer = st.radio(
                        f"Select answer for Q{qid}:",
                        options,
                        key=f"quiz_q_{qid}",
                        label_visibility="collapsed",
                    )
                    st.session_state.quiz_answers[qid] = answer
                else:
                    answer = st.text_input(
                        f"Your answer for Q{qid}:",
                        key=f"quiz_q_{qid}",
                    )
                    st.session_state.quiz_answers[qid] = answer

                st.markdown("---")

            # Submit and grade
            col_submit, col_retry = st.columns(2)
            with col_submit:
                if st.button("✅ Submit & Grade", use_container_width=True):
                    user_ans = [
                        st.session_state.quiz_answers.get(q["id"], "")
                        for q in questions
                    ]
                    with st.spinner("Grading your answers…"):
                        results = grade_quiz(questions, user_ans, resolved_key)
                    st.session_state.quiz_results = results

                    # Record scores in learner profile
                    lp = st.session_state.learner_profile
                    topic = quiz.get("quiz_topic", "General")
                    total = results.get("total_score", 0)
                    max_total = results.get("max_total", len(questions))
                    lp.record_quiz_score(topic, total, max_total)

                    # Log quiz score
                    if st.session_state.logs:
                        st.session_state.logs[-1]["quiz_score"] = total
                        st.session_state.logs[-1]["quiz_max"] = max_total

            with col_retry:
                if st.button("🔄 Retry Quiz", use_container_width=True):
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_results = None
                    st.rerun()

        # Display grading results
        if st.session_state.quiz_results:
            results = st.session_state.quiz_results
            st.divider()

            st.markdown("""
            <div class="section-card">
                <h3>📊 Quiz Results</h3>
            </div>
            """, unsafe_allow_html=True)

            total = results.get("total_score", 0)
            max_total = results.get("max_total", 1)
            pct = round((total / max_total) * 100, 1) if max_total > 0 else 0

            # Score display
            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.metric("Score", f"{total}/{max_total}")
            rcol2.metric("Percentage", f"{pct}%")
            rcol3.metric("Grade", "A" if pct >= 90 else "B" if pct >= 75 else "C" if pct >= 60 else "D" if pct >= 40 else "F")

            # Per-question feedback
            for r in results.get("results", []):
                qid = r["question_id"]
                score = r.get("score", 0)
                mx = r.get("max_score", 1)
                is_correct = r.get("is_correct", False)

                if is_correct:
                    icon = "✅"
                    css = "quiz-correct"
                elif score > 0:
                    icon = "🟡"
                    css = "quiz-partial"
                else:
                    icon = "❌"
                    css = "quiz-wrong"

                st.markdown(f"{icon} **Q{qid}**: <span class='{css}'>{score}/{mx}</span>", unsafe_allow_html=True)
                st.markdown(f"  💬 {r.get('feedback', '')}")
                if r.get("hint"):
                    st.markdown(f"  💡 **Hint:** {r['hint']}")
                if not is_correct:
                    st.markdown(f"  📖 **Correct answer:** {r.get('correct_answer', 'N/A')}")

            st.markdown(f"**Overall:** {results.get('overall_feedback', '')}")

            # Recommended reading
            if st.session_state.last_chunks:
                st.markdown("##### 📚 Recommended Reading")
                seen = set()
                for chunk in st.session_state.last_chunks[:3]:
                    meta = chunk.get("metadata", {})
                    source = meta.get("source", "Unknown")
                    page = meta.get("page", "?")
                    key = f"{source}_p{page}"
                    if key not in seen:
                        st.markdown(f"- **{source}**, Page {page}")
                        seen.add(key)


# =====================================================================
# TAB 4 – FLASHCARDS
# =====================================================================
with tab_flash:
    st.markdown("""
    <div class="section-card">
        <h3>📇 Flashcard Generator</h3>
        <p>Auto-generate study flashcards from your uploaded PDFs. Click a card to reveal the answer!</p>
    </div>
    """, unsafe_allow_html=True)

    resolved_key = get_gemini_api_key(st.session_state.api_key)

    if not resolved_key:
        st.warning("🔑 Please set your API key in the Setup tab first.")
    elif not st.session_state.last_chunks:
        st.info("💬 Ask at least one question in Chat Tutor first, so EduMentor has context for flashcards.")
    else:
        fc_col1, fc_col2 = st.columns([2, 1])
        with fc_col1:
            num_cards = st.slider("Number of flashcards", 5, 20, 10)
        with fc_col2:
            if st.button("🃏 Generate Flashcards", use_container_width=True):
                with st.spinner("Creating flashcards..."):
                    try:
                        cards = generate_flashcards(
                            chunks=st.session_state.last_chunks,
                            api_key=resolved_key,
                            count=num_cards,
                        )
                        st.session_state.flashcards = cards
                    except Exception as e:
                        st.error(f"Error generating flashcards: {e}")

    if st.session_state.flashcards:
        st.markdown(f"##### 📇 {len(st.session_state.flashcards)} Flashcards Generated")
        for i, card in enumerate(st.session_state.flashcards):
            with st.expander(f"🃏 Card {i+1}: {card.get('front', 'Question')[:80]}"):
                st.markdown(f"**❓ Question:**\n{card.get('front', 'N/A')}")
                st.divider()
                st.markdown(f"**✅ Answer:**\n{card.get('back', 'N/A')}")


# =====================================================================
# TAB 5 – SUMMARY
# =====================================================================
with tab_summary:
    st.markdown("""
    <div class="section-card">
        <h3>📋 PDF Summary Generator</h3>
        <p>Get a comprehensive, structured summary of your uploaded study material in one click.</p>
    </div>
    """, unsafe_allow_html=True)

    resolved_key = get_gemini_api_key(st.session_state.api_key)

    if not resolved_key:
        st.warning("🔑 Please set your API key in the Setup tab first.")
    elif not st.session_state.last_chunks:
        st.info("💬 Ask at least one question in Chat Tutor first to build context for the summary.")
    else:
        sum_col1, sum_col2 = st.columns([2, 1])
        with sum_col2:
            if st.button("📋 Generate Summary", use_container_width=True):
                with st.spinner("Generating summary... this may take a moment."):
                    try:
                        summary = generate_summary(
                            chunks=st.session_state.last_chunks,
                            api_key=resolved_key,
                            language=st.session_state.language,
                        )
                        st.session_state.summary = summary
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        st.divider()
        st.download_button(
            "📥 Download Summary",
            data=st.session_state.summary,
            file_name="edumentor_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =====================================================================
# =====================================================================
# TAB 6 – COMPETITIVE PROGRAMMING HUB (Timed)
# =====================================================================
with tab_code:
    st.markdown("""
    <div class="section-card">
        <h3>🏆 Competitive Programming Hub</h3>
        <p>Master coding with curated standard problems or AI-generated challenges from your material!</p>
    </div>
    """, unsafe_allow_html=True)

    resolved_key = get_gemini_api_key(st.session_state.api_key)

    if not resolved_key:
        st.warning("🔑 Please set your API key in the Setup tab first.")
    else:
        # --- Problem Source Selector ---
        source_opt = st.radio("Choose Program Source:", ["AI Generated (From My Docs)", "Curated Problem Bank"], horizontal=True)

        if source_opt == "AI Generated (From My Docs)":
            if not st.session_state.last_chunks:
                st.info("💬 Ask a question in Chat Tutor first to generate challenges from your material.")
            else:
                cc1, cc2 = st.columns(2)
                with cc1:
                    ch_difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="ai_ch_diff")
                with cc2:
                    ch_lang = st.selectbox("Language", ["Python", "JavaScript", "Java", "C++", "C"], key="ai_ch_lang")
                
                if st.button("🚀 Generate AI Challenge", use_container_width=True):
                    with st.spinner("Generating coding challenge..."):
                        challenge = generate_coding_challenge(st.session_state.last_chunks, resolved_key, ch_difficulty, ch_lang)
                        if "error" in challenge: st.error(challenge["error"])
                        else:
                            st.session_state.current_challenge = challenge
                            st.session_state.challenge_start = time.time()
                            st.session_state.mentor_history = []
                            st.rerun()

        else:
            # Curated Bank selection
            cb1, cb2 = st.columns(2)
            with cb1:
                cur_cat = st.selectbox("Category", ["Python", "Java", "DSA"], key="cur_cat")
            
            problems = get_curated_problems(cur_cat)
            with cb2:
                problem_titles = [p["title"] for p in problems]
                selected_title = st.selectbox("Select Problem", problem_titles)
            
            if st.button("🎯 Start Curated Challenge", use_container_width=True):
                selected_p = next((p for p in problems if p["title"] == selected_title), None)
                if selected_p:
                    st.session_state.current_challenge = selected_p
                    st.session_state.challenge_start = time.time()
                    st.session_state.mentor_history = []
                    st.rerun()

        # --- Challenge Rendering ---
        if st.session_state.current_challenge:
            ch = st.session_state.current_challenge
            st.divider()
            
            # Header info
            h1, h2, h3 = st.columns(3)
            h1.metric("Difficulty", ch.get('difficulty', 'Medium'))
            h2.metric("Level", ch.get('level', 'N/A'))
            h3.metric("Goal", f"{ch.get('time_limit', 30)} min")

            # Timer
            if st.session_state.challenge_start:
                elapsed = int(time.time() - st.session_state.challenge_start)
                limit_sec = ch.get("time_limit", 30) * 60
                rem = max(0, limit_sec - elapsed)
                m, s = divmod(rem, 60)
                t_color = "#ef4444" if rem < 120 else "#22c55e"
                st.markdown(f'<div style="text-align:center; font-size:2em; color:{t_color}; padding:10px; border:2px solid {t_color}; border-radius:10px; margin-bottom:20px;">⏱️ {m:02d}:{s:02d}</div>', unsafe_allow_html=True)
                if rem == 0: st.error("⏰ Time is up! Submit now for partial credit.")

            st.markdown(f"### 📌 {ch.get('title')}")
            st.markdown(ch.get("description", ""))

            if ch.get("examples"):
                st.markdown("**Examples:**")
                for ex in ch["examples"]:
                    st.code(f"Input: {ex.get('input')}\nOutput: {ex.get('output')}")

            st.divider()
            
            # --- Coding Area ---
            st.markdown("### 💻 Your Solution")
            user_code = st.text_area("Write your code here:", value=ch.get("solution_template", ""), height=300, key="user_code")

            # --- Expert Mentor ---
            with st.expander("👨‍🏫 Need Help? Ask Your Expert Mentor", expanded=False):
                # Mentor logic (reuse existing session_state check)
                mentor_container = st.container(height=200)
                with mentor_container:
                    if not st.session_state.mentor_history: st.write("_No conversation yet._")
                    for msg in st.session_state.mentor_history:
                        r = "🧑‍🎓 Student" if msg["role"] == "user" else "👨‍🏫 Mentor"
                        c = "#e0e0e0" if msg["role"] == "user" else "#6366f1"
                        st.markdown(f"**<span style='color:{c};'>{r}:</span>** {msg['content']}", unsafe_allow_html=True)
                
                mentor_q = st.text_input("Ask Mentor...", key="mentor_query")
                if st.button("💬 Send", key="ask_button"):
                    if mentor_q:
                        st.session_state.mentor_history.append({"role": "user", "content": mentor_q})
                        with st.spinner("Mentor thinking..."):
                            guidance = get_mentor_guidance(mentor_q, ch, user_code, st.session_state.mentor_history, resolved_key)
                            st.session_state.mentor_history.append({"role": "assistant", "content": guidance})
                            st.rerun()

            if st.button("✅ Submit Code", use_container_width=True):
                with st.spinner("Evaluating..."):
                    result = evaluate_solution(ch, user_code, resolved_key)
                    if "error" in result: st.error(result["error"])
                    else:
                        score = result.get("score", 0)
                        if score >= 70: st.balloons(); st.success(f"🎉 Perfect! Score: {score}/100")
                        else: st.warning(f"⚡ Good effort! Score: {score}/100")
                        st.markdown(f"**Feedback:** {result.get('feedback')}")
                        earned = int(ch.get("max_score", 20) * score / 100)
                        update_score(st.session_state.username, earned, "challenge")
                        st.info(f"🏅 +{earned} points added!")


# =====================================================================
# TAB 7 – VIDEO SUMMARIZER
# =====================================================================
with tab_video:
    st.markdown("""
    <div class="section-card">
        <h3>🎥 YouTube Video Summarizer</h3>
        <p>Paste a YouTube link to get an AI-powered summary with key topics and practice questions.</p>
    </div>
    """, unsafe_allow_html=True)

    resolved_key = get_gemini_api_key(st.session_state.api_key)

    if not resolved_key:
        st.warning("🔑 Please set your API key in the Setup tab first.")
    else:
        video_url = st.text_input("🔗 YouTube URL:", placeholder="https://www.youtube.com/watch?v=...",
                                   key="video_url_input")
        if st.button("📺 Summarize Video", use_container_width=True):
            if not video_url:
                st.warning("Please enter a YouTube URL.")
            else:
                with st.spinner("Fetching transcript and summarizing..."):
                    result = summarize_video(video_url, resolved_key)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state["video_summary"] = result
                        st.success("✅ Video summarized!")

        if st.session_state.get("video_summary"):
            vs = st.session_state["video_summary"]
            if vs.get("video_id"):
                st.markdown(f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{vs["video_id"]}" '
                           f'frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
            st.markdown(vs.get("summary", ""))
            st.download_button("📥 Download Summary", data=vs.get("summary", ""),
                              file_name="video_summary.md", mime="text/markdown",
                              use_container_width=True)


# =====================================================================
# TAB 8 – BLOG SUMMARIZER
# =====================================================================
with tab_blog:
    st.markdown("""
    <div class="section-card">
        <h3>📰 Blog / Article Summarizer</h3>
        <p>Paste any blog or article URL to extract and summarize its content instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    resolved_key = get_gemini_api_key(st.session_state.api_key)

    if not resolved_key:
        st.warning("🔑 Please set your API key in the Setup tab first.")
    else:
        blog_url = st.text_input("🔗 Blog/Article URL:", placeholder="https://example.com/article",
                                  key="blog_url_input")
        if st.button("📰 Summarize Article", use_container_width=True):
            if not blog_url:
                st.warning("Please enter a blog URL.")
            else:
                with st.spinner("Extracting and summarizing..."):
                    result = summarize_blog(blog_url, resolved_key)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state["blog_summary"] = result
                        st.success("✅ Article summarized!")

        if st.session_state.get("blog_summary"):
            bs = st.session_state["blog_summary"]
            st.markdown(bs.get("summary", ""))
            st.download_button("📥 Download Summary", data=bs.get("summary", ""),
                              file_name="blog_summary.md", mime="text/markdown",
                              use_container_width=True)


# =====================================================================
# TAB 9 – MIND MAP
# =====================================================================
with tab_mindmap:
    st.markdown("""
    <div class="section-card">
        <h3>🧠 Mind Map Generator</h3>
        <p>Visualize any topic as an interactive concept hierarchy to boost understanding and retention.</p>
    </div>
    """, unsafe_allow_html=True)

    resolved_key = get_gemini_api_key(st.session_state.api_key)

    if not resolved_key:
        st.warning("🔑 Please set your API key in the Setup tab first.")
    else:
        mm_topic = st.text_input("📌 Topic:", placeholder="e.g., Machine Learning Algorithms",
                                  key="mindmap_topic")
        mm_context = ""
        if st.session_state.last_chunks:
            mm_context = " ".join([c.get("content", "") for c in st.session_state.last_chunks[:3]])

        if st.button("🧠 Generate Mind Map", use_container_width=True):
            if not mm_topic:
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating mind map..."):
                    # Generate concept tree (text-based, always works)
                    tree_result = generate_concept_tree(mm_topic, resolved_key)
                    if "error" not in tree_result:
                        st.session_state["mindmap_tree"] = tree_result

                    # Also try Mermaid diagram
                    mm_result = generate_mindmap(mm_topic, mm_context, resolved_key)
                    if "error" not in mm_result:
                        st.session_state["mindmap_mermaid"] = mm_result

        if st.session_state.get("mindmap_tree"):
            st.markdown("### 🌳 Concept Hierarchy")
            st.code(st.session_state["mindmap_tree"]["tree"], language=None)

        if st.session_state.get("mindmap_mermaid"):
            st.markdown("### 🗺️ Visual Mind Map (Mermaid)")
            mermaid_code = st.session_state["mindmap_mermaid"]["mermaid"]
            # Render Mermaid via HTML
            mermaid_html = f"""
            <div class="mermaid" style="background: rgba(255,255,255,0.02); padding: 25px; border-radius: 20px; border: 1px solid rgba(212,175,55,0.15);">
            {mermaid_code}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true, theme:'dark', themeVariables: {{ 'primaryColor': '#d4af37', 'edgeLabelBackground':'#121214', 'tertiaryColor': '#1a1a1c' }}}});</script>
            """
            st.components.v1.html(mermaid_html, height=500, scrolling=True)

            with st.expander("📝 Raw Mermaid Code"):
                st.code(mermaid_code, language="text")


# =====================================================================
# TAB 10 – AUDIO SUMMARIZER
# =====================================================================
with tab_audio:
    st.markdown("""
    <div class="section-card">
        <h3>🎧 Audio Summarizer</h3>
        <p>Upload a lecture recording or audio file to get a structured AI summary.</p>
    </div>
    """, unsafe_allow_html=True)

    resolved_key = get_gemini_api_key(st.session_state.api_key)

    if not resolved_key:
        st.warning("🔑 Please set your API key in the Setup tab first.")
    else:
        audio_file = st.file_uploader("🎙️ Upload Audio File",
                                       type=["mp3", "wav", "ogg", "m4a", "flac", "aac"],
                                       key="audio_upload")
        if audio_file:
            st.audio(audio_file)
            if st.button("🎧 Summarize Audio", use_container_width=True):
                with st.spinner("Transcribing and summarizing audio... this may take a minute."):
                    result = summarize_audio(audio_file.getvalue(), audio_file.name, resolved_key)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state["audio_summary"] = result
                        st.success("✅ Audio summarized!")

        if st.session_state.get("audio_summary"):
            st.markdown(st.session_state["audio_summary"].get("summary", ""))
            st.download_button("📥 Download Summary",
                              data=st.session_state["audio_summary"].get("summary", ""),
                              file_name="audio_summary.md", mime="text/markdown",
                              use_container_width=True)


# =====================================================================
# TAB 11 – UNIFIED VISUAL EXPLAINER (Pictorial & Video)
# =====================================================================
with tab_visual:
    st.markdown("""
    <div class="section-card">
        <h3>🎨 Unified Pictorial & Video Explainer</h3>
        <p>A powerhouse learning tool: AI images, animated video breakdowns with voice, and smart YouTube links.</p>
    </div>
    """, unsafe_allow_html=True)

    resolved_key = get_gemini_api_key(st.session_state.api_key)

    if not resolved_key:
        st.warning("🔑 Please set your API key in the Setup tab first.")
    else:
        vis_topic = st.text_input("🎨 Concept to Explain:", placeholder="e.g., Backpropagation in Neural Networks",
                                   key="visual_topic")
        vis_context = ""
        if st.session_state.last_chunks:
            vis_context = " ".join([c.get("content", "") for c in st.session_state.last_chunks[:3]])

        if st.button("🚀 Generate Visual Explanation", use_container_width=True):
            if not vis_topic:
                st.warning("Please enter a concept.")
            else:
                with st.spinner("Creating visual explanation..."):
                    result = generate_visual_explanation(vis_topic, vis_context, resolved_key)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state["visual_explain"] = result
                        st.success("✅ Visual explanation ready!")

        if st.session_state.get("visual_explain"):
            ve = st.session_state["visual_explain"]
            st.markdown(f"### 🚀 {ve.get('title', 'Unified Visual Explanation')}")
            
            # --- Pictorial Analogy (AI Image) ---
            if ve.get("image_prompt"):
                st.markdown("#### 🖼️ AI Pictorial Analogy")
                img_prompt = ve["image_prompt"].replace(" ", "%20")
                img_url = f"https://image.pollinations.ai/prompt/{img_prompt}?width=800&height=400&nologo=true"
                st.image(img_url, caption=f"AI Representation: {ve.get('title')}", use_container_width=True)

            # --- Presentation & Voice ---
            st.divider()
            col_v1, col_v2 = st.columns([2, 1])
            with col_v2:
                # Related Video Search
                if ve.get("related_video_queries"):
                    query = ve["related_video_queries"][0].replace(" ", "+")
                    st.link_button("📺 Watch Related Videos on YouTube", 
                                  f"https://www.youtube.com/results?search_query={query}",
                                  use_container_width=True)

                # Voice Presentation
                full_text = f"Concept: {ve.get('title')}. "
                for s in ve.get("steps", []): full_text += f"{s.get('text')}. "
                clean_text = full_text.replace('"', '\\"').replace("'", "\\'")
                if st.button("🔊 Play Voice Presentation", use_container_width=True):
                    st.components.v1.html(f"<script>const m=new SpeechSynthesisUtterance('{clean_text}');m.rate=0.9;window.speechSynthesis.speak(m);</script>", height=0)

            with col_v1:
                # Mermaid Diagram
                if ve.get("mermaid_code"):
                    mermaid_html = f"""
                    <div class="mermaid" style="background: #1a1a2e; padding: 20px; border-radius: 12px;">
                    {ve['mermaid_code']}
                    </div>
                    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                    <script>mermaid.initialize({{startOnLoad:true, theme:'dark'}});</script>
                    """
                    st.components.v1.html(mermaid_html, height=400, scrolling=True)

            # Animated Steps
            if ve.get("steps"):
                st.markdown("#### 🔄 Step-by-Step Breakdown")
                cols = st.columns(min(len(ve["steps"]), 3))
                for i, step in enumerate(ve["steps"]):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="step-card fade-in">
                            <span style="font-size:1.5em; color:#6366f1;">{i+1}</span><br>
                            {step.get('text', '')}
                        </div>
                        """, unsafe_allow_html=True)
            
            if ve.get("summary"):
                st.info(f"💡 **Key Insight:** {ve['summary']}")


# =====================================================================
# TAB 12 – LEADERBOARD
# =====================================================================
with tab_leader:
    st.markdown("""
    <div class="section-card">
        <h3>🏅 Leaderboard</h3>
        <p>See how you rank against other learners! Earn points from quizzes and coding challenges.</p>
    </div>
    """, unsafe_allow_html=True)

    board = get_leaderboard(top_n=20)

    if not board:
        st.info("No scores yet. Take quizzes and coding challenges to appear on the leaderboard!")
    else:
        # Highlight current user
        st.markdown(f"**Your Username:** `{st.session_state.username}`")

        # Display leaderboard as table
        for rank, entry in enumerate(board, 1):
            is_me = entry["username"].lower() == st.session_state.username.lower()
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            bg = "background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white;" if is_me else ""
            st.markdown(f"""
            <div style="display:flex; align-items:center; padding:10px 15px;
                        margin:4px 0; border-radius:10px; {bg}
                        border: 1px solid {'#6366f1' if is_me else '#333'};">
                <span style="font-size:1.3em; width:50px;">{medal}</span>
                <span style="flex:1; font-weight:{'bold' if is_me else 'normal'};">
                    {entry['username']} {'(You)' if is_me else ''}</span>
                <span style="margin-right:20px;">🏆 {entry['score']} pts</span>
                <span style="margin-right:20px;">📝 {entry['quizzes']} quizzes</span>
                <span style="margin-right:20px;">💻 {entry['challenges']} challenges</span>
                <span>🔥 {entry['streak']} best streak</span>
            </div>
            """, unsafe_allow_html=True)

        # Score breakdown chart using plotly
        if board:
            st.markdown("### 📊 Score Distribution")
            df = pd.DataFrame(board[:10])
            fig = px.bar(df, x="username", y="score",
                        color="score", title="Top 10 Learners",
                        color_continuous_scale="Viridis")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis_title="Learner",
                yaxis_title="Score",
            )
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 12 – INSIGHTS
# =====================================================================
with tab_insights:
    st.markdown("""
    <div class="section-card">
        <h3>📊 Learning & RAG Insights Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)

    lp = st.session_state.learner_profile
    logs = st.session_state.logs

    if not logs and not lp.quiz_scores:
        st.info("No data yet. Chat with the tutor and take quizzes to see insights here.")
    else:
        # --- Row 1: Summary metrics ---
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-card"><h3>{}</h3><p>Questions Asked</p></div>'.format(
                len(logs)), unsafe_allow_html=True)
        with m2:
            avg_sim = sum(l.get("avg_similarity", 0) for l in logs) / len(logs) if logs else 0
            st.markdown('<div class="metric-card"><h3>{:.2f}</h3><p>Avg Similarity</p></div>'.format(
                avg_sim), unsafe_allow_html=True)
        with m3:
            avg_rouge = sum(l.get("rouge_l_f1", 0) for l in logs) / len(logs) if logs else 0
            st.markdown('<div class="metric-card"><h3>{:.3f}</h3><p>Avg ROUGE-L F1</p></div>'.format(
                avg_rouge), unsafe_allow_html=True)
        with m4:
            hallucination_count = sum(1 for l in logs if l.get("hallucination_risk", False))
            st.markdown('<div class="metric-card"><h3>{}</h3><p>⚠️ Low Confidence</p></div>'.format(
                hallucination_count), unsafe_allow_html=True)

        st.divider()

        # --- Quiz Score Trend (Plotly) ---
        if lp.quiz_scores:
            st.markdown("##### 📈 Quiz Score Trend")
            trend = lp.get_quiz_trend()
            df_trend = pd.DataFrame({
                "Attempt": list(range(1, len(trend) + 1)),
                "Score (%)": [s["percentage"] for s in trend],
                "Concept": [s["concept"] for s in trend]
            })
            fig_trend = px.line(df_trend, x="Attempt", y="Score (%)", hover_name="Concept", markers=True)
            fig_trend.update_traces(line_color="#d4af37", marker=dict(size=10, color="#d4af37", line=dict(width=2, color="white")))
            fig_trend.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", margin=dict(l=0, r=0, t=20, b=0), height=350,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # Quiz score gain: first vs last attempt per concept
            st.markdown("##### ⬆️ Quiz Score Gain (First ➜ Latest per Concept)")
            concepts_seen = {}
            for s in trend:
                c = s["concept"]
                if c not in concepts_seen:
                    concepts_seen[c] = {"first": s["percentage"], "latest": s["percentage"]}
                else:
                    concepts_seen[c]["latest"] = s["percentage"]

            for concept, scores in concepts_seen.items():
                gain = scores["latest"] - scores["first"]
                arrow = "🟢 +" if gain > 0 else "🔴 " if gain < 0 else "⚪ "
                st.markdown(f"- **{concept}**: {scores['first']}% → {scores['latest']}% ({arrow}{gain:.1f}%)")

        # --- Most Asked Concepts (Plotly) ---
        if lp.concepts_asked:
            st.divider()
            st.markdown("##### 🔤 Most Asked Concepts")
            freq = lp.get_concept_frequency()
            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            df_freq = pd.DataFrame({
                "Concept": [c for c, _ in sorted_freq],
                "Times Asked": [n for _, n in sorted_freq]
            })
            fig_bar = px.bar(df_freq, x="Concept", y="Times Asked", text_auto=True)
            fig_bar.update_traces(marker_color="#d4af37", marker_line_color="#ffffff", marker_line_width=1, opacity=0.8)
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", margin=dict(l=0, r=0, t=20, b=0), height=350,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Weak Areas ---
        if lp.weak_concepts:
            st.markdown("##### ⚠️ Weak Areas (Need Reinforcement)")
            for wc in lp.weak_concepts:
                st.markdown(f"- 🔴 **{wc}**")

        # --- RAG Metrics Log ---
        if logs:
            st.divider()
            st.markdown("##### 📋 RAG Interaction Log")
            import pandas as pd
            df = pd.DataFrame(logs)
            display_cols = [
                "timestamp", "query", "avg_similarity", "hit_rate",
                "rouge_l_f1", "bleu", "is_confident", "hallucination_risk",
                "quiz_score", "quiz_max",
            ]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols], use_container_width=True)

        # --- Export ---
        st.divider()
        st.markdown("##### 💾 Export Reports")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            if logs:
                csv_str = export_logs_csv(logs)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_str,
                    file_name="edumentor_logs.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with ecol2:
            if logs:
                json_str = export_logs_json(logs)
                st.download_button(
                    "📥 Download JSON",
                    data=json_str,
                    file_name="edumentor_logs.json",
                    mime="application/json",
                    use_container_width=True,
                )
