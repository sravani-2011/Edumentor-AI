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
import os, sys

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
import json as json_lib

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
# ✨ PREMIUM DARK-THEME CSS — Minor Project Showcase Edition
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* ── Dark theme base ──────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
        color: #e0e0e0;
    }
    .stApp > header { background: transparent !important; }

    /* ── Animated hero header ─────────────────────────── */
    @keyframes gradientFlow {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-6px); }
    }
    @keyframes shimmer {
        0%   { left: -100%; }
        100% { left: 200%; }
    }
    .hero {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe, #43e97b);
        background-size: 400% 400%;
        animation: gradientFlow 10s ease infinite;
        padding: 2.5rem 2.8rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4), 0 0 80px rgba(118, 75, 162, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 60%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        animation: shimmer 4s ease-in-out infinite;
    }
    .hero-icon {
        font-size: 2.8rem;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    .hero h1 {
        margin: 0.3rem 0 0 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        text-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }
    .hero .tagline {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    .hero .badges {
        display: flex;
        gap: 0.6rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .hero .badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        padding: 0.3rem 0.9rem;
        border-radius: 50px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* ── Section cards ────────────────────────────────── */
    .section-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .section-card:hover {
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.08);
    }
    .section-card h3 {
        margin: 0 0 0.8rem 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: #b8c5ff;
    }

    /* ── Tab bar ──────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.03);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #a0a0c0 !important;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: #c5cafe !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.35);
    }
    /* tab panel text color */
    .stTabs [data-baseweb="tab-panel"] {
        color: #e0e0e0;
    }

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(102, 126, 234, 0.45);
        background: linear-gradient(135deg, #7b93f5, #8b5dc5) !important;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Inputs ───────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px;
        color: #e0e0e0 !important;
        transition: all 0.2s ease;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #666688 !important;
    }
    /* Labels */
    .stTextInput > label, .stTextArea > label, .stSelectbox > label,
    .stFileUploader > label, .stSlider > label, .stRadio > label {
        color: #b0b0d0 !important;
        font-weight: 600;
    }

    /* ── Selectbox ────────────────────────────────────── */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px;
        color: #e0e0e0 !important;
    }

    /* ── Chat messages ────────────────────────────────── */
    .stChatMessage {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* ── File uploader ────────────────────────────────── */
    .stFileUploader > div > div {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 14px;
    }
    .stFileUploader > div > div:hover {
        border-color: rgba(102, 126, 234, 0.5) !important;
    }

    /* ── Metric cards (Insights) ──────────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.5rem 1rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.12);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
        color: #b8c5ff !important;
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
        color: #8888aa;
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
        background: linear-gradient(180deg, #13112b 0%, #1a1840 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #c5cafe !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] span {
        color: #c5cafe !important;
    }

    /* ── Streamlit metrics override ───────────────────── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: #8888aa !important; }
    [data-testid="stMetricValue"] { color: #b8c5ff !important; }

    /* ── Dividers ──────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
    }

    /* ── Alerts / info boxes ──────────────────────────── */
    .stAlert {
        border-radius: 14px;
        border: none;
    }

    /* ── Subheaders ───────────────────────────────────── */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: #d0d0f0 !important;
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
        background: rgba(102, 126, 234, 0.15) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: #b8c5ff !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(102, 126, 234, 0.25) !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
    }

    /* ── Radio buttons ────────────────────────────────── */
    .stRadio > div { color: #c5cafe !important; }
    .stRadio label span { color: #c5cafe !important; }

    /* ── Caption text ─────────────────────────────────── */
    .stCaption, small { color: #7777a0 !important; }

    /* ── Chat input ───────────────────────────────────── */
    .stChatInput > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px;
    }
    .stChatInput textarea {
        color: #e0e0e0 !important;
    }
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
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# ✨ Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <span class="hero-icon">🎓</span>
    <h1>EduMentor AI</h1>
    <p class="tagline">Your Personal AI Tutor — Adaptive Learning Powered by RAG Pipeline</p>
    <div class="badges">
        <span class="badge">🤖 Gemini 2.5 Flash Lite</span>
        <span class="badge">🔍 ChromaDB RAG</span>
        <span class="badge">📊 Auto-Evaluation</span>
        <span class="badge">🧠 Adaptive Quizzes</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_setup, tab_chat, tab_quiz, tab_flash, tab_summary, tab_insights = st.tabs([
    "⚙️ Setup", "💬 Chat Tutor", "📝 Practice Quiz", "📇 Flashcards", "📋 Summary", "📊 Insights"
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

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                # Save follow-ups for persistent rendering
                st.session_state.last_follow_ups = [
                    f"Can you explain more about {prompt[:40]}?",
                    f"What are common mistakes related to this topic?",
                    f"Give me a real-world example of this concept.",
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

    # Voice input via Web Speech API (sidebar)
    with st.sidebar:
        st.divider()
        st.markdown("##### 🎤 Voice Input")
        st.markdown(
            """<p style="font-size:0.85em; color:#aaa;">
            Click the button below, speak your question, then paste it into the chat box.</p>""",
            unsafe_allow_html=True,
        )
        voice_html = """
        <div id="voice-container" style="text-align:center;">
            <button id="voice-btn" onclick="startVoice()" style="
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                color: white; border: none; padding: 12px 24px;
                border-radius: 12px; cursor: pointer; font-size: 16px;
                width: 100%; transition: all 0.3s;">
                🎤 Tap to Speak
            </button>
            <p id="voice-status" style="margin-top:8px; font-size:0.85em; color:#aaa;"></p>
            <textarea id="voice-result" readonly style="
                width:100%; min-height:60px; margin-top:8px;
                background: #1a1a2e; color: #e0e0e0; border: 1px solid #333;
                border-radius: 8px; padding: 8px; font-size: 14px;
                display:none; resize:vertical;
            "></textarea>
        </div>
        <script>
        function startVoice() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                document.getElementById('voice-status').textContent = '❌ Speech not supported in this browser';
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
            btn.textContent = '🔴 Listening...';
            btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
            status.textContent = 'Speak now...';
            recognition.start();
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                result.style.display = 'block';
                result.value = transcript;
                status.textContent = '✅ Copy the text above and paste it into the chat!';
                btn.textContent = '🎤 Tap to Speak';
                btn.style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
            };
            recognition.onerror = function(event) {
                status.textContent = '❌ Error: ' + event.error;
                btn.textContent = '🎤 Tap to Speak';
                btn.style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
            };
            recognition.onend = function() {
                btn.textContent = '🎤 Tap to Speak';
                btn.style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
            };
        }
        </script>
        """
        st.components.v1.html(voice_html, height=200)


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
# TAB 6 – INSIGHTS
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

        # --- Quiz Score Trend ---
        if lp.quiz_scores:
            st.markdown("##### 📈 Quiz Score Trend")
            trend = lp.get_quiz_trend()
            chart_data = {
                "Attempt": list(range(1, len(trend) + 1)),
                "Score (%)": [s["percentage"] for s in trend],
            }
            st.line_chart(chart_data, x="Attempt", y="Score (%)")

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

        # --- Most Asked Concepts ---
        if lp.concepts_asked:
            st.markdown("##### 🔤 Most Asked Concepts")
            freq = lp.get_concept_frequency()
            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            concept_data = {
                "Concept": [c for c, _ in sorted_freq],
                "Times Asked": [n for _, n in sorted_freq],
            }
            st.bar_chart(concept_data, x="Concept", y="Times Asked")

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
