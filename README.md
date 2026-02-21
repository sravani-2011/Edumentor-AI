# 🎓 EduMentor AI: Adaptive Learning Chatbot

A production-ready e-learning chatbot that provides personalized tutoring using a RAG (Retrieval-Augmented Generation) pipeline. Built with Streamlit, LangChain, ChromaDB, and OpenAI.

## Features

- **📂 PDF Knowledge Ingestion** – Upload course PDFs, auto-chunk with metadata, persist in ChromaDB
- **💬 RAG-Powered Tutoring** – Ask questions, get cited answers with follow-up suggestions
- **🧒↔️🎓 Adaptive Personalization** – Explanations adapt to Beginner / Intermediate / Advanced
- **📝 Micro-Quiz Generator** – Auto-generated MCQ + short answer with LLM grading
- **📊 Insights Dashboard** – ROUGE-L, BLEU metrics, quiz score trends, CSV/JSON export

## Quick Start

### 1. Prerequisites
- Python 3.10+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 2. Install Dependencies

```bash
# Windows
cd C:\Users\devag\.gemini\antigravity\scratch\edu_mentor_ai
pip install -r requirements.txt

# Mac / Linux
cd /path/to/edu_mentor_ai
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. First-Time Setup

1. Go to the **⚙️ Setup** tab
2. Enter your **OpenAI API key** (stored in session only)
3. Upload **1-2 PDF files** (course notes, slides, handouts)
4. Click **Build / Update Knowledge Base**
5. Set your **Learner Profile** (name, skill level)
6. Switch to **💬 Chat Tutor** and ask a question!

## Sample Dataset Flow

1. Find any educational PDF (e.g., a chapter from a textbook, lecture slides)
2. Upload it in the Setup tab
3. Ask: *"What are the main concepts covered in this material?"*
4. Try the quiz: Go to **📝 Practice Quiz** → Generate Quiz → Answer → Submit

## Project Structure

```
edu_mentor_ai/
├── app.py                  # Streamlit entry point (4-tab UI)
├── requirements.txt        # Python dependencies
├── rag/
│   ├── ingest.py           # PDF loading, chunking, ChromaDB storage
│   ├── retriever.py        # Similarity search with confidence scoring
│   └── chain.py            # RAG chain with tutor persona prompts
├── tutor/
│   ├── personalize.py      # Learner profile + adaptation rules
│   ├── quiz.py             # MCQ/short-answer generation
│   └── grader.py           # LLM-based rubric grading
├── eval/
│   ├── metrics.py          # ROUGE-L, BLEU computation (pure Python)
│   └── logger.py           # Structured logging + CSV/JSON export
└── utils/
    └── config.py           # Configuration + defaults
```

## Customization Guide (For Educators)

| What to customize          | Where to change it             |
|---------------------------|--------------------------------|
| Chunk size & overlap       | `utils/config.py` – `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Number of retrieved chunks | `utils/config.py` – `TOP_K` |
| Tutor personality          | `rag/chain.py` – `SYSTEM_PROMPT` |
| Quiz style & count         | `tutor/quiz.py` – `QUIZ_SYSTEM_PROMPT` |
| Grading rubric             | `tutor/grader.py` – `GRADING_SYSTEM_PROMPT` |
| Skill level rules          | `rag/chain.py` – personalization block in `get_rag_answer()` |

## Optional: .env File

Create a `.env` file (copy from `.env.example`) to store your API key as a fallback:

```
OPENAI_API_KEY=sk-your-key-here
```

> ⚠️ The UI key entry is the **primary** method. The `.env` file is a fallback only.

## License

MIT
