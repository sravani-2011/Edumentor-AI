"""
tutor/summarizer.py – Generate a structured summary of uploaded PDF content using Gemini.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from utils.config import CHAT_MODEL, DEFAULT_TEMPERATURE

SUMMARY_PROMPT = """You are an expert academic summarizer. Given the following study material,
create a comprehensive, well-structured summary.

Use this format:
## 📋 Document Summary

### 🎯 Main Topics
- List the main topics covered

### 📝 Key Concepts
- Explain each key concept briefly (1-2 sentences each)

### 🔑 Important Definitions
- List important terms and their definitions

### 💡 Key Takeaways
- 5-7 bullet points of the most important things to remember

### 📊 Quick Facts
- Any numbers, dates, or factual data worth remembering

Study Material:
{content}"""


def generate_summary(
    chunks: list[dict],
    api_key: str,
    language: str = "English",
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Generate a structured summary from retrieved chunks using Gemini."""
    if not chunks:
        return "No content available. Please upload and ingest PDFs first."

    # Combine chunk content
    combined = "\n\n".join(c.get("content", "") for c in chunks)
    combined = combined[:10000]  # Truncate to avoid token limits

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )

    prompt = SUMMARY_PROMPT.format(content=combined)
    if language != "English":
        prompt += f"\n\nIMPORTANT: Write the entire summary in {language}."

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
