"""
tutor/flashcards.py – Generate flashcards from retrieved PDF chunks using Gemini.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils.config import CHAT_MODEL, DEFAULT_TEMPERATURE

FLASHCARD_PROMPT = """You are an educational content creator. Given the following study material,
create exactly {count} flashcards for effective studying.

Each flashcard should have:
- "front": A clear, specific question
- "back": A concise but complete answer (2-3 sentences max)

Return ONLY valid JSON — an array of objects with "front" and "back" keys.
Do not include any other text, markdown, or code fences.

Study Material:
{content}"""


def generate_flashcards(
    chunks: list[dict],
    api_key: str,
    count: int = 10,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict]:
    """Generate flashcards from retrieved chunks using Gemini."""
    if not chunks:
        return []

    # Combine chunk content
    combined = "\n\n".join(c.get("content", "") for c in chunks)
    # Truncate to avoid token limits
    combined = combined[:8000]

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )

    prompt = FLASHCARD_PROMPT.format(count=count, content=combined)
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse JSON from response
    text = response.content.strip()
    # Remove code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    try:
        cards = json.loads(text)
        if isinstance(cards, list):
            return cards[:count]
    except json.JSONDecodeError:
        pass

    return [{"front": "Could not generate flashcards", "back": "Try again or upload more content."}]
""", "Complexity": 5, "Description": "New module that generates study flashcards from PDF chunks using Gemini API", "EmptyFile": false, "IsArtifact": false, "Overwrite": false, "TargetFile": "C:\\Users\\devag\\.gemini\\antigravity\\scratch\\edu_mentor_ai\\tutor\\flashcards.py"}
