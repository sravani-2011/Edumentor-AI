"""
tutor/quiz.py – Micro-quiz generation from retrieved context.

Generates 3-5 questions (MCQ + short answer) strictly from the retrieved
context chunks. Questions include correct answers and difficulty tags.
"""

import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.config import CHAT_MODEL


# ---------------------------------------------------------------------------
# Quiz Generation Prompt
# ---------------------------------------------------------------------------
QUIZ_SYSTEM_PROMPT = """You are a quiz generator for educational content. Generate questions STRICTLY from the provided context.

Rules:
1. Generate exactly {num_questions} questions.
2. Mix question types: some Multiple Choice (MCQ) and some Short Answer.
3. Each question must be answerable from the provided context alone.
4. Include the correct answer for each question.
5. Tag each question with a difficulty: Easy, Medium, or Hard.
6. For MCQ questions, provide exactly 4 options (A, B, C, D).

You MUST respond with valid JSON only. No markdown, no explanation. Use this exact format:
{{
  "quiz_topic": "Brief topic description",
  "questions": [
    {{
      "id": 1,
      "type": "MCQ",
      "question": "What is ...?",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A) ...",
      "difficulty": "Easy",
      "explanation": "Brief explanation of why this is correct."
    }},
    {{
      "id": 2,
      "type": "ShortAnswer",
      "question": "Explain ...",
      "options": null,
      "correct_answer": "The expected answer ...",
      "difficulty": "Medium",
      "explanation": "Key points that should be covered."
    }}
  ]
}}"""

QUIZ_HUMAN_TEMPLATE = """### Context (generate questions from this only):
{context}

### Learner Skill Level: {skill_level}

Generate {num_questions} quiz questions now. Respond with JSON only."""


def generate_quiz(
    chunks: list[dict],
    api_key: str,
    skill_level: str = "Intermediate",
    num_questions: int = 4,
    temperature: float = 0.4,
) -> dict:
    """
    Generate a quiz from retrieved context chunks.

    Parameters
    ----------
    chunks : list[dict]
        Retrieved context chunks with 'content' and 'metadata'.
    api_key : str
        OpenAI API key.
    skill_level : str
        Learner's skill level (adjusts difficulty distribution).
    num_questions : int
        Number of questions to generate (3-5 recommended).
    temperature : float
        LLM temperature for variety.

    Returns
    -------
    dict
        Parsed quiz JSON with 'quiz_topic' and 'questions' list, or
        an error dict if generation/parsing fails.
    """
    # Build context string from chunks
    context_parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        context_parts.append(f"[{source}, p.{page}] {chunk['content']}")
    context_str = "\n\n".join(context_parts)

    if not context_str.strip():
        return {"error": "No context available to generate quiz questions."}

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
        google_api_key=api_key,
    )

    messages = [
        SystemMessage(content=QUIZ_SYSTEM_PROMPT.format(num_questions=num_questions)),
        HumanMessage(content=QUIZ_HUMAN_TEMPLATE.format(
            context=context_str,
            skill_level=skill_level,
            num_questions=num_questions,
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Parse JSON – handle potential markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        quiz = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse quiz response.", "raw": raw}

    return quiz
