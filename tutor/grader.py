"""
tutor/grader.py – LLM-based rubric grading with partial credit and feedback.

Compares learner answers to reference answers using the LLM as a grader.
Provides score, feedback, hints, and the correct answer.
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.config import CHAT_MODEL


# ---------------------------------------------------------------------------
# Grading Prompt
# ---------------------------------------------------------------------------
GRADING_SYSTEM_PROMPT = """You are a fair and encouraging educational grader.

### Grading Rules:
1. Compare the learner's answer to the reference answer.
2. Award partial credit where appropriate (e.g., if the learner captures the main idea but misses details).
3. For MCQ: score is 0 or 1 (no partial credit).
4. For ShortAnswer: score is 0.0 to 1.0 (partial credit allowed).
5. Provide brief, constructive feedback.
6. If wrong, give a hint (not the full answer) to help them learn.
7. Always be encouraging – acknowledge what they got right.

Respond with valid JSON only. No markdown. Use this exact format:
{{
  "results": [
    {{
      "question_id": 1,
      "score": 1.0,
      "max_score": 1.0,
      "is_correct": true,
      "feedback": "Great job! You correctly identified ...",
      "hint": null,
      "correct_answer": "The correct answer."
    }},
    {{
      "question_id": 2,
      "score": 0.5,
      "max_score": 1.0,
      "is_correct": false,
      "feedback": "You're on the right track. You mentioned X but missed Y.",
      "hint": "Think about how Y relates to Z.",
      "correct_answer": "The full correct answer."
    }}
  ],
  "total_score": 1.5,
  "max_total": 2.0,
  "overall_feedback": "Good effort! Focus on ... for improvement."
}}"""

GRADING_HUMAN_TEMPLATE = """### Questions and Answers to Grade:

{qa_pairs}

Grade each answer now. Respond with JSON only."""


def grade_quiz(
    questions: list[dict],
    user_answers: list[str],
    api_key: str,
    temperature: float = 0.1,
) -> dict:
    """
    Grade learner answers against quiz reference answers.

    Parameters
    ----------
    questions : list[dict]
        Quiz questions from quiz.py (each has 'question', 'correct_answer', 'type').
    user_answers : list[str]
        Learner's answers in the same order as questions.
    api_key : str
        OpenAI API key.
    temperature : float
        Low temperature for consistent grading.

    Returns
    -------
    dict
        Grading results with scores, feedback, and hints.
    """
    # Build Q&A pairs for grading
    qa_parts = []
    for i, (q, ans) in enumerate(zip(questions, user_answers)):
        qa_parts.append(
            f"**Question {i+1}** (Type: {q.get('type', 'Unknown')}):\n"
            f"  Question: {q['question']}\n"
            f"  Reference Answer: {q['correct_answer']}\n"
            f"  Learner's Answer: {ans}\n"
        )
    qa_str = "\n".join(qa_parts)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=temperature,
        openai_api_key=api_key,
    )

    messages = [
        SystemMessage(content=GRADING_SYSTEM_PROMPT),
        HumanMessage(content=GRADING_HUMAN_TEMPLATE.format(qa_pairs=qa_str)),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Parse JSON – handle markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: simple grading
        result = _fallback_grading(questions, user_answers)

    return result


def _fallback_grading(questions: list[dict], user_answers: list[str]) -> dict:
    """Simple string-match fallback if LLM grading fails."""
    results = []
    total = 0.0
    for i, (q, ans) in enumerate(zip(questions, user_answers)):
        correct = q.get("correct_answer", "").strip().lower()
        user = ans.strip().lower()

        if q.get("type") == "MCQ":
            # Check if the user selected the correct option letter
            is_correct = (
                user == correct or
                user.startswith(correct[:2]) or
                correct.startswith(user[:2])
            )
            score = 1.0 if is_correct else 0.0
        else:
            # Simple word overlap for short answers
            correct_words = set(correct.split())
            user_words = set(user.split())
            if correct_words:
                overlap = len(correct_words & user_words) / len(correct_words)
                score = round(min(overlap, 1.0), 2)
            else:
                score = 0.0
            is_correct = score >= 0.7

        total += score
        results.append({
            "question_id": i + 1,
            "score": score,
            "max_score": 1.0,
            "is_correct": is_correct,
            "feedback": "Correct!" if is_correct else "Not quite right.",
            "hint": None if is_correct else "Review the relevant section in your notes.",
            "correct_answer": q.get("correct_answer", ""),
        })

    return {
        "results": results,
        "total_score": round(total, 2),
        "max_total": float(len(questions)),
        "overall_feedback": "Graded using basic matching (LLM grading unavailable).",
    }
