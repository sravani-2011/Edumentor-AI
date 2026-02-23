"""
coding_challenge.py – Generate timed coding challenges from PDF context using Gemini.
"""

import json
import re
import google.generativeai as genai


def generate_coding_challenge(
    chunks: list[dict],
    api_key: str,
    difficulty: str = "Medium",
    language: str = "Python",
) -> dict:
    """
    Generate a coding challenge based on PDF context.

    Parameters
    ----------
    chunks : list[dict]
        Retrieved context chunks.
    api_key : str
        Gemini API key.
    difficulty : str
        Easy, Medium, or Hard.
    language : str
        Programming language for the challenge.

    Returns
    -------
    dict with challenge details or {"error": "..."}.
    """
    context_parts = []
    for chunk in chunks:
        context_parts.append(chunk.get("content", ""))
    context = "\n".join(context_parts)[:5000]

    if not context.strip():
        return {"error": "No context available. Ask a question in Chat Tutor first."}

    # Time limits based on difficulty
    time_limits = {"Easy": 15, "Medium": 30, "Hard": 45}
    time_limit = time_limits.get(difficulty, 30)

    # Score multiplier
    multipliers = {"Easy": 10, "Medium": 20, "Hard": 30}
    max_score = multipliers.get(difficulty, 20)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Based on this educational context, generate a coding challenge.

CONTEXT:
{context}

REQUIREMENTS:
- Difficulty: {difficulty}
- Language: {language}
- Time limit: {time_limit} minutes

Return a JSON object with EXACTLY these fields:
{{
    "title": "Challenge title",
    "description": "Detailed problem description",
    "examples": [
        {{"input": "example input", "output": "expected output"}},
        {{"input": "example input 2", "output": "expected output 2"}}
    ],
    "constraints": ["constraint 1", "constraint 2"],
    "hints": ["hint 1", "hint 2"],
    "solution_template": "def solve(input):\\n    # Your code here\\n    pass",
    "test_cases": [
        {{"input": "test input", "expected": "expected output"}},
        {{"input": "test input 2", "expected": "expected output 2"}}
    ],
    "topics": ["topic1", "topic2"]
}}

Return ONLY valid JSON, no markdown fences.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1500, "temperature": 0.5},
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        challenge = json.loads(raw)
        challenge["difficulty"] = difficulty
        challenge["time_limit"] = time_limit
        challenge["max_score"] = max_score
        challenge["language"] = language
        return challenge

    except json.JSONDecodeError:
        return {"error": "Failed to parse challenge. Try again."}
    except Exception as e:
        return {"error": f"Challenge generation failed: {str(e)}"}


def evaluate_solution(
    challenge: dict,
    user_code: str,
    api_key: str,
) -> dict:
    """
    Evaluate a user's coding solution using Gemini.

    Returns dict with score, feedback, and correctness.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Evaluate this coding solution.

PROBLEM: {challenge.get('title', 'Unknown')}
DESCRIPTION: {challenge.get('description', '')}
TEST CASES: {json.dumps(challenge.get('test_cases', []))}

USER'S CODE:
```{challenge.get('language', 'python')}
{user_code}
```

Evaluate and return a JSON object:
{{
    "score": <0-100>,
    "passed_tests": <number of tests likely passed>,
    "total_tests": <total test cases>,
    "feedback": "Detailed feedback on the solution",
    "correctness": "correct" or "partial" or "incorrect",
    "suggestions": ["improvement 1", "improvement 2"]
}}

Return ONLY valid JSON.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 800, "temperature": 0.2},
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
