"""
tools/expert_mentor.py – Provides real-time guidance (Expert Mentor) for coding challenges.
"""
import google.generativeai as genai

def get_mentor_guidance(question: str, challenge: dict, user_code: str, history: list[dict], api_key: str) -> str:
    \"\"\"
    Act as an expert mentor guiding a student through a specific coding challenge.
    Provide hints and guiding questions instead of direct solutions.
    \"\"\"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Format history for prompt
    hist_str = ""
    for msg in history[-5:]: # Last 5 messages
        role = "Student" if msg["role"] == "user" else "Mentor"
        hist_str += f"{role}: {msg['content']}\n"

    prompt = f\"\"\"You are an **Expert Coding Mentor**. Your goal is to guide the student through a programming challenge.
    
    CHALLENGE: {challenge.get('title')}
    DESCRIPTION: {challenge.get('description')}
    DIFFICULTY: {challenge.get('difficulty')}
    LANGUAGE: {challenge.get('language')}
    
    STUDENT'S CURRENT CODE:
    ```{challenge.get('language', 'python')}
    {user_code}
    ```
    
    CONVERSATION HISTORY:
    {hist_str}
    
    STUDENT'S QUESTION: {question}
    
    GUIDELINES:
    1. Do NOT give the full solution.
    2. Provide a small hint or ask a guiding question to lead them to the answer.
    3. Use an encouraging, professional expert tone.
    4. You can explain concepts (e.g., 'What is a dictionary in Python?') if they ask.
    
    RESPONSE:
    \"\"\"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Mentor is currently busy: {str(e)}"
