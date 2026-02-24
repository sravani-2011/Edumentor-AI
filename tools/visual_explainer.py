"""
tools/visual_explainer.py – Generates unified pictorial, video-step, and diagram explanations.
"""
import google.generativeai as genai

def generate_visual_explanation(topic: str, context: str, api_key: str) -> dict:
    """
    Generate a Mermaid diagram, animated steps, and a pictorial analogy prompt for a topic.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""Explain the concept of '{topic}' visually.
    
    Context: {context[:2000] if context else 'No specific context.'}
    
    Your goal is to provide a unified visual and instructional experience:
    1. A Mermaid flowchart (graph TD) showing the process.
    2. A structured step-by-step breakdown for an animated presentation.
    3. A 'Pictorial Analogy' prompt: A highly descriptive, artistic, or diagrammatic prompt that an AI image generator can use to illustrate this concept.
    4. Related search queries to find educational YouTube videos for this topic.
    
    Return a JSON object with:
    {{
        "title": "Unified Explanation Title",
        "image_prompt": "A detailed descriptive prompt for AI image generation (e.g., 'A futuristic 3D render of a neural network acting like a biological brain...').",
        "mermaid_code": "graph TD\\n...",
        "steps": [
            {{"step": 1, "text": "Step 1 text...", "animation": "fade-in"}},
            {{"step": 2, "text": "Step 2 text...", "animation": "slide-left"}}
        ],
        "summary": "Concise concept summary.",
        "related_video_queries": ["Search query 1", "Search query 2"]
    }}
    
    Ensure the Mermaid code is valid.
    Return ONLY JSON.
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        import json
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"Failed to generate visual explanation: {str(e)}"}
