"""
mindmap.py – Generate mind maps from topics using Gemini + Mermaid syntax.
"""

import google.generativeai as genai


def generate_mindmap(topic: str, context: str, api_key: str) -> dict:
    """
    Generate a Mermaid mindmap diagram for a topic.

    Parameters
    ----------
    topic : str
        The main topic for the mind map.
    context : str
        Additional context (e.g., from RAG answer or PDF chunks).
    api_key : str
        Gemini API key.

    Returns
    -------
    dict with 'mermaid' (Mermaid code) and 'explanation', or {"error": "..."}.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Create a mind map for the following topic. Use Mermaid mindmap syntax.

Topic: {topic}
Context: {context[:2000] if context else 'No additional context.'}

Requirements:
1. Use valid Mermaid mindmap syntax (start with 'mindmap' on the first line)
2. Use the root node as the main topic
3. Include 4-6 main branches
4. Each branch should have 2-3 sub-items
5. Keep labels concise (max 5 words each)

Return ONLY the Mermaid code block, nothing else. Example format:
mindmap
  root((Main Topic))
    Branch 1
      Sub item 1
      Sub item 2
    Branch 2
      Sub item 1
      Sub item 2
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 800, "temperature": 0.4},
        )
        raw = response.text.strip()

        # Clean up markdown code fences if present
        raw = raw.replace("```mermaid", "").replace("```", "").strip()

        # Validate it starts with 'mindmap'
        if not raw.startswith("mindmap"):
            raw = "mindmap\n" + raw

        return {"mermaid": raw, "topic": topic}
    except Exception as e:
        return {"error": f"Mind map generation failed: {str(e)}"}


def generate_concept_tree(topic: str, api_key: str) -> dict:
    """Generate a text-based concept tree (fallback for non-Mermaid rendering)."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Create a detailed concept tree/hierarchy for: {topic}

Use this format with indentation:
📌 Main Topic
├── 📂 Branch 1
│   ├── 📄 Sub-concept 1
│   ├── 📄 Sub-concept 2
│   └── 📄 Sub-concept 3
├── 📂 Branch 2
│   ├── 📄 Sub-concept 1
│   └── 📄 Sub-concept 2
└── 📂 Branch 3
    ├── 📄 Sub-concept 1
    └── 📄 Sub-concept 2

Include 4-6 main branches with 2-3 sub-concepts each.
Make it comprehensive and educational.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1000, "temperature": 0.4},
        )
        return {"tree": response.text.strip(), "topic": topic}
    except Exception as e:
        return {"error": f"Concept tree generation failed: {str(e)}"}
