"""
blog_summarizer.py – Extract and summarize content from blog/article URLs.
"""

import re
import google.generativeai as genai


def fetch_blog_content(url: str) -> str | None:
    """Fetch and extract text content from a blog URL."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try to find main content area
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return None

        # Get text
        text = main.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text if len(text) > 100 else None

    except Exception as e:
        return None


def summarize_blog(url: str, api_key: str) -> dict:
    """
    Extract and summarize a blog/article URL.

    Returns dict with summary content or {"error": "..."}.
    """
    content = fetch_blog_content(url)
    if not content:
        return {"error": "Could not extract content from this URL. Check the link and try again."}

    # Truncate very long articles
    if len(content) > 12000:
        content = content[:12000] + "... [truncated]"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Analyze this blog/article content and provide a structured summary.

ARTICLE CONTENT:
{content}

Please provide:
## 📰 TL;DR
A 2-3 sentence summary of the entire article.

## 🔑 Key Takeaways
5-7 bullet points of the most important information.

## 📝 Detailed Summary
A well-structured summary covering all major points.

## 💡 Important Concepts
List and briefly explain any important technical concepts mentioned.

## 🚀 Related Topics to Explore
3 related topics the reader should explore next.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 2000, "temperature": 0.3},
        )
        return {"summary": response.text, "url": url}
    except Exception as e:
        return {"error": f"Summarization failed: {str(e)}"}
