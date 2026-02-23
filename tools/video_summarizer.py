"""
video_summarizer.py – Summarize YouTube videos using transcript + Gemini.
"""

import re
import google.generativeai as genai


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        match = re.search(pat, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_id: str) -> str | None:
    """Fetch transcript for a YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript_list])
        return full_text
    except Exception as e:
        return None


def summarize_video(url: str, api_key: str) -> dict:
    """
    Summarize a YouTube video.

    Returns dict with: title, summary, key_topics, timestamps, practice_questions
    or {"error": "..."} on failure.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return {"error": "Invalid YouTube URL. Please provide a valid link."}

    transcript = get_youtube_transcript(video_id)
    if not transcript:
        return {"error": "Could not fetch transcript. The video may not have captions."}

    # Truncate very long transcripts
    if len(transcript) > 15000:
        transcript = transcript[:15000] + "... [truncated]"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Analyze this YouTube video transcript and provide a structured summary.

TRANSCRIPT:
{transcript}

Please provide:
## 📺 Video Summary
A concise 3-5 sentence overview of what the video covers.

## 🔑 Key Topics
List the main topics covered (5-8 bullet points).

## 📝 Detailed Notes
Structured notes with the most important concepts explained clearly.

## ❓ Practice Questions
Generate 3 practice questions based on the video content to test understanding.

## 🚀 What to Learn Next
3 suggested topics to explore after watching this video.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 2000, "temperature": 0.3},
        )
        return {"summary": response.text, "video_id": video_id}
    except Exception as e:
        return {"error": f"Summarization failed: {str(e)}"}
