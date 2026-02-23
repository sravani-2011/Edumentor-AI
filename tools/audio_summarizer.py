"""
audio_summarizer.py – Summarize audio files using Gemini's multimodal capabilities.
"""

import google.generativeai as genai
import tempfile
import os


def summarize_audio(audio_bytes: bytes, filename: str, api_key: str) -> dict:
    """
    Summarize an audio file using Gemini's multimodal model.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content.
    filename : str
        Original filename (used to detect MIME type).
    api_key : str
        Gemini API key.

    Returns
    -------
    dict with 'summary' or {"error": "..."}.
    """
    # Determine MIME type
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
    }
    mime_type = mime_map.get(ext, "audio/mpeg")

    genai.configure(api_key=api_key)

    # Save to temp file for upload
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Upload file to Gemini
        audio_file = genai.upload_file(tmp_path, mime_type=mime_type)

        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = """Listen to this audio and provide a comprehensive summary.

Please provide:
## 🎧 Audio Summary
A concise overview of what is discussed in the audio.

## 🔑 Key Points
The main points covered (bullet list).

## 📝 Detailed Notes
Structured notes covering the important concepts discussed.

## ❓ Comprehension Questions
3 questions to test understanding of the audio content.

## 📚 Key Terms
Any important technical terms mentioned, with brief definitions.
"""

        response = model.generate_content(
            [prompt, audio_file],
            generation_config={"max_output_tokens": 2000, "temperature": 0.3},
        )

        return {"summary": response.text}

    except Exception as e:
        return {"error": f"Audio summarization failed: {str(e)}"}
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
