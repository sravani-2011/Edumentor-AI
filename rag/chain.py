"""
rag/chain.py – LangChain RAG chain with tutor persona and prompt templates.

Constructs the answer using retrieved context, learner profile, and a
carefully designed system prompt that enforces citation, no-fabrication,
and adaptive explanation style.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.config import CHAT_MODEL, DEFAULT_TEMPERATURE


# ---------------------------------------------------------------------------
# System prompt – Tutor Persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are **EduMentor**, a patient, encouraging, and knowledgeable AI tutor.

### Your Core Rules:
1. **Always cite your sources.** When you use information from the provided context, reference the source PDF name and page number in parentheses, e.g. (Source: Intro_to_ML.pdf, Page 5).
2. **Never fabricate information.** If the provided context does not contain enough information to answer, say so honestly and suggest what the learner could read or search for.
3. **Ask clarifying questions** when the learner's question is ambiguous.
4. **Encourage learning** – check understanding, offer analogies, and reinforce key concepts.
5. **Be structured** – use clear headings, bullet points, and numbered steps.

### Personalization:
{personalization_rules}

### Response Format:
Always structure your response with these sections (use markdown):
- **📝 Answer**: A clear, stepwise explanation.
- **🔑 Key Concepts**: 2-3 core concepts involved.
- **💡 Example**: A concrete example or analogy.
- **⚠️ Common Mistakes**: 1-2 mistakes learners typically make.
- **📚 Citations**: Source references from the context.
- **🚀 What to Learn Next**: 3 bullet points suggesting follow-up topics.

If you are uncertain or the context is insufficient, respond with:
- **🤔 I'm not sure about this**: Explain what you don't know and why.
- **📖 Suggested Reading**: Suggest specific pages or topics to review.
- **❓ Clarifying Question**: Ask the learner for more details.
"""

# ---------------------------------------------------------------------------
# RAG Answer Prompt Template
# ---------------------------------------------------------------------------
RAG_ANSWER_TEMPLATE = """### Learner Profile
- **Name**: {learner_name}
- **Skill Level**: {skill_level}
- **Course**: {course}
- **Goals**: {goals}

### Retrieved Context (use this to answer – cite sources!)
{context}

### Learner's Question
{question}

Please answer following your tutor persona rules and response format above."""


def build_context_string(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string with source annotations."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        score = chunk.get("score", 0)
        parts.append(
            f"--- Chunk {i} (Source: {source}, Page: {page}, Relevance: {score}) ---\n"
            f"{chunk['content']}\n"
        )
    return "\n".join(parts) if parts else "(No relevant context found.)"


def get_rag_answer(
    question: str,
    chunks: list[dict],
    is_confident: bool,
    learner_profile: dict,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    explain_simply: bool = False,
    verbosity: int = 5,
    language: str = "English",
) -> str:
    """
    Generate a tutor-style answer using retrieved context.

    Parameters
    ----------
    question : str
        The learner's question.
    chunks : list[dict]
        Retrieved chunks from the retriever.
    is_confident : bool
        Whether retrieval confidence is above threshold.
    learner_profile : dict
        {name, skill_level, course, goals}
    api_key : str
        OpenAI API key.
    temperature : float
        LLM temperature (0 = deterministic, 1 = creative).
    explain_simply : bool
        If True, override skill level to explain like learner is 12 years old.
    verbosity : int
        1-10 scale; lower = more concise, higher = more detailed.

    Returns
    -------
    str
        The formatted tutor response.
    """
    # Build personalization rules based on skill level
    skill = learner_profile.get("skill_level", "Intermediate")
    if explain_simply:
        personalization = (
            "The learner has toggled 'Explain Like I'm 12'. Use very simple language, "
            "fun analogies, short sentences, and relatable everyday examples. "
            "Avoid jargon entirely."
        )
    elif skill == "Beginner":
        personalization = (
            "The learner is a **Beginner**. Use simple language, provide analogies "
            "to everyday concepts, include mini examples, and define any technical terms."
        )
    elif skill == "Advanced":
        personalization = (
            "The learner is **Advanced**. Be concise and formal. Include edge cases, "
            "deeper reasoning, trade-offs, and assume familiarity with foundational concepts."
        )
    else:
        personalization = (
            "The learner is **Intermediate**. Balance clarity with depth. "
            "Define terms only when they might be unfamiliar."
        )

    # Add verbosity guidance
    personalization += f"\n\nVerbosity level: {verbosity}/10. "
    if verbosity <= 3:
        personalization += "Keep responses very short and to the point."
    elif verbosity >= 8:
        personalization += "Provide detailed, thorough explanations with many examples."

    # Add low-confidence guidance
    if not is_confident:
        personalization += (
            "\n\n⚠️ IMPORTANT: The retrieval confidence is LOW for this query. "
            "You may not have enough context to answer accurately. Follow your "
            "'I'm not sure' protocol: be transparent, suggest reading material, "
            "and ask a clarifying question."
        )

    # Add language instruction
    if language != "English":
        personalization += (
            f"\n\n🌍 LANGUAGE REQUIREMENT: You MUST respond entirely in **{language}**. "
            f"All section headers, explanations, bullet points, and examples should "
            f"be in {language}. Only keep technical terms and source citations in English."
        )

    # Build messages
    system_msg = SystemMessage(content=SYSTEM_PROMPT.format(personalization_rules=personalization))
    context_str = build_context_string(chunks)
    human_msg = HumanMessage(
        content=RAG_ANSWER_TEMPLATE.format(
            learner_name=learner_profile.get("name", "Learner"),
            skill_level=skill,
            course=learner_profile.get("course", "General"),
            goals=learner_profile.get("goals", "Learn and understand"),
            context=context_str,
            question=question,
        )
    )

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
        google_api_key=api_key,
    )

    response = llm.invoke([system_msg, human_msg])
    return response.content
