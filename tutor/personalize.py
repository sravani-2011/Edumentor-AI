"""
tutor/personalize.py – Learner profile management and adaptive personalization.

Maintains a lightweight learner model in session state:
  - Profile: name, course, skill_level, goals
  - Session memory: concepts asked, misunderstandings, quiz history
  - Personalization rules that modify prompt behavior
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LearnerProfile:
    """
    Lightweight learner model.

    Educators can extend this with additional fields (e.g., learning_style,
    preferred_language) to further customize the tutoring experience.
    """
    name: str = "Learner"
    course: str = "General"
    skill_level: str = "Intermediate"   # Beginner / Intermediate / Advanced
    goals: str = "Learn and understand the material"

    # Session memory (accumulated during a session)
    concepts_asked: list = field(default_factory=list)
    misunderstanding_flags: list = field(default_factory=list)
    quiz_scores: list = field(default_factory=list)       # [{concept, score, max, timestamp}]
    weak_concepts: list = field(default_factory=list)      # Concepts with repeated low scores

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "course": self.course,
            "skill_level": self.skill_level,
            "goals": self.goals,
        }

    def record_concept(self, concept: str) -> None:
        """Track a concept that was asked about."""
        if concept and concept not in self.concepts_asked:
            self.concepts_asked.append(concept)

    def record_quiz_score(self, concept: str, score: float, max_score: float) -> None:
        """
        Record quiz performance and detect weak areas.

        If a concept scores below 50% twice, it is flagged as a weak concept
        for targeted reinforcement.
        """
        self.quiz_scores.append({
            "concept": concept,
            "score": score,
            "max_score": max_score,
            "percentage": round((score / max_score) * 100, 1) if max_score > 0 else 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Detect weak concepts: if scored <50% on this concept at least twice
        concept_scores = [
            s for s in self.quiz_scores
            if s["concept"] == concept and s["percentage"] < 50
        ]
        if len(concept_scores) >= 2 and concept not in self.weak_concepts:
            self.weak_concepts.append(concept)

    def get_reinforcement_prompt(self) -> str:
        """
        Generate a prompt modifier for reinforcing weak concepts.

        Returns an empty string if no reinforcement is needed.
        """
        if not self.weak_concepts:
            return ""

        weak_list = ", ".join(self.weak_concepts[-3:])  # Last 3 weak concepts
        return (
            f"\n\n🔄 **Reinforcement needed**: The learner has struggled with: {weak_list}. "
            f"If the current question relates to any of these concepts, provide extra "
            f"explanation, a different analogy, and a quick check-for-understanding question."
        )

    def get_quiz_trend(self) -> list[dict]:
        """Return quiz scores sorted by time for trend analysis."""
        return sorted(self.quiz_scores, key=lambda s: s["timestamp"])

    def get_concept_frequency(self) -> dict[str, int]:
        """Return how many times each concept was asked about."""
        freq = {}
        for concept in self.concepts_asked:
            freq[concept] = freq.get(concept, 0) + 1
        return freq
