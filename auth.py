"""
auth.py – Simple authentication system for EduMentor AI.
JSON-based user store with password hashing.
"""

import hashlib
import json
import os
from datetime import datetime

# Use /tmp for Streamlit Cloud compatibility
_IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_SERVER_HEADLESS")
USER_DB_PATH = "/tmp/edu_users.json" if _IS_CLOUD else "./edu_users.json"


def _hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict:
    """Load user database from JSON file."""
    if os.path.exists(USER_DB_PATH):
        try:
            with open(USER_DB_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_users(users: dict):
    """Save user database to JSON file."""
    with open(USER_DB_PATH, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str) -> dict:
    """Register a new user. Returns {"ok": True} or {"error": "..."}."""
    if not username or not password:
        return {"error": "Username and password are required."}
    if len(password) < 4:
        return {"error": "Password must be at least 4 characters."}

    users = _load_users()
    if username.lower() in users:
        return {"error": "Username already exists."}

    users[username.lower()] = {
        "display_name": username,
        "password_hash": _hash_password(password),
        "created_at": datetime.now().isoformat(),
        "score": 0,
        "quizzes_completed": 0,
        "challenges_completed": 0,
        "streak": 0,
        "best_streak": 0,
        "topics_covered": [],
    }
    _save_users(users)
    return {"ok": True}


def login_user(username: str, password: str) -> dict:
    """Authenticate a user. Returns {"ok": True, "user": {...}} or {"error": "..."}."""
    users = _load_users()
    user = users.get(username.lower())
    if not user:
        return {"error": "Username not found."}
    if user["password_hash"] != _hash_password(password):
        return {"error": "Incorrect password."}
    return {"ok": True, "user": user}


def update_score(username: str, points: int, category: str = "quiz"):
    """Add points to a user's score."""
    users = _load_users()
    user = users.get(username.lower())
    if not user:
        return
    user["score"] = user.get("score", 0) + points
    if category == "quiz":
        user["quizzes_completed"] = user.get("quizzes_completed", 0) + 1
    elif category == "challenge":
        user["challenges_completed"] = user.get("challenges_completed", 0) + 1
    user["streak"] = user.get("streak", 0) + 1
    user["best_streak"] = max(user.get("best_streak", 0), user["streak"])
    _save_users(users)


def get_leaderboard(top_n: int = 20) -> list[dict]:
    """Return top N users sorted by score."""
    users = _load_users()
    board = []
    for uname, data in users.items():
        board.append({
            "username": data.get("display_name", uname),
            "score": data.get("score", 0),
            "quizzes": data.get("quizzes_completed", 0),
            "challenges": data.get("challenges_completed", 0),
            "streak": data.get("best_streak", 0),
        })
    board.sort(key=lambda x: x["score"], reverse=True)
    return board[:top_n]
