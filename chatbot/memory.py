"""
Conversation memory store.
Keeps the last N turns per session for multi-turn context.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time

MAX_TURNS = 10  # Keep last 10 exchanges
SESSION_TTL = 3600  # 1 hour timeout


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class MemoryStore:
    def __init__(self):
        self._sessions: dict[str, list[Turn]] = defaultdict(list)
        self._turn_counts: dict[str, int] = defaultdict(int)  # total user turns

    def add_user_message(self, session_id: str, content: str):
        self._cleanup_stale(session_id)
        self._sessions[session_id].append(Turn(role="user", content=content))
        self._turn_counts[session_id] += 1
        self._trim(session_id)

    def add_assistant_message(self, session_id: str, content: str):
        self._sessions[session_id].append(Turn(role="assistant", content=content))
        self._trim(session_id)

    def get_history(self, session_id: str) -> list[dict]:
        """Return conversation history as list of {role, content} dicts."""
        return [
            {"role": t.role, "content": t.content}
            for t in self._sessions.get(session_id, [])
        ]

    def get_user_turn_count(self, session_id: str) -> int:
        return self._turn_counts.get(session_id, 0)

    def get_last_bot_response(self, session_id: str) -> Optional[str]:
        turns = self._sessions.get(session_id, [])
        for t in reversed(turns):
            if t.role == "assistant":
                return t.content
        return None

    def _trim(self, session_id: str):
        """Keep only the last MAX_TURNS * 2 messages (user + assistant pairs)."""
        turns = self._sessions[session_id]
        if len(turns) > MAX_TURNS * 2:
            self._sessions[session_id] = turns[-(MAX_TURNS * 2):]

    def _cleanup_stale(self, session_id: str):
        """Remove session if last message is older than TTL."""
        turns = self._sessions.get(session_id, [])
        if turns and (time.time() - turns[-1].timestamp) > SESSION_TTL:
            del self._sessions[session_id]
            self._turn_counts[session_id] = 0


# Global singleton
memory = MemoryStore()
