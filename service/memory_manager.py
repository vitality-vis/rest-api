"""Sliding-window conversation memory and doc cache for the research assistant."""
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Turn:
    role: str          # "user" or "assistant"
    content: str

@dataclass
class MemoryManager:
    turns: List[Turn] = field(default_factory=list)
    doc_cache: List[Dict[str, Any]] = field(default_factory=list)

    # TODO: Replace the fixed sliding window with recent turns plus a compact
    # conversation summary and on-demand retrieval of relevant older history.
    MAX_TURNS: int = 6     # sliding window size (keep last 6 user+assistant pairs)
    MAX_DOCS: int = 20     # store up to 20 retrieved docs

    # ---------------------------
    # Sliding Window Conversation
    # ---------------------------
    def add_turn(self, role: str, content: str):
        """Append one conversation turn and enforce sliding window."""
        self.turns.append(Turn(role, content))
        # Keep last MAX_TURNS * 2 turns (user+assistant pairs)
        self.turns = self.turns[-self.MAX_TURNS * 2:]

    def get_history_text(self) -> str:
        """Return conversation history for rewriting prompt."""
        return "\n".join(
            f"{t.role.upper()}: {t.content}"
            for t in self.turns
        )

    def restore_history(self, history: List[Dict[str, str]]) -> None:
        """Replace in-memory conversation state with validated client history."""
        self.turns = []
        self.doc_cache = []
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                self.add_turn(role, content)

    # ---------------------------
    # Retrieved Docs
    # ---------------------------
    def set_docs(self, docs: List[Any]):
        """Store metadata of retrieved docs."""
        reduced = []
        for d in docs[:self.MAX_DOCS]:
            md = getattr(d, "metadata", {}) or {}
            reduced.append({
                "id": md.get("id"),
                "title": md.get("title"),
                "authors": md.get("authors", [])
            })
        self.doc_cache = reduced

    def clear_docs(self):
        self.doc_cache = []

    # ---------------------------
    # Full Reset
    # ---------------------------
    def clear(self):
        """Reset all memory (conversation + docs)."""
        self.turns = []
        self.doc_cache = []
