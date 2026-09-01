"""Conversation Memory Manager with Semantic Vector Embeddings and In-Process Fallback."""

import hashlib
import logging
import math
import os
import re
from typing import Any

import httpx

from router.memory.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)


def _deterministic_hash_vector(text: str, dim: int = 128) -> list[float]:
    """Fast, deterministic in-process sub-millisecond fallback vectorizer."""
    if not text:
        return [0.0] * dim

    words = re.findall(r"\w+", text.lower())
    if not words:
        return [0.0] * dim

    vec = [0.0] * dim
    for word in words:
        # MD5 hash into integer buckets
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 7) & 1 else -1.0
        vec[idx] += sign

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class ConversationMemoryManager:
    """Manages conversational memory ingestion, vector embedding, and context retrieval."""

    def __init__(
        self,
        db_path: str = "data/conversation_memory.db",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
    ):
        self.vector_store = LocalVectorStore(db_path=db_path)
        self.ollama_url = ollama_url.rstrip("/")
        self.embedding_model = embedding_model
        self._total_recalled = 0

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding using local Ollama if available, with instant in-process fallback."""
        if not text.strip():
            return _deterministic_hash_vector("")

        # 1. Try Ollama local embedding endpoint
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text[:1000]},
                )
                if res.status_code == 200:
                    data = res.json()
                    emb = data.get("embedding")
                    if emb and isinstance(emb, list):
                        return emb
        except Exception:
            pass  # Fall back to instant deterministic in-process hash vector

        # 2. Fast deterministic sub-millisecond vectorizer (<0.05ms)
        return _deterministic_hash_vector(text)

    async def store_turn(
        self,
        content: str,
        role: str = "user",
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Asynchronously record a conversation turn and its vector embedding."""
        if not content or len(content.strip()) < 3:
            return 0

        # Don't store purely redundant terminal logs or giant repetitive blocks
        clean_content = content.strip()[:4000]
        emb = await self.get_embedding(clean_content)

        row_id = self.vector_store.insert(
            content=clean_content,
            role=role,
            session_id=session_id,
            embedding=emb,
            metadata=metadata,
        )
        logger.debug(f"Stored conversation turn #{row_id} in vector memory for session '{session_id}'")
        return row_id

    async def recall_relevant_context(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 2,
        min_similarity: float = 0.28,
    ) -> list[dict[str, Any]]:
        """Find the top-k most relevant past conversation turns for a given query."""
        if not query or len(query.strip()) < 3:
            return []

        query_emb = await self.get_embedding(query)
        results = self.vector_store.query_similar(
            query_embedding=query_emb,
            query_text=query,
            top_k=top_k,
            min_similarity=min_similarity,
            session_id=session_id,
        )
        if results:
            self._total_recalled += len(results)
            logger.info(f"Recalled {len(results)} relevant memories for query: '{query[:60]}...'")
        return results

    def format_memory_injection(self, recalled_turns: list[dict[str, Any]]) -> str:
        """Format recalled memories into a clean markdown block for prompt injection."""
        if not recalled_turns:
            return ""

        lines = ["[RECALLED CONVERSATION MEMORY]"]
        for turn in recalled_turns:
            role = turn.get("role", "user").capitalize()
            snippet = turn.get("content", "").replace("\n", " ").strip()
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."
            lines.append(f"• {role} previously stated: {snippet}")
        lines.append("[/RECALLED CONVERSATION MEMORY]")
        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        return {
            "total_memories_stored": self.vector_store.count(),
            "total_memories_recalled": self._total_recalled,
        }
