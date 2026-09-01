"""Local Persistent Vector Store for Semantic Conversation Memory."""

import json
import logging
import math
import os
import sqlite3
import struct
import time
from typing import Any

logger = logging.getLogger(__name__)


def _pack_vector(vector: list[float]) -> bytes:
    """Pack a float list into binary bytes."""
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack_vector(blob: bytes) -> list[float]:
    """Unpack binary bytes into a float list."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class LocalVectorStore:
    """Lightweight persistent SQLite-based Vector Store with zero external heavy dependencies."""

    def __init__(self, db_path: str = "data/conversation_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    created_at REAL,
                    metadata_json TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_vectors(session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_created ON conversation_vectors(created_at)"
            )
            conn.commit()
        finally:
            conn.close()

    def insert(
        self,
        content: str,
        role: str = "user",
        session_id: str = "default",
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert a conversation turn with its embedding into SQLite."""
        emb_blob = _pack_vector(embedding) if embedding else None
        meta_str = json.dumps(metadata or {})
        now = time.time()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversation_vectors (session_id, role, content, embedding, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, emb_blob, now, meta_str),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            conn.close()

    def query_similar(
        self,
        query_embedding: list[float],
        query_text: str = "",
        top_k: int = 3,
        min_similarity: float = 0.20,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search top-k most similar conversation turns using Cosine Similarity & BM25 overlap."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    """
                    SELECT id, session_id, role, content, embedding, created_at, metadata_json
                    FROM conversation_vectors
                    WHERE session_id = ?
                    ORDER BY id DESC LIMIT 500
                    """,
                    (session_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, session_id, role, content, embedding, created_at, metadata_json
                    FROM conversation_vectors
                    ORDER BY id DESC LIMIT 500
                    """
                )
            rows = cursor.fetchall()
        finally:
            conn.close()

        query_tokens = set(query_text.lower().split()) if query_text else set()
        scored_results = []

        for row in rows:
            row_id, sess, role, content, emb_blob, created_at, meta_json = row
            sim = 0.0

            # 1. Cosine similarity via embeddings
            if emb_blob and query_embedding:
                vec = _unpack_vector(emb_blob)
                sim = _cosine_similarity(query_embedding, vec)

            # 2. Hybrid Lexical boost (keyword matches for exact terms)
            if query_tokens and content:
                content_tokens = set(content.lower().split())
                overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
                # Blend: 70% vector semantic similarity + 30% lexical keyword overlap
                sim = (0.7 * sim) + (0.3 * overlap)

            if sim >= min_similarity:
                try:
                    meta = json.loads(meta_json) if meta_json else {}
                except Exception:
                    meta = {}

                scored_results.append(
                    {
                        "id": row_id,
                        "session_id": sess,
                        "role": role,
                        "content": content,
                        "similarity": round(sim, 4),
                        "created_at": created_at,
                        "metadata": meta,
                    }
                )

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_results[:top_k]

    def count(self) -> int:
        """Get total number of memories stored."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversation_vectors")
            row = cursor.fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
