"""Unit and integration tests for Persistent Vector Conversation Memory."""

import os
import tempfile
import pytest

from router.memory.vector_store import LocalVectorStore, _cosine_similarity
from router.memory.memory_manager import ConversationMemoryManager, _deterministic_hash_vector


def test_cosine_similarity():
    """Verify vector math correctness."""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    assert _cosine_similarity(vec_a, vec_b) == pytest.approx(1.0)

    vec_c = [0.0, 1.0, 0.0]
    assert _cosine_similarity(vec_a, vec_c) == pytest.approx(0.0)


def test_deterministic_hash_vector():
    """Verify sub-millisecond hash vectorizer produces normalized vectors."""
    vec = _deterministic_hash_vector("prime numbers in python")
    assert len(vec) == 128
    norm = sum(x * x for x in vec) ** 0.5
    assert norm == pytest.approx(1.0, rel=1e-3)


@pytest.mark.asyncio
async def test_vector_store_persistence_and_query():
    """Verify local SQLite vector store insertion and semantic retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        store = LocalVectorStore(db_path=db_path)

        # Insert conversation turns
        vec_prime = _deterministic_hash_vector("create python file for prime numbers")
        vec_db = _deterministic_hash_vector("sqlite database connection pool and transactions")
        vec_auth = _deterministic_hash_vector("user login authentication token bearer")

        store.insert(
            content="User: create a python file for prime numbers\nAssistant: def prime_numbers(n): ...",
            role="conversation_turn",
            session_id="test_sess",
            embedding=vec_prime,
        )
        store.insert(
            content="User: configure database pool\nAssistant: class StoragePool: ...",
            role="conversation_turn",
            session_id="test_sess",
            embedding=vec_db,
        )
        store.insert(
            content="User: how to authenticate\nAssistant: pass Bearer token in header",
            role="conversation_turn",
            session_id="test_sess",
            embedding=vec_auth,
        )

        assert store.count() == 3

        # Query for prime numbers
        query_vec = _deterministic_hash_vector("prime numbers in python")
        results = store.query_similar(
            query_embedding=query_vec,
            query_text="prime numbers in python",
            top_k=2,
            min_similarity=0.15,
        )

        assert len(results) >= 1
        assert "prime_numbers" in results[0]["content"]


@pytest.mark.asyncio
async def test_conversation_memory_manager_recall_and_injection():
    """Verify high-level Memory Manager flow and prompt injection formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        manager = ConversationMemoryManager(db_path=db_path, ollama_url="http://invalid-url:9999")

        # Store past turn
        await manager.store_turn(
            content="User: WAP to print first n prime numbers in d:\\AI_CP\\main.py\nAssistant: def prime_numbers(n): return [2, 3, 5]",
            role="conversation_turn",
            session_id="dev_session",
        )

        # Recall for a historical query
        recalled = await manager.recall_relevant_context(
            query="prime numbers in python",
            top_k=1,
            min_similarity=0.15,
        )

        assert len(recalled) == 1
        formatted = manager.format_memory_injection(recalled)

        assert "[RECALLED CONVERSATION MEMORY]" in formatted
        assert "prime numbers" in formatted
        assert "[/RECALLED CONVERSATION MEMORY]" in formatted

        stats = manager.get_stats()
        assert stats["total_memories_stored"] == 1
        assert stats["total_memories_recalled"] == 1
