"""Content-Addressable Recovery (CAR) for reversible context compression."""

import hashlib
import time
from typing import NamedTuple


class CachedChunk(NamedTuple):
    content: str
    created_at: float
    token_count: int


class ContentAddressableRecovery:
    """In-memory reversible context index with 16-character SHA-256 reference handles."""

    def __init__(self, max_entries: int = 5000, ttl_seconds: int = 86400):
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._store: dict[str, CachedChunk] = {}

    def store_chunk(self, content: str) -> str:
        """Store raw content chunk and return a 16-character deterministic hex reference handle."""
        if not content:
            return ""

        hash_handle = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        est_tokens = max(1, int(len(content) / 3.8))

        # Evict old entries if capacity reached
        if len(self._store) >= self.max_entries:
            oldest_key = min(self._store.keys(), key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]

        self._store[hash_handle] = CachedChunk(
            content=content,
            created_at=time.time(),
            token_count=est_tokens,
        )
        return hash_handle

    def get_chunk(self, ref_handle: str) -> str | None:
        """Retrieve original raw content chunk by its reference handle."""
        clean_handle = ref_handle.strip().replace("[Ref: ", "").replace("]", "")
        item = self._store.get(clean_handle)
        if item:
            # Check TTL
            if (time.time() - item.created_at) < self.ttl:
                return item.content
            else:
                del self._store[clean_handle]
        return None

    def embed_reference_marker(self, content: str) -> str:
        """Store chunk and return a formatted markdown reference marker."""
        handle = self.store_chunk(content)
        return f"[Ref: {handle}]"
