"""Deterministic Force-Keep Rule Engine to prevent state, secret, and syntax dropping."""

import re

# Compiled regex patterns for critical structural lines that must NEVER be pruned
FORCE_KEEP_PATTERNS = [
    # Security, Credentials, and Authentication Tokens
    re.compile(r"(?i)\b(bearer\s+[a-zA-Z0-9_\-\.=]+|api[_-]?key|jwt|access[_-]?token|secret[_-]?key|password|auth[_-]?token)\b"),
    re.compile(r"(?i)(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_\-]{20,})"),

    # File System Paths, URIs, and Endpoints
    re.compile(r"(?i)(https?://[^\s\"'>]+|file://[^\s\"'>]+)"),
    re.compile(r"([A-Za-z]:\\[\w\.\-\\]+|\/(?:etc|usr|var|home|app|tmp|bin|opt)[\w\.\-\/]*)"),
    re.compile(r"\b[\w\.\-]+\.(?:py|js|ts|json|yaml|yml|sql|sh|rs|go|cpp|c|h|md)\b"),

    # Critical Programming Declarations & Signatures
    re.compile(r"^\s*(def\s+\w+|class\s+\w+|async\s+def\s+\w+|fn\s+\w+|func\s+\w+|interface\s+\w+)"),
    re.compile(r"^\s*(import\s+[\w\.]+|from\s+[\w\.]+\s+import|#include|package\s+\w+)"),
    re.compile(r"^\s*(return\b|assert\b|raise\b|throw\b|yield\b)"),

    # Explicit Behavioral and Task Constraints
    re.compile(r"(?i)\b(MUST\b|NEVER\b|REQUIRED\b|DO NOT\b|CRITICAL\b|IMPORTANT\b)"),

    # Block Markers for Bi/Tri Block structures
    re.compile(r"^\[(CONTEXT|TASK|CONSTRAINTS|REF:[a-zA-Z0-9]+)\]", re.IGNORECASE),
]


def is_force_keep(span_text: str) -> bool:
    """Check if a span must be force-kept regardless of its statistical relevance score.

    Args:
        span_text: The string text of the span.

    Returns:
        True if the span contains protected identifiers, secrets, or essential syntax.
    """
    if not span_text:
        return False

    stripped = span_text.strip()
    if not stripped:
        return False

    # Headings and fence markers are always preserved
    if stripped.startswith(("#", "```", "===", "---")):
        return True

    for pattern in FORCE_KEEP_PATTERNS:
        if pattern.search(span_text):
            return True

    return False
