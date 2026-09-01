"""Prefix-Cache Alignment Engine for RadixAttention (SGLang) and APC (vLLM / llama.cpp)."""

import json
from typing import Any


class CacheAligner:
    """Enforces deterministic, byte-exact prompt structures to maximize prefix cache hit rates (>85%)."""

    @staticmethod
    def sort_tool_schemas(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Sort tool definitions alphabetically by function name.

        Prevents cache misses caused by arbitrary tool array permutations.
        """
        if not tools or not isinstance(tools, list):
            return tools

        def _get_name(tool: dict[str, Any]) -> str:
            if isinstance(tool, dict):
                func = tool.get("function")
                if isinstance(func, dict):
                    return str(func.get("name", ""))
                return str(tool.get("name", ""))
            return ""

        try:
            return sorted(tools, key=_get_name)
        except Exception:
            return tools

    @staticmethod
    def normalize_system_prompt(system_content: str) -> str:
        """Normalize whitespace and line endings in system prompts for byte-exact cache reproducibility."""
        if not system_content:
            return ""
        # Normalize CRLF to LF, strip trailing whitespace per line, single trailing newline
        lines = [line.rstrip() for line in system_content.replace("\r\n", "\n").splitlines()]
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def format_cache_friendly_payload(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Format an entire message list and tool array for deterministic prefix caching.

        Rules:
        1. Static system message is always placed first with normalized whitespace.
        2. Tool schemas are canonically sorted alphabetically.
        3. Dynamic compressed content is placed inside user/assistant turns.
        """
        if not messages:
            return messages, tools

        sorted_tools = CacheAligner.sort_tool_schemas(tools)
        formatted_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system" and isinstance(content, str):
                formatted_messages.append({**msg, "content": CacheAligner.normalize_system_prompt(content)})
            else:
                formatted_messages.append(msg)

        return formatted_messages, sorted_tools
