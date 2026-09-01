"""Dynamic Context-Window Allocation (DCA) and Query Complexity Classifier."""

import re
from enum import Enum
from typing import NamedTuple


class QueryCategory(str, Enum):
    RAG_KNOWLEDGE = "rag_knowledge"
    CONVERSATIONAL_REASONING = "conversational_reasoning"
    CODE_DEVELOPMENT = "code_development"
    GENERAL = "general"


class TokenBudget(NamedTuple):
    target_limit: int
    system_tokens: int
    history_tokens: int
    rag_tokens: int
    generation_tokens: int


class DynamicContextAllocator:
    """Dynamic Context-Window Allocation (DCA) partitioner."""

    def __init__(
        self,
        default_target_limit: int = 8192,
        default_generation_reserve: int = 2048,
    ):
        self.target_limit = default_target_limit
        self.generation_reserve = default_generation_reserve

    def classify_query(self, prompt: str, history_len: int = 0) -> QueryCategory:
        """Classify incoming query complexity to determine the optimal budget partitioning strategy."""
        prompt_lower = prompt.lower()

        # Check coding markers
        code_markers = [
            "def ", "class ", "function", "import ", "const ", "var ",
            "```", "syntax", "refactor", "bug", "traceback", "exception",
            "assert", "unit test", "pytest", "return "
        ]
        if any(marker in prompt_lower for marker in code_markers):
            return QueryCategory.CODE_DEVELOPMENT

        # Check RAG / Document / Knowledge lookup markers
        rag_markers = [
            "according to", "documentation", "reference", "manual", "source code",
            "file:", "context:", "based on", "excerpt", "search result"
        ]
        if any(marker in prompt_lower for marker in rag_markers) or len(prompt) > 2500:
            return QueryCategory.RAG_KNOWLEDGE

        # Check Conversational Reasoning
        if history_len >= 3 or any(w in prompt_lower for w in ["earlier", "previously", "step 2", "why did you"]):
            return QueryCategory.CONVERSATIONAL_REASONING

        return QueryCategory.GENERAL

    def allocate_budget(
        self,
        category: QueryCategory,
        total_limit: int | None = None,
        reserve_generation: int | None = None,
    ) -> TokenBudget:
        """Dynamically distribute the total token budget based on query category.

        Formula: L_limit = T_system + T_history + T_rag + T_generation
        """
        limit = total_limit or self.target_limit
        gen_reserve = reserve_generation or self.generation_reserve
        usable_context = max(512, limit - gen_reserve)

        # Baseline system prompt allocation
        sys_tokens = 500

        available_for_context = max(256, usable_context - sys_tokens)

        if category == QueryCategory.RAG_KNOWLEDGE:
            # Strategy A: Maximize RAG / context documents (70%), tight history sliding window
            rag_tokens = int(available_for_context * 0.70)
            hist_tokens = available_for_context - rag_tokens
        elif category == QueryCategory.CONVERSATIONAL_REASONING:
            # Strategy B: Maximize chat history (65%), smaller RAG allowance
            hist_tokens = int(available_for_context * 0.65)
            rag_tokens = available_for_context - hist_tokens
        elif category == QueryCategory.CODE_DEVELOPMENT:
            # Balanced code context: 50% workspace RAG/code, 50% history/tests
            rag_tokens = int(available_for_context * 0.50)
            hist_tokens = available_for_context - rag_tokens
        else:
            # General distribution: 40% RAG, 60% history
            rag_tokens = int(available_for_context * 0.40)
            hist_tokens = available_for_context - rag_tokens

        return TokenBudget(
            target_limit=limit,
            system_tokens=sys_tokens,
            history_tokens=hist_tokens,
            rag_tokens=rag_tokens,
            generation_tokens=gen_reserve,
        )
