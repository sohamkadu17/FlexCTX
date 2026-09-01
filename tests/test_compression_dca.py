"""Tests for Dynamic Context-Window Allocation and Query Classification."""

from router.compression.dca import DynamicContextAllocator, QueryCategory


def test_dca_classification():
    dca = DynamicContextAllocator()

    # Code development query
    code_query = "def calculate_matrix_determinant(matrix: list[list[float]]) -> float:"
    assert dca.classify_query(code_query) == QueryCategory.CODE_DEVELOPMENT

    # RAG / Knowledge query
    rag_query = "According to the official documentation and reference manual, what is the API port?"
    assert dca.classify_query(rag_query) == QueryCategory.RAG_KNOWLEDGE

    # Conversational reasoning
    chat_query = "Earlier in step 2, why did we choose that algorithm?"
    assert dca.classify_query(chat_query, history_len=4) == QueryCategory.CONVERSATIONAL_REASONING


def test_dca_budget_allocation():
    dca = DynamicContextAllocator(default_target_limit=8192, default_generation_reserve=2048)

    rag_budget = dca.allocate_budget(QueryCategory.RAG_KNOWLEDGE)
    assert rag_budget.target_limit == 8192
    assert rag_budget.generation_tokens == 2048
    # Strategy A gives large budget to RAG
    assert rag_budget.rag_tokens > rag_budget.history_tokens

    chat_budget = dca.allocate_budget(QueryCategory.CONVERSATIONAL_REASONING)
    # Strategy B gives large budget to history
    assert chat_budget.history_tokens > chat_budget.rag_tokens


def test_dca_nvml_fault_tolerance():
    dca = DynamicContextAllocator()

    # None or non-numeric or negative values must safely return a stable conservative bucket
    assert dca.calculate_dynamic_limit_from_vram(None) in (2048, 4096, 8192)
    assert dca.calculate_dynamic_limit_from_vram(-1.0) == 4096
    assert dca.calculate_dynamic_limit_from_vram("invalid") == 4096


def test_dca_hysteresis_boundary_margins():
    dca = DynamicContextAllocator()
    dca._current_bucket = 4096

    # Free VRAM at 2.6 GB (below 2.7 GB threshold) should NOT trigger step-up to 8k
    assert dca.calculate_dynamic_limit_from_vram(2.6) == 4096

    # Free VRAM at 2.8 GB (above 2.7 GB threshold) triggers step-up to 8k
    assert dca.calculate_dynamic_limit_from_vram(2.8) == 8192

    # Free VRAM drops to 2.5 GB (above 2.3 GB drop threshold) stays at 8k (no oscillation)
    assert dca.calculate_dynamic_limit_from_vram(2.5) == 8192

    # Free VRAM drops to 2.2 GB (below 2.3 GB drop threshold) drops back to 4k
    assert dca.calculate_dynamic_limit_from_vram(2.2) == 4096
