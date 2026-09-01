"""Dynamic Context-Window Allocation and Semantic Compression Middleware."""

from router.compression.arbitrage import CrossLingualArbitrage, validate_light
from router.compression.cache_aligner import CacheAligner
from router.compression.car import ContentAddressableRecovery
from router.compression.chunker import ContextSpan, chunk_context_code_aware
from router.compression.dca import DynamicContextAllocator, QueryCategory, TokenBudget
from router.compression.force_keep import is_force_keep
from router.compression.pipeline import CompressionResult, ContextCompressionPipeline
from router.compression.scanner import should_trigger_arbitrage
from router.compression.statistical import StatisticalPruner

__all__ = [
    "ContextCompressionPipeline",
    "CompressionResult",
    "DynamicContextAllocator",
    "QueryCategory",
    "TokenBudget",
    "StatisticalPruner",
    "CrossLingualArbitrage",
    "CacheAligner",
    "ContentAddressableRecovery",
    "ContextSpan",
    "chunk_context_code_aware",
    "is_force_keep",
    "should_trigger_arbitrage",
    "validate_light",
]
