"""In-process, sub-millisecond LeanCTX deterministic statistical pruner."""

import math
import re
from collections import Counter
from typing import NamedTuple

from router.compression.chunker import ContextSpan, chunk_context_code_aware
from router.compression.force_keep import is_force_keep


# Standard English stopwords and conversational filler terms
FILLER_WORDS = {
    "actually", "basically", "essentially", "literally", "obviously", "clearly",
    "frankly", "honestly", "really", "simply", "just", "very", "somewhat",
    "furthermore", "moreover", "as", "mentioned", "previously", "stated", "above",
    "below", "hope", "well", "please", "note", "remember", "thank", "thanks"
}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "doing", "this", "that", "these", "those", "it", "its"
}


def _tokenize(text: str) -> list[str]:
    """Fast lexical word tokenizer."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())


def _calculate_entropy(tokens: list[str]) -> float:
    """Calculate Shannon entropy over token distribution. Low entropy = repetitive noise/logs."""
    if not tokens:
        return 0.0
    total = len(tokens)
    counts = Counter(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _calculate_jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


class ScoredSpan(NamedTuple):
    span: ContextSpan
    score: float
    force_keep: bool
    index: int
    token_count: int


class StatisticalPruner:
    """High-speed in-process deterministic lexical pruner (LeanCTX-style).

    Operates entirely on CPU in <1ms without neural network or GPU execution overhead.
    """

    def __init__(
        self,
        bm25_weight: float = 0.35,
        overlap_weight: float = 0.25,
        position_weight: float = 0.15,
        entropy_weight: float = 0.15,
        inv_filler_weight: float = 0.10,
        jaccard_threshold: float = 0.85,
    ):
        self.w_bm25 = bm25_weight
        self.w_overlap = overlap_weight
        self.w_pos = position_weight
        self.w_entropy = entropy_weight
        self.w_filler = inv_filler_weight
        self.jaccard_threshold = jaccard_threshold

    def prune_context(
        self,
        context_text: str,
        query: str,
        target_token_budget: int,
    ) -> tuple[str, int, int]:
        """Prune context_text down to target_token_budget while preserving code syntax and chronological ordering.

        Args:
            context_text: Raw prompt context, RAG document, or workspace text.
            query: User's query/instruction to compute relevance against.
            target_token_budget: Maximum allowed tokens for the pruned output.

        Returns:
            Tuple of (pruned_text, original_estimated_tokens, pruned_estimated_tokens).
        """
        if not context_text:
            return "", 0, 0

        # Segment context using AST & code-aware chunker
        spans = chunk_context_code_aware(context_text)
        if not spans:
            return context_text, 0, 0

        total_spans = len(spans)
        query_tokens = set(_tokenize(query)) if query else set()

        # Build term frequencies for BM25
        doc_frequencies: Counter[str] = Counter()
        span_token_lists: list[list[str]] = []
        span_token_sets: list[set[str]] = []
        span_lengths: list[int] = []

        for span in spans:
            toks = _tokenize(span.text)
            span_token_lists.append(toks)
            span_token_sets.append(set(toks))
            span_lengths.append(len(toks))
            for unique_tok in set(toks):
                doc_frequencies[unique_tok] += 1

        avg_len = sum(span_lengths) / total_spans if total_spans > 0 else 1.0
        k1 = 1.5
        b = 0.75

        # Score each span
        scored_spans: list[ScoredSpan] = []
        original_token_count = 0

        for idx, span in enumerate(spans):
            toks = span_token_lists[idx]
            tok_set = span_token_sets[idx]
            est_tokens = max(1, int(len(span.text) / 3.8))
            original_token_count += est_tokens

            # Check force-keep
            force = is_force_keep(span.text)

            # 1. BM25 score
            bm25_score = 0.0
            if query_tokens and toks:
                tf_map = Counter(toks)
                doc_len = len(toks)
                for q_term in query_tokens:
                    if q_term in tf_map:
                        tf = tf_map[q_term]
                        df = doc_frequencies.get(q_term, 1)
                        idf = math.log(1 + (total_spans - df + 0.5) / (df + 0.5))
                        term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
                        bm25_score += term_score

            # 2. Query Overlap (normalized)
            overlap_score = len(tok_set & query_tokens) / (len(query_tokens) + 1e-5) if query_tokens else 0.5

            # 3. Position decay (Head & tail bias)
            rel_pos = idx / max(1, total_spans - 1)
            pos_score = 1.0 - 0.4 * (rel_pos if rel_pos <= 0.5 else (1.0 - rel_pos))

            # 4. Entropy score (Normalized 0..1, typical text 2..5 bits)
            raw_entropy = _calculate_entropy(toks)
            entropy_score = min(1.0, max(0.0, raw_entropy / 4.5))

            # 5. Inverse filler score
            filler_count = sum(1 for t in toks if t in FILLER_WORDS)
            filler_ratio = filler_count / (len(toks) + 1e-5)
            inv_filler_score = max(0.0, 1.0 - filler_ratio * 3.0)

            composite = (
                self.w_bm25 * bm25_score
                + self.w_overlap * overlap_score
                + self.w_pos * pos_score
                + self.w_entropy * entropy_score
                + self.w_filler * inv_filler_score
            )

            # If force-kept, ensure high base score
            if force:
                composite += 10.0

            scored_spans.append(
                ScoredSpan(
                    span=span,
                    score=composite,
                    force_keep=force,
                    index=idx,
                    token_count=est_tokens,
                )
            )

        # If already within budget, return as is
        if original_token_count <= target_token_budget:
            return context_text, original_token_count, original_token_count

        # Sort by score descending for greedy selection & deduplication
        sorted_by_score = sorted(scored_spans, key=lambda s: s.score, reverse=True)

        selected_spans: list[ScoredSpan] = []
        selected_token_sets: list[set[str]] = []
        current_budget = 0

        for item in sorted_by_score:
            # Check near-duplicate Jaccard with already selected items
            tok_set = span_token_sets[item.index]
            is_duplicate = False

            if not item.force_keep and tok_set:
                for existing_set in selected_token_sets:
                    if _calculate_jaccard_similarity(tok_set, existing_set) > self.jaccard_threshold:
                        is_duplicate = True
                        break

            if is_duplicate:
                continue

            # Budget check (always allow force_keep if reasonably possible)
            if current_budget + item.token_count <= target_token_budget or item.force_keep:
                selected_spans.append(item)
                selected_token_sets.append(tok_set)
                current_budget += item.token_count

        # Reassemble in original chronological order to preserve reasoning continuity
        selected_spans.sort(key=lambda s: s.index)

        pruned_text = "\n".join(s.span.text for s in selected_spans)
        pruned_token_count = current_budget

        return pruned_text, original_token_count, pruned_token_count
