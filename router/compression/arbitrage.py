"""Cross-lingual token arbitrage and structured Bi/Tri-Block prompt rewriter."""

import logging
from typing import Any

from router.compression.scanner import should_trigger_arbitrage

logger = logging.getLogger(__name__)

# System instructions for the local SLM rewriter
ARBITRAGE_SYSTEM_PROMPT = """You are an edge-side prompt optimizer and translation compiler.
Your task is to translate the user prompt to clear, highly token-efficient English, stripping conversational filler while strictly preserving all technical details, function names, types, constraints, and assert lines.

Output EXACTLY in one of two structured bracket formats:

Format 1 (Bi-Block for code & tests):
[CONTEXT]
<Brief domain or repository context>

[TASK]
<Precise, distilled English instruction with function signatures, types, and logic>
assert <Verbatim target assert lines if present>

Format 2 (Tri-Block for general/constrained tasks):
[CONTEXT]
<Brief environment or module context>

[TASK]
<Precise English task instruction>

[CONSTRAINTS]
- <Explicit constraint 1>
- <Explicit constraint 2>

DO NOT output markdown code fences (```python) or solutions. Output ONLY the bracket-formatted prompt."""


def validate_light(rewritten: str, original: str, required_markers: list[str] | None = None) -> bool:
    """Validate that the SLM output is structurally sound and achieves token efficiency.

    Args:
        rewritten: The rewritten prompt from the SLM.
        original: The original user prompt.
        required_markers: Expected bracket markers (e.g. ['[CONTEXT]', '[TASK]']).

    Returns:
        True if the rewrite passes all validation checks, False otherwise.
    """
    if not rewritten or len(rewritten.strip()) < 15:
        return False

    # Prevent premature code solutions leaking in the prompt stage
    if "```python" in rewritten or "<solution>" in rewritten:
        return False

    # Check structural markers
    markers = required_markers or ["[CONTEXT]", "[TASK]"]
    for marker in markers:
        if marker not in rewritten:
            return False

    # For long prompts (>80 estimated tokens), verify it doesn't inflate token budget
    est_orig = sum(1.0 if ord(c) > 127 else 0.25 for c in original)
    est_rewritten = sum(1.0 if ord(c) > 127 else 0.25 for c in rewritten)

    if est_orig > 80 and est_rewritten > (est_orig * 1.05):
        logger.debug(
            f"Arbitrage validation failed: est_rewritten={est_rewritten:.1f} exceeds orig={est_orig:.1f}"
        )
        return False

    return True


class CrossLingualArbitrage:
    """Cross-Lingual Token Arbitrage Engine using a lightweight local SLM."""

    def __init__(
        self,
        slm_model: str = "qwen2.5:3b",
        multilingual_threshold: float = 0.15,
        max_retries: int = 2,
    ):
        self.slm_model = slm_model
        self.threshold = multilingual_threshold
        self.max_retries = max_retries

    async def rewrite_if_needed(
        self,
        text: str,
        backend: Any,
        is_code_task: bool = False,
    ) -> tuple[str, bool]:
        """Inspect and conditionally rewrite multilingual prompts via local SLM.

        Args:
            text: Original prompt text.
            backend: LLM backend to execute the lightweight SLM call.
            is_code_task: If True, target Bi-Block format; otherwise Tri-Block format.

        Returns:
            Tuple of (final_text, was_rewritten).
        """
        # Fast non-ASCII scanner check
        if not should_trigger_arbitrage(text, self.threshold):
            return text, False

        if not backend:
            return text, False

        required_markers = ["[CONTEXT]", "[TASK]"]
        if not is_code_task:
            required_markers.append("[CONSTRAINTS]")

        # Attempt rewrite with SLM
        for attempt in range(self.max_retries + 1):
            try:
                response = await backend.chat(
                    model=self.slm_model,
                    messages=[
                        {"role": "system", "content": ARBITRAGE_SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    stream=False,
                    temperature=0.1 + (0.2 * attempt),  # slight variation on repair
                    max_tokens=600,
                )

                content = ""
                if isinstance(response, dict) and "choices" in response:
                    content = response["choices"][0]["message"]["content"]
                elif hasattr(response, "choices"):
                    content = response.choices[0].message.content

                content = content.strip()
                if validate_light(content, text, required_markers):
                    logger.info(f"Cross-lingual arbitrage succeeded for {len(text)} chars (attempt {attempt + 1})")
                    return content, True
                else:
                    logger.warning(f"Arbitrage rewrite attempt {attempt + 1} failed validation")
            except Exception as e:
                logger.warning(f"Arbitrage SLM call failed (attempt {attempt + 1}): {e}")

        # Fallback to original prompt
        logger.info("Arbitrage failed validation or execution; gracefully falling back to raw prompt")
        return text, False
