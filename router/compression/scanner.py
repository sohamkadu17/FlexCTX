"""Sub-millisecond character scanner for selective multilingual detection."""

import unicodedata


def should_trigger_arbitrage(text: str, threshold: float = 0.15) -> bool:
    """Check if text contains sufficient non-ASCII / multilingual characters to warrant SLM translation.

    Operating in <0.1ms on CPU, this prevents unnecessary 500-2000ms SLM latency
    on standard English queries.

    Args:
        text: Raw text to inspect.
        threshold: Ratio of non-ASCII characters to trigger translation (default 0.15 = 15%).

    Returns:
        True if multilingual translation should be triggered, False otherwise.
    """
    if not text or len(text) < 30:
        return False

    non_ascii_count = 0
    total_chars = 0

    for char in text:
        # Ignore whitespace in density calculation
        if char.isspace():
            continue
        total_chars += 1

        # Check for non-ASCII or specific unicode category (e.g. Letter, other for CJK / Arabic)
        if ord(char) > 127 or unicodedata.category(char).startswith("Lo"):
            non_ascii_count += 1

    if total_chars == 0:
        return False

    return (non_ascii_count / total_chars) > threshold
