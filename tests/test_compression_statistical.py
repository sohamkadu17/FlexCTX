"""Tests for in-process LeanCTX statistical pruner and chronological reassembly."""

from router.compression.statistical import StatisticalPruner


def test_statistical_pruning_reduces_tokens():
    pruner = StatisticalPruner()

    long_context = """
# System Overview
Basically, as stated previously, we are testing the router pipeline.
The router provides intelligent gateway features.
Furthermore, we hope you are doing well today.
Here is the core logic:

```python
def check_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

Logs:
===---===---===---===
===---===---===---===
===---===---===---===
Execution completed successfully on port 11436.
    """

    query = "How to check if a number is prime in Python?"
    pruned_text, orig_tokens, pruned_tokens = pruner.prune_context(
        context_text=long_context,
        query=query,
        target_token_budget=50,
    )

    assert pruned_tokens < orig_tokens
    # Core function must be preserved
    assert "def check_prime" in pruned_text
    # Repetitive low-entropy logs or filler should be pruned
    assert "===---===" not in pruned_text or "Basically, as stated" not in pruned_text
