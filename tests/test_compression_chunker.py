"""Tests for AST & code-aware chunker preserving syntax and identifiers."""

from router.compression.chunker import chunk_context_code_aware


def test_chunk_code_aware_preserves_identifiers():
    sample_text = """Here is the documentation for the module.
Please inspect the class implementation below.

```python
import os.path
from typing import List

class RouteOptimizer:
    def __init__(self, config_path: str):
        self.config_path = os.path.abspath(config_path)
        self.cache = {}

    def get_route(self, dest: str) -> str:
        return self.cache.get(dest, "default")
```

After checking the code, please run the assertions.
"""
    spans = chunk_context_code_aware(sample_text)
    assert len(spans) >= 2

    # Verify code spans contain intact syntax without period splitting
    code_spans = [s for s in spans if s.is_code]
    assert len(code_spans) >= 1

    combined_code = "\n".join(s.text for s in code_spans)
    assert "import os.path" in combined_code
    assert "self.config_path = os.path.abspath(config_path)" in combined_code
    assert "def get_route" in combined_code
