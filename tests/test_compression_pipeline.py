"""Integration tests for master ContextCompressionPipeline."""

import pytest

from router.compression.pipeline import ContextCompressionPipeline


@pytest.mark.asyncio
async def test_compression_pipeline_end_to_end():
    pipeline = ContextCompressionPipeline(
        enabled=True,
        mode="full",
        target_context_limit=4096,
    )

    tools = [
        {"type": "function", "function": {"name": "search_files"}},
        {"type": "function", "function": {"name": "edit_file"}},
        {"type": "function", "function": {"name": "analyze_repo"}},
    ]

    messages = [
        {"role": "system", "content": "  You are an expert coding assistant.\r\n"},
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Yes, I am ready to help."},
        {
            "role": "user",
            "content": """
# Project Context
Basically, as stated previously, we are working on the database module.
The project uses SQLite for storage.

```python
import os.path

class DBManager:
    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)

    def connect(self):
        return True
```

Logs:
===---===---===---===
===---===---===---===
Write a function to verify the connection.
""",
        },
    ]

    result = await pipeline.process_chat_payload(
        messages=messages,
        tools=tools,
        backend=None,
    )

    assert result.original_tokens > 0
    assert result.compressed_tokens <= result.original_tokens

    # Verify tool sorting (analyze_repo, edit_file, search_files)
    tool_names = [t["function"]["name"] for t in result.tools]
    assert tool_names == sorted(tool_names)

    # Verify metrics summary
    summary = pipeline.get_metrics_summary()
    assert summary["total_requests_processed"] == 1
    assert summary["total_compressed_tokens"] > 0
