"""AST and indentation-aware code chunking to prevent syntax corruption."""

import ast
import re
from typing import NamedTuple


class ContextSpan(NamedTuple):
    """A single segmented span of text with syntactic metadata."""
    text: str
    is_code: bool
    is_heading: bool
    line_start: int


def _split_prose_sentences(prose: str, start_line: int = 1) -> list[ContextSpan]:
    """Split prose text by sentence and paragraph boundaries."""
    spans: list[ContextSpan] = []
    lines = prose.splitlines()
    curr_line = start_line

    for line in lines:
        stripped = line.strip()
        if not stripped:
            curr_line += 1
            continue

        is_heading = stripped.startswith(("#", "===", "---", "**", "##"))
        if is_heading:
            spans.append(ContextSpan(text=line, is_code=False, is_heading=True, line_start=curr_line))
        else:
            # Split sentences by terminal punctuation (.!?), without breaking abbreviations
            sentence_parts = re.split(r"(?<=[.!?])\s+", line)
            for part in sentence_parts:
                part_stripped = part.strip()
                if part_stripped:
                    spans.append(ContextSpan(text=part_stripped, is_code=False, is_heading=False, line_start=curr_line))
        curr_line += 1

    return spans


def _split_code_block(code_block: str, start_line: int = 1) -> list[ContextSpan]:
    """Segment a code block using Python AST when possible, or line-indentation blocks."""
    # Check if this is a python-like code block
    lines = code_block.splitlines()
    if not lines:
        return []

    # If small block (<= 6 lines), keep it whole as one atomic unit
    if len(lines) <= 6:
        return [ContextSpan(text=code_block, is_code=True, is_heading=False, line_start=start_line)]

    # Attempt Python AST parsing
    try:
        # Strip markdown fence if present
        clean_code = code_block
        fence_header = ""
        fence_footer = ""
        if lines[0].startswith("```"):
            fence_header = lines[0]
            clean_code = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
            if lines[-1].startswith("```"):
                fence_footer = "```"

        tree = ast.parse(clean_code)
        code_lines = clean_code.splitlines()
        spans: list[ContextSpan] = []

        if fence_header:
            spans.append(ContextSpan(text=fence_header, is_code=True, is_heading=True, line_start=start_line))

        # Chunk top-level nodes (classes, functions, import blocks, statements)
        last_lineno = 1
        for node in tree.body:
            node_start = getattr(node, "lineno", last_lineno)
            node_end = getattr(node, "end_lineno", node_start)

            node_slice = code_lines[node_start - 1 : node_end]
            node_text = "\n".join(node_slice)
            if node_text.strip():
                spans.append(ContextSpan(text=node_text, is_code=True, is_heading=False, line_start=start_line + node_start - 1))
            last_lineno = node_end + 1

        if fence_footer:
            spans.append(ContextSpan(text=fence_footer, is_code=True, is_heading=True, line_start=start_line + len(lines) - 1))

        if spans:
            return spans
    except Exception:
        # Fallback to indentation block splitting
        pass

    # Indentation block chunker fallback
    spans = []
    curr_chunk: list[str] = []
    curr_start = start_line

    for idx, line in enumerate(lines):
        line_num = start_line + idx
        # If line is not indented and starts a new definition, break chunk
        if curr_chunk and (line.startswith(("def ", "class ", "@", "import ", "from ", "```")) or not line.startswith(" ")):
            if len(curr_chunk) >= 2:
                spans.append(ContextSpan(text="\n".join(curr_chunk), is_code=True, is_heading=False, line_start=curr_start))
                curr_chunk = []
                curr_start = line_num
        curr_chunk.append(line)

    if curr_chunk:
        spans.append(ContextSpan(text="\n".join(curr_chunk), is_code=True, is_heading=False, line_start=curr_start))

    return spans if spans else [ContextSpan(text=code_block, is_code=True, is_heading=False, line_start=start_line)]


def chunk_context_code_aware(text: str) -> list[ContextSpan]:
    """Segment raw context text into syntactic spans, preserving AST code blocks and identifiers.

    Args:
        text: Raw multi-turn prompt, RAG document, or code context.

    Returns:
        List of ContextSpan objects.
    """
    if not text:
        return []

    # Regex for fenced code blocks
    code_fence_regex = re.compile(r"(```[\w-]*\n[\s\S]*?```)", re.MULTILINE)
    spans: list[ContextSpan] = []

    last_idx = 0
    current_line = 1

    for match in code_fence_regex.finditer(text):
        start, end = match.span()
        # Process prose before the code fence
        if start > last_idx:
            prose_part = text[last_idx:start]
            spans.extend(_split_prose_sentences(prose_part, start_line=current_line))
            current_line += prose_part.count("\n")

        # Process the code block
        code_part = match.group(0)
        spans.extend(_split_code_block(code_part, start_line=current_line))
        current_line += code_part.count("\n")
        last_idx = end

    # Trailing prose after the last code fence
    if last_idx < len(text):
        trailing_prose = text[last_idx:]
        spans.extend(_split_prose_sentences(trailing_prose, start_line=current_line))

    return spans
