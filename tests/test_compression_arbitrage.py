"""Tests for Cross-Lingual Arbitrage validation and Bi/Tri-block formatting."""

from router.compression.arbitrage import validate_light


def test_validate_light_passes_valid_bi_block():
    original = "Bu fonksiyon verilen listenin toplamını döndürmeli ve her elemanı kontrol etmelidir."
    valid_rewrite = """[CONTEXT]
Python list summation task.

[TASK]
Write a function sum_list(items: list[int]) -> int that calculates and returns total sum.
assert sum_list([1, 2, 3]) == 6
"""
    assert validate_light(valid_rewrite, original, ["[CONTEXT]", "[TASK]"])


def test_validate_light_rejects_leaked_solutions():
    original = "Turkish text here."
    invalid_rewrite = """[CONTEXT]
Task context.

[TASK]
Here is the code:
```python
def solve(): return 42
```
"""
    assert not validate_light(invalid_rewrite, original, ["[CONTEXT]", "[TASK]"])


def test_validate_light_rejects_missing_markers():
    original = "Turkish text here."
    missing_marker = "Just plain English without bracket markers."
    assert not validate_light(missing_marker, original, ["[CONTEXT]", "[TASK]"])
