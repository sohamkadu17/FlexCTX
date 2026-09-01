"""Tests for selective multilingual scanner and threshold gate."""

from router.compression.scanner import should_trigger_arbitrage


def test_english_text_bypasses_arbitrage():
    english_prompt = (
        "Write a clean, optimized Python function to calculate the Fibonacci series up to N elements. "
        "Make sure to include type hints and docstrings."
    )
    assert not should_trigger_arbitrage(english_prompt, threshold=0.15)


def test_short_text_bypasses_arbitrage():
    short_text = "Hola amigo"
    assert not should_trigger_arbitrage(short_text, threshold=0.15)


def test_multilingual_text_triggers_arbitrage():
    turkish_prompt = (
        "Bu Python fonksiyonu verilen bir dizideki en büyük asal sayıyı bulmalı ve "
        "sonucu ekrana yazdırmalıdır. Lütfen kodlama standartlarına dikkat ediniz."
    )
    assert should_trigger_arbitrage(turkish_prompt, threshold=0.05)

    chinese_prompt = "请用Python编写一个快速排序算法，并为每个步骤添加详细的注释和时间复杂度分析。"
    assert should_trigger_arbitrage(chinese_prompt, threshold=0.15)
