"""Tests for regex security, secrets, paths, and syntax force-keep rules."""

from router.compression.force_keep import is_force_keep


def test_force_keep_secrets_and_tokens():
    assert is_force_keep("Authorization: Bearer sk-ant-api03-abcdef1234567890abcdef1234567890")
    assert is_force_keep("export OPENAI_API_KEY=sk-proj-1234567890abcdef1234567890")
    assert is_force_keep("Set access_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'")


def test_force_keep_paths_and_urls():
    assert is_force_keep("The configuration file is located at /etc/smarterrouter/config.yaml")
    assert is_force_keep("Target database file: C:\\Users\\project\\data\\router.db")
    assert is_force_keep("Fetch updates from https://api.openai.com/v1/models")


def test_force_keep_code_declarations():
    assert is_force_keep("def solve_prime_numbers(limit: int) -> list[int]:")
    assert is_force_keep("class VRAMManager:")
    assert is_force_keep("assert result == [2, 3, 5, 7]")


def test_prose_is_not_force_kept():
    assert not is_force_keep("Basically, in this example, we can see that the performance is good.")
    assert not is_force_keep("Furthermore, as stated previously, the values are standard.")
