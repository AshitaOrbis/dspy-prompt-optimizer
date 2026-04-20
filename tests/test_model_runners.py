"""Tests for model runner output parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer.model_runners import GeminiModelRunner


def test_gemini_clean_output_strips_yolo_preamble():
    raw = (
        "YOLO mode is enabled. All tool calls will be automatically approved.\n"
        "\n"
        "Loaded cached credentials.\n"
        "\n"
        "## MUST FIX\n"
        "1. Real finding here.\n"
    )
    cleaned = GeminiModelRunner._clean_output(raw)
    assert cleaned.startswith("## MUST FIX"), f"Got: {cleaned[:80]!r}"
    assert "YOLO" not in cleaned
    assert "Loaded cached credentials" not in cleaned


def test_gemini_clean_output_preserves_plain_output():
    raw = "## MUST FIX\n1. Finding.\n"
    assert GeminiModelRunner._clean_output(raw).startswith("## MUST FIX")


def test_gemini_clean_output_strips_ansi():
    raw = "\x1b[32mhello\x1b[0m"
    assert GeminiModelRunner._clean_output(raw) == "hello"


def test_gemini_clean_output_handles_empty():
    assert GeminiModelRunner._clean_output("") == ""
    assert GeminiModelRunner._clean_output("\n\n\n") == ""


def test_gemini_clean_output_strips_multiple_preamble_types():
    raw = (
        "YOLO mode is enabled.\n"
        "Loaded cached credentials.\n"
        "Data collection is disabled.\n"
        "Using model: gemini-3.1-pro-preview\n"
        "\n"
        "Actual content starts here.\n"
    )
    cleaned = GeminiModelRunner._clean_output(raw)
    assert cleaned == "Actual content starts here."


if __name__ == "__main__":
    test_gemini_clean_output_strips_yolo_preamble()
    print("PASS: strips YOLO preamble")
    test_gemini_clean_output_preserves_plain_output()
    print("PASS: preserves plain output")
    test_gemini_clean_output_strips_ansi()
    print("PASS: strips ANSI")
    test_gemini_clean_output_handles_empty()
    print("PASS: handles empty")
    test_gemini_clean_output_strips_multiple_preamble_types()
    print("PASS: strips multiple preamble types")
    print("\nAll Gemini parser tests passed!")
