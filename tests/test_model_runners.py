"""Parser tests for the repository's mocked agy stream-json fixtures.

These fixtures are useful fail-closed regression coverage, but a sandboxed test
does not establish the live agy input or output contract.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer.model_runners import GeminiModelRunner


def test_gemini_clean_output_parses_mocked_stream_json_fixture():
    """Parse the init/step_update/result shape asserted by the local mock."""
    raw = "\n".join(
        [
            json.dumps({"type": "init", "model": "gemini-3.1-pro"}),
            json.dumps(
                {
                    "type": "step_update",
                    "step_type": "model_message",
                    "text": "intermediate text must not become the result",
                }
            ),
            json.dumps({"type": "result", "result": "## MUST FIX\n1. Real finding."}),
        ]
    )

    assert GeminiModelRunner._clean_output(raw) == "## MUST FIX\n1. Real finding."


def test_gemini_clean_output_uses_only_terminal_result_event():
    raw = "\n".join(
        [
            json.dumps({"type": "init", "credential": "cached"}),
            json.dumps({"type": "step_update", "text": "Using model: banner-like text"}),
            json.dumps({"type": "result", "result": "Actual content starts here."}),
        ]
    )

    cleaned = GeminiModelRunner._clean_output(raw)
    assert cleaned == "Actual content starts here."
    assert "credential" not in cleaned
    assert "Using model" not in cleaned


def test_gemini_clean_output_strips_ansi_from_result():
    raw = json.dumps({"type": "result", "result": "\x1b[32mhello\x1b[0m"})
    assert GeminiModelRunner._clean_output(raw) == "hello"


@pytest.mark.parametrize("raw", ["", "\n\n\n"])
def test_gemini_clean_output_rejects_missing_result(raw):
    with pytest.raises(ValueError, match="terminal result"):
        GeminiModelRunner._clean_output(raw)


def test_gemini_clean_output_rejects_malformed_stream_json():
    raw = '{"type":"init"}\nnot-json\n'
    with pytest.raises(ValueError, match="invalid agy stream-json"):
        GeminiModelRunner._clean_output(raw)
