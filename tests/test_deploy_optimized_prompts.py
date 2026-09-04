"""Regression tests for safe, reviewable prompt deployment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "deploy_optimized_prompts.py"
SPEC = importlib.util.spec_from_file_location("deploy_optimized_prompts", SCRIPT_PATH)
assert SPEC and SPEC.loader
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)

BEGIN_MARKER = "<!-- prompt-optimizer:generated:start -->"
END_MARKER = "<!-- prompt-optimizer:generated:end -->"


@pytest.fixture
def agent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    return agents_dir


def _recorded_code_review_demo() -> list[dict[str, object]]:
    return [
        {
            "input_text": "Review this patch.",
            "output_text": (
                "```markdown\n"
                "## Code Review Summary\n"
                "The patch needs one change.\n"
                "```"
            ),
            "score": 0.95,
        }
    ]


def test_redeploy_is_idempotent_with_fenced_code_review_heading(agent_home: Path) -> None:
    agent_path = agent_home / "code-reviewer.md"
    agent_path.write_text(
        "# Code Reviewer\n\nBase instructions.\n\n## Guidelines\n\nKeep findings precise.\n",
        encoding="utf-8",
    )

    assert deploy.inject_demos_to_agent(
        "code-reviewer", _recorded_code_review_demo(), verbose=False
    )
    first_deploy = agent_path.read_text(encoding="utf-8")

    assert deploy.inject_demos_to_agent(
        "code-reviewer", _recorded_code_review_demo(), verbose=False
    )
    second_deploy = agent_path.read_text(encoding="utf-8")

    assert second_deploy == first_deploy
    assert second_deploy.count(BEGIN_MARKER) == 1
    assert second_deploy.count(END_MARKER) == 1
    assert second_deploy.count("## Few-Shot Examples") == 1
    assert second_deploy.count("## Code Review Summary") == 1
    assert second_deploy.count("## Guidelines") == 1


def test_insertion_ignores_fenced_guidelines_heading(agent_home: Path) -> None:
    agent_path = agent_home / "code-reviewer.md"
    agent_path.write_text(
        "# Code Reviewer\n\n"
        "```markdown\n"
        "## Guidelines\n"
        "Fixture text only.\n"
        "```\n\n"
        "## Guidelines\n\n"
        "Live guidance.\n",
        encoding="utf-8",
    )

    assert deploy.inject_demos_to_agent(
        "code-reviewer", _recorded_code_review_demo(), verbose=False
    )
    first_deploy = agent_path.read_text(encoding="utf-8")
    assert deploy.inject_demos_to_agent(
        "code-reviewer", _recorded_code_review_demo(), verbose=False
    )
    second_deploy = agent_path.read_text(encoding="utf-8")

    assert second_deploy == first_deploy
    assert second_deploy.count(BEGIN_MARKER) == 1
    assert second_deploy.index(BEGIN_MARKER) > second_deploy.index("Fixture text only.")


def test_dry_run_emits_exact_unified_diff_without_writing(
    agent_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent_path = agent_home / "security-auditor.md"
    original = "# Security Auditor\n\n## Guidelines\nProtect token=topsecret.\n"
    agent_path.write_text(original, encoding="utf-8")

    assert deploy.inject_demos_to_agent(
        "security-auditor", _recorded_code_review_demo(), dry_run=True
    )

    output = capsys.readouterr().out
    assert f"--- {agent_path}" in output
    assert f"+++ {agent_path} (proposed)" in output
    assert f"+{BEGIN_MARKER}" in output
    assert "+## Few-Shot Examples" in output
    # The dry-run must show exactly what the live path would write, including
    # existing content; it must not apply a separate redaction transform.
    assert " token=topsecret." in output
    assert agent_path.read_text(encoding="utf-8") == original


def test_dry_run_fails_when_every_demo_is_invalid(
    agent_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent_path = agent_home / "test-writer.md"
    original = "# Test Writer\n"
    agent_path.write_text(original, encoding="utf-8")
    invalid_demos = [{"input_text": ["not", "text"], "output_text": None}]

    assert not deploy.inject_demos_to_agent(
        "test-writer", invalid_demos, dry_run=True
    )
    assert "no valid demos" in capsys.readouterr().out.lower()
    assert agent_path.read_text(encoding="utf-8") == original
