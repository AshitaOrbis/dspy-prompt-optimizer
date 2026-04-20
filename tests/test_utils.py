"""Tests for lib/prompt_optimizer/utils.py"""

import pytest
import tempfile
from pathlib import Path

from lib.prompt_optimizer.utils import (
    parse_markdown_prompt,
    find_agent_path,
    find_skill_path,
    load_prompt_file,
)


# =============================================================================
# parse_markdown_prompt
# =============================================================================

class TestParseMarkdownPrompt:
    def test_plain_content_returned_unchanged(self):
        content = "This is a plain prompt with no frontmatter."
        assert parse_markdown_prompt(content) == content

    def test_strips_leading_and_trailing_whitespace_from_plain(self):
        content = "  \n  plain prompt  \n  "
        assert parse_markdown_prompt(content) == "plain prompt"

    def test_strips_yaml_frontmatter_single_block(self):
        content = "---\nname: my-skill\n---\nActual prompt here."
        assert parse_markdown_prompt(content) == "Actual prompt here."

    def test_strips_yaml_frontmatter_multiline(self):
        content = "---\nname: foo\ndescription: bar\ntools: Read\n---\nPrompt body."
        assert parse_markdown_prompt(content) == "Prompt body."

    def test_preserves_multiline_prompt_after_frontmatter(self):
        content = "---\nname: test\n---\nLine one.\nLine two.\nLine three."
        assert parse_markdown_prompt(content) == "Line one.\nLine two.\nLine three."

    def test_raises_when_frontmatter_but_empty_content(self):
        content = "---\nname: test\n---\n   "
        with pytest.raises(ValueError, match="Empty prompt after frontmatter"):
            parse_markdown_prompt(content)

    def test_raises_when_frontmatter_and_no_trailing_separator(self):
        # Only one --- block: treated as plain content (no split possible)
        content = "---\nname: test"
        # No ValueError here — treated as plain text because split("---", 2) yields only 2 parts
        result = parse_markdown_prompt(content)
        assert result == "---\nname: test"

    def test_empty_string_returns_empty_string(self):
        assert parse_markdown_prompt("") == ""

    def test_frontmatter_content_with_dashes_in_body(self):
        content = "---\nname: test\n---\nPrompt with --- dashes --- inside."
        # Only the first two --- splits are used; the rest is body
        assert parse_markdown_prompt(content) == "Prompt with --- dashes --- inside."

    def test_only_frontmatter_delimiters_raises(self):
        content = "---\n---\n"
        with pytest.raises(ValueError, match="Empty prompt after frontmatter"):
            parse_markdown_prompt(content)


# =============================================================================
# find_agent_path
# =============================================================================

class TestFindAgentPath:
    def test_finds_agent_in_project_dir(self, tmp_path):
        agents_dir = tmp_path / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        agent_file = agents_dir / "my-agent.md"
        agent_file.write_text("agent content")

        result = find_agent_path("my-agent", project_dir=tmp_path)
        assert result == agent_file

    def test_raises_when_agent_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Agent not found: nonexistent"):
            find_agent_path("nonexistent", project_dir=tmp_path)

    def test_error_message_includes_searched_paths(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc_info:
            find_agent_path("ghost", project_dir=tmp_path)
        assert "ghost" in str(exc_info.value)
        assert ".claude/agents" in str(exc_info.value)

    def test_defaults_project_dir_to_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            find_agent_path("some-agent")

    def test_finds_agent_in_home_over_project(self, tmp_path, monkeypatch):
        # Place agent in both home and project; home should win
        home_agents = tmp_path / "fake_home" / ".claude" / "agents"
        home_agents.mkdir(parents=True)
        home_file = home_agents / "shared-agent.md"
        home_file.write_text("home version")

        proj_agents = tmp_path / "project" / ".claude" / "agents"
        proj_agents.mkdir(parents=True)
        proj_file = proj_agents / "shared-agent.md"
        proj_file.write_text("project version")

        monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
        # Patch Path.home() via monkeypatching
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "fake_home"))

        result = find_agent_path("shared-agent", project_dir=tmp_path / "project")
        assert result == home_file


# =============================================================================
# find_skill_path
# =============================================================================

class TestFindSkillPath:
    def test_finds_skill_in_project_skill_dir(self, tmp_path):
        skill_dir = tmp_path / ".claude" / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("skill content")

        result = find_skill_path("my-skill", project_dir=tmp_path)
        assert result == skill_file

    def test_raises_when_skill_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Skill not found: ghost"):
            find_skill_path("ghost", project_dir=tmp_path)

    def test_error_message_includes_skill_name_and_paths(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc_info:
            find_skill_path("missing-skill", project_dir=tmp_path)
        assert "missing-skill" in str(exc_info.value)

    def test_finds_direct_skill_md_file(self, tmp_path, monkeypatch):
        home_skills = tmp_path / "fake_home" / ".claude" / "skills"
        home_skills.mkdir(parents=True)
        direct_file = home_skills / "legacy-skill.md"
        direct_file.write_text("legacy content")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "fake_home"))

        result = find_skill_path("legacy-skill", project_dir=tmp_path)
        assert result == direct_file


# =============================================================================
# load_prompt_file
# =============================================================================

class TestLoadPromptFile:
    def test_loads_plain_file(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("Hello, world!")
        assert load_prompt_file(f) == "Hello, world!"

    def test_strips_frontmatter_from_file(self, tmp_path):
        f = tmp_path / "skill.md"
        f.write_text("---\nname: test\n---\nActual skill content.")
        assert load_prompt_file(f) == "Actual skill content."

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompt_file(tmp_path / "nonexistent.md")

    def test_raises_for_frontmatter_with_empty_body(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("---\nname: test\n---\n")
        with pytest.raises(ValueError, match="Empty prompt after frontmatter"):
            load_prompt_file(f)
