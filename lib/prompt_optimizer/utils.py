"""
Shared utility functions for prompt optimizer.

Provides centralized functions for:
- YAML frontmatter parsing
- Agent/skill path resolution
- Common file operations
"""

from pathlib import Path
from typing import Optional

from .validation import validate_name, contained_path


def parse_markdown_prompt(content: str) -> str:
    """
    Extract prompt content from markdown, handling YAML frontmatter.

    Args:
        content: Raw file content that may include YAML frontmatter

    Returns:
        Clean prompt content without frontmatter

    Raises:
        ValueError: If file has frontmatter but no content after it
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            prompt = parts[2].strip()
            if not prompt:
                raise ValueError("Empty prompt after frontmatter")
            return prompt
    return content.strip()


def find_agent_path(agent_name: str, project_dir: Optional[Path] = None) -> Path:
    """
    Find agent file with consistent fallback logic.

    Searches in order:
    1. User global agents (~/.claude/agents/)
    2. Project agents (.claude/agents/)

    Args:
        agent_name: Name of the agent (without .md extension)
        project_dir: Optional project directory to search (defaults to cwd)

    Returns:
        Path to the agent file

    Raises:
        FileNotFoundError: If agent not found in any location
    """
    project_dir = project_dir or Path.cwd()

    # Reject traversal/injection in the name, then enforce that each resolved
    # path stays inside its intended agents directory.
    validate_name(agent_name, kind="agent name")
    paths = [
        contained_path(Path.home() / ".claude" / "agents", f"{agent_name}.md"),
        contained_path(project_dir / ".claude" / "agents", f"{agent_name}.md"),
    ]

    for p in paths:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Agent not found: {agent_name}\n"
        f"Searched: {', '.join(str(p) for p in paths)}"
    )


def find_skill_path(skill_name: str, project_dir: Optional[Path] = None) -> Path:
    """
    Find skill file with consistent fallback logic.

    Searches in order:
    1. User global skills (~/.claude/skills/<name>/SKILL.md)
    2. Project skills (.claude/skills/<name>/SKILL.md)
    3. Direct file (~/.claude/skills/<name>.md)

    Args:
        skill_name: Name of the skill
        project_dir: Optional project directory to search (defaults to cwd)

    Returns:
        Path to the skill file

    Raises:
        FileNotFoundError: If skill not found in any location
    """
    project_dir = project_dir or Path.cwd()

    # Reject traversal/injection in the name. The directory-style skill paths
    # (<base>/<name>/SKILL.md) are contained against the skills root; the flat
    # variant (<base>/<name>.md) against the same root.
    validate_name(skill_name, kind="skill name")
    home_skills = Path.home() / ".claude" / "skills"
    project_skills = project_dir / ".claude" / "skills"
    paths = [
        contained_path(home_skills, skill_name, "SKILL.md"),
        contained_path(project_skills, skill_name, "SKILL.md"),
        contained_path(home_skills, f"{skill_name}.md"),
    ]

    for p in paths:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Skill not found: {skill_name}\n"
        f"Searched: {', '.join(str(p) for p in paths)}"
    )


def load_prompt_file(path: Path) -> str:
    """
    Load and parse a prompt file, handling YAML frontmatter.

    Convenience function that combines file reading with frontmatter parsing.

    Args:
        path: Path to the prompt file

    Returns:
        Clean prompt content

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file has frontmatter but empty content
    """
    with open(path) as f:
        content = f.read()
    return parse_markdown_prompt(content)
