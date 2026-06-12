#!/usr/bin/env python3
"""
Deploy optimized prompts to production agent files.

Injects few-shot examples from optimized prompts into the agent
markdown files in ~/.claude/agents/.

Usage:
    # Deploy all approved agents
    python deploy_optimized_prompts.py --agents security-auditor,performance-analyzer,test-writer

    # Preview without making changes
    python deploy_optimized_prompts.py --agents security-auditor --dry-run
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

# Make the optimizer library importable for the shared validation helpers.
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from prompt_optimizer.validation import validate_name, contained_path, ValidationError


def _safe_fence(*bodies: str) -> str:
    """
    Return a backtick fence longer than any run of backticks in the bodies.

    Prevents demo/model text containing ``` from breaking out of its code fence
    and becoming live instruction text in the deployed agent prompt.
    """
    max_ticks = 0
    for body in bodies:
        for m in re.finditer(r"`+", body or ""):
            max_ticks = max(max_ticks, len(m.group(0)))
    return "`" * max(3, max_ticks + 1)


def load_optimized_prompt(agent_name: str) -> dict:
    """Load the optimized prompt JSON for an agent."""
    # agent_name is already validated by the caller, but re-validate and enforce
    # path containment here as defense in depth.
    validate_name(agent_name, kind="agent name")
    prompt_dir = Path.home() / ".claude/prompt_optimizer/prompts"
    latest_path = contained_path(prompt_dir, f"{agent_name}_latest.json")

    if not latest_path.exists():
        raise FileNotFoundError(f"No optimized prompt found for {agent_name}")

    with open(latest_path) as f:
        return json.load(f)


def format_demo_section(demos: list) -> str:
    """Format demos as a markdown section."""
    if not demos:
        return ""

    lines = [
        "",
        "## Few-Shot Examples",
        "",
        "These examples demonstrate the expected output format:",
        "",
    ]

    for i, demo in enumerate(demos, 1):
        # Schema guard: demo content is untrusted (it originates from model
        # output). Coerce/validate types so malformed JSON can't crash deploy or
        # smuggle non-string payloads into the agent file.
        raw_in = demo.get("input_text")
        raw_out = demo.get("output_text")
        if not isinstance(raw_in, str) or not isinstance(raw_out, str):
            # Skip malformed demos rather than crashing the whole deploy.
            continue
        input_text = raw_in[:500] + ("..." if len(raw_in) > 500 else "")
        output_text = raw_out[:1000] + ("..." if len(raw_out) > 1000 else "")

        in_fence = _safe_fence(input_text)
        out_fence = _safe_fence(output_text)

        lines.append(f"### Example {i}")
        lines.append("")
        lines.append("**Input:**")
        lines.append(in_fence)
        lines.append(input_text)
        lines.append(in_fence)
        lines.append("")
        lines.append("**Output:**")
        lines.append(out_fence)
        lines.append(output_text)
        lines.append(out_fence)
        lines.append("")
        score = demo.get("score")
        if isinstance(score, (int, float)):
            lines.append(f"*Score: {float(score):.3f}*")
        lines.append("")

    return "\n".join(lines)


def inject_demos_to_agent(
    agent_name: str,
    demos: list,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """Inject few-shot demos into agent markdown file."""
    # Validate the name and enforce that the resolved path stays inside the
    # agents directory (blocks ../ traversal and absolute-path injection).
    validate_name(agent_name, kind="agent name")
    agents_dir = Path.home() / ".claude/agents"
    # Raw (unresolved) path: used for the symlink check so we inspect the link
    # itself, not its target. contained_path() resolves symlinks, so checking
    # is_symlink() on its result would always be False and never fire.
    raw_agent_path = (agents_dir.expanduser() / f"{agent_name}.md")
    agent_path = contained_path(agents_dir, f"{agent_name}.md")

    if not agent_path.exists():
        if verbose:
            print(f"Error: Agent file not found: {agent_path}")
        return False

    # Refuse to operate on a symlinked agent file: even though contained_path
    # confirms the *resolved* target stays in-tree, a symlink whose target is
    # later swapped is a footgun, and we never want to overwrite through a link.
    if raw_agent_path.is_symlink():
        if verbose:
            print(f"Error: refusing to modify symlinked agent file: {raw_agent_path}")
        return False

    # Read current content. Keep the ORIGINAL bytes immutable so the backup is a
    # true snapshot of the pre-deploy file (the previous code backed up the
    # already-mutated content, making restore impossible).
    with open(agent_path, encoding="utf-8") as f:
        original_content = f.read()
    content = original_content

    # Check if demos already exist
    if "## Few-Shot Examples" in content:
        # Remove existing section
        start = content.find("## Few-Shot Examples")
        # Find next ## or end of file
        rest = content[start + len("## Few-Shot Examples"):]
        next_section = rest.find("\n## ")
        if next_section == -1:
            content = content[:start].rstrip()
        else:
            content = content[:start].rstrip() + rest[next_section:]

    # Add demo section before the final guidelines (or at end)
    demo_section = format_demo_section(demos)

    # Find a good insertion point (before Guidelines if it exists)
    if "## Guidelines" in content:
        idx = content.find("## Guidelines")
        new_content = content[:idx].rstrip() + "\n" + demo_section + "\n" + content[idx:]
    else:
        new_content = content.rstrip() + "\n" + demo_section

    if dry_run:
        if verbose:
            print(f"Would update: {agent_path}")
            print(f"  - Adding {len(demos)} few-shot examples")
            print(f"  - New length: {len(new_content)} chars")
        return True

    # Backup the ORIGINAL content (not the mutated `content`) before writing.
    # Use microsecond precision + exclusive-create so concurrent deploys cannot
    # clobber each other's backups.
    backup_path = agent_path.with_suffix(
        f".md.bak.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    with open(backup_path, "x", encoding="utf-8") as f:
        f.write(original_content)

    # Write updated content atomically: write to a temp file in the same
    # directory, fsync, then os.replace() so an interrupted run can never leave
    # the production agent file half-written.
    fd, tmp_name = tempfile.mkstemp(dir=str(agent_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, agent_path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    if verbose:
        print(f"Updated: {agent_path}")
        print(f"  - Added {len(demos)} few-shot examples")
        print(f"  - Backup: {backup_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Deploy optimized prompts to agent files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--agents",
        type=str,
        required=True,
        help="Comma-separated list of agents to deploy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print progress",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet and args.verbose

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    # Validate all names up front; reject the whole batch on any bad name so a
    # traversal attempt fails loudly rather than silently skipping.
    try:
        for name in agents:
            validate_name(name, kind="agent name")
    except ValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if verbose:
        print(f"{'='*50}")
        print(f"DEPLOYING OPTIMIZED PROMPTS")
        print(f"{'='*50}")
        print(f"Agents: {', '.join(agents)}")
        print(f"Dry run: {args.dry_run}")
        print()

    results = []
    for agent_name in agents:
        if verbose:
            print(f"Processing: {agent_name}")

        try:
            prompt_data = load_optimized_prompt(agent_name)
            demos = prompt_data.get("demos", [])

            if not demos:
                if verbose:
                    print(f"  Warning: No demos found for {agent_name}")
                results.append((agent_name, False, "no demos"))
                continue

            success = inject_demos_to_agent(
                agent_name=agent_name,
                demos=demos,
                dry_run=args.dry_run,
                verbose=verbose,
            )

            results.append((agent_name, success, f"{len(demos)} demos"))

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            results.append((agent_name, False, str(e)))

    if verbose:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for name, success, note in results:
            status = "✓" if success else "✗"
            print(f"{status} {name}: {note}")

    failed = sum(1 for _, s, _ in results if not s)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
