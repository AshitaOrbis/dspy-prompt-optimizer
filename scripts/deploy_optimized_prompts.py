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
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def load_optimized_prompt(agent_name: str) -> dict:
    """Load the optimized prompt JSON for an agent."""
    prompt_dir = Path.home() / ".claude/prompt_optimizer/prompts"
    latest_path = prompt_dir / f"{agent_name}_latest.json"

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
        lines.append(f"### Example {i}")
        lines.append("")
        lines.append("**Input:**")
        lines.append("```")
        # Truncate very long inputs
        input_text = demo["input_text"][:500] + ("..." if len(demo["input_text"]) > 500 else "")
        lines.append(input_text)
        lines.append("```")
        lines.append("")
        lines.append("**Output:**")
        lines.append("```")
        # Truncate very long outputs
        output_text = demo["output_text"][:1000] + ("..." if len(demo["output_text"]) > 1000 else "")
        lines.append(output_text)
        lines.append("```")
        lines.append("")
        if demo.get("score"):
            lines.append(f"*Score: {demo['score']:.3f}*")
        lines.append("")

    return "\n".join(lines)


def inject_demos_to_agent(
    agent_name: str,
    demos: list,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """Inject few-shot demos into agent markdown file."""
    agent_path = Path.home() / ".claude/agents" / f"{agent_name}.md"

    if not agent_path.exists():
        if verbose:
            print(f"Error: Agent file not found: {agent_path}")
        return False

    # Read current content
    with open(agent_path) as f:
        content = f.read()

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

    # Backup original
    backup_path = agent_path.with_suffix(f".md.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    with open(backup_path, 'w') as f:
        f.write(content)

    # Write updated content
    with open(agent_path, 'w') as f:
        f.write(new_content)

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

    agents = [a.strip() for a in args.agents.split(",")]

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
