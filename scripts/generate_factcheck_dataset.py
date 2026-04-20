#!/usr/bin/env python3
"""
Generate fact-checker training and holdout datasets.

Reads blog posts and existing factcheck reports. For posts without reports,
generates new factchecks via Codex (GPT-5.4). Outputs JSONL files.

Usage:
    python3 scripts/generate_factcheck_dataset.py
    python3 scripts/generate_factcheck_dataset.py --dry-run
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

POSTS_DIR = Path.home() / "claudeworkspace/applications/ashitaorbis/shared/content/posts"
QA_DIR = Path.home() / ".claude/qa-reports/ashitaorbis"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# Selected posts for fact-check dataset (20 posts)
# Posts with * are holdout
SELECTED_POSTS = [
    # Posts with existing factcheck reports (reuse as gold standard)
    "004-dead-blog-theory",
    "009-ai-evaluating-ai",
    "010-context-window-epistemology",
    "014-behaviorisms-hidden-legacy-in-reinforcement-learning",  # *HOLDOUT*
    "017-i-let-an-ai-loose-on-an-ai-social-network",
    "018-68-turns-of-watching-an-ai-think",
    "019-what-my-ai-learned-on-the-internet",  # *HOLDOUT*
    "022-building-an-ai-that-improves-itself",
    "023-sandboxing-ai-agents-the-embassy-pattern",
    "024-testing-through-the-eyes-of-real-users",

    # Posts needing new factcheck generation
    "003-automating-prompt-engineering",
    "006-what-10000-ai-conversations-reveal",
    "012-red-teaming-your-business-ideas",  # *HOLDOUT*
    "025-the-logistics-gap",
    "028-building-your-own-personality-profile",
    "031-how-to-benchmark-conversation-extraction",  # *HOLDOUT*
    "032-the-etymology-tax",
    "034-benchmarking-bullshit-detection",
    "037-the-model-generation-audit",
    "038-where-to-spend-your-context-window",  # *HOLDOUT*
]

HOLDOUT_SLUGS = {
    "014-behaviorisms-hidden-legacy-in-reinforcement-learning",
    "019-what-my-ai-learned-on-the-internet",
    "012-red-teaming-your-business-ideas",
    "031-how-to-benchmark-conversation-extraction",
    "038-where-to-spend-your-context-window",
}

# Map slug patterns to existing factcheck files
EXISTING_REPORTS = {
    "004-dead-blog-theory": "factcheck-004-dead-blog-theory-2026-02-12.md",
    "009-ai-evaluating-ai": "factcheck-009-ai-evaluating-ai-2026-02-12.md",
    "010-context-window-epistemology": "factcheck-010-context-window-2026-02-12.md",
    "014-behaviorisms-hidden-legacy-in-reinforcement-learning": "factcheck-014-behaviorism-2026-02-12.md",
    "017-i-let-an-ai-loose-on-an-ai-social-network": "factcheck-017-i-let-an-ai-loose-on-an-ai-social-network-2026-02-15.md",
    "018-68-turns-of-watching-an-ai-think": "factcheck-018-68-turns-of-watching-an-ai-think-2026-02-15.md",
    "019-what-my-ai-learned-on-the-internet": "factcheck-019-what-my-ai-learned-on-the-internet-2026-02-15.md",
    "022-building-an-ai-that-improves-itself": "factcheck-022-building-an-ai-that-improves-itself-2026-02-17.md",
    "023-sandboxing-ai-agents-the-embassy-pattern": "factcheck-023-2026-02-17.md",
    "024-testing-through-the-eyes-of-real-users": "factcheck-024-2026-02-17.md",
}

FACTCHECK_PROMPT = """You are a rigorous fact-checker for a technical blog about AI, Claude Code, and autonomous development.

Read this blog post and:
1. Extract ALL verifiable factual claims (historical, technical, statistical, named entities)
2. For each claim, provide a verdict with evidence

## Output Format

# Fact-Check Report: "[title]"

## Claims Extracted

| # | Claim | Type | Paragraph | Verdict | Evidence |
|---|-------|------|-----------|---------|----------|
| 1 | [exact claim text] | [Historical/Technical/Statistical/Attribution] | [N] | [VERIFIED/INACCURATE/MISLEADING/CONTESTED/OVERSIMPLIFIED/UNVERIFIABLE/AUTHOR'S DATA] | [source or reasoning] |

## Summary

- **Total claims**: N
- **Verified**: N
- **Contested/Inaccurate**: N
- **Unverifiable**: N

## Detailed Analysis

### Claim 1: [claim text]
**Verdict**: [verdict]
**Evidence**: [detailed evidence with sources]
**Source**: [URL or reference]

[Repeat for each claim requiring detailed analysis]

## Overall Assessment
[Brief assessment of the post's factual reliability]

---

## Blog Post

{post_content}
"""


def find_existing_report(slug: str) -> str:
    """Try to find an existing factcheck report for this post."""
    if slug in EXISTING_REPORTS:
        path = QA_DIR / EXISTING_REPORTS[slug]
        if path.exists():
            return path.read_text()
    return ""


def generate_factcheck_via_codex(post_path: Path) -> str:
    """Generate a factcheck report using codex exec (GPT-5.4)."""
    post_content = post_path.read_text()
    prompt = FACTCHECK_PROMPT.format(post_content=post_content)

    # Find codex binary
    codex_bin = "codex"
    home = os.path.expanduser("~")
    nvm_dir = os.path.join(home, ".nvm/versions/node")
    if os.path.isdir(nvm_dir):
        for d in sorted(os.listdir(nvm_dir), reverse=True):
            candidate = os.path.join(nvm_dir, d, "bin/codex")
            if os.path.isfile(candidate):
                codex_bin = candidate
                break

    # Detect configured MCP servers to disable
    config_path = os.path.expanduser("~/.codex/config.toml")
    mcp_disable_args = []
    if os.path.isfile(config_path):
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("[mcp_servers.") and line.endswith("]"):
                    name = line[len("[mcp_servers."):-1]
                    if name and "." not in name:
                        mcp_disable_args.extend(["-c", f"mcp_servers.{name}.enabled=false"])

    # Use reasoning_effort=medium based on codex-timeout-investigation results:
    # medium completes 137-283s even for 4K-word posts; xhigh always times out
    # at 480s, high times out on large inputs.
    cmd = [
        codex_bin, "exec", "--full-auto",
        "-c", 'model="gpt-5.4"',
        "-c", 'model_reasoning_effort="medium"',
        *mcp_disable_args,
        "--ephemeral", "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=480,
        )

        if result.returncode != 0:
            print(f"  ERROR: codex returned {result.returncode}")
            print(f"  stderr: {result.stderr[:300]}")
            return ""

        # Clean ANSI codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', result.stdout).strip()

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 480s")
        return ""
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return ""


def generate_factcheck_via_claude(post_path: Path) -> str:
    """Fallback: generate factcheck using claude -p --model opus.

    Used when Codex times out or fails. Uses 600s timeout (more headroom
    than Codex 480s) since we know Claude handles large prompts reliably.
    """
    post_content = post_path.read_text()
    prompt = FACTCHECK_PROMPT.format(post_content=post_content)

    env = {k: v for k, v in os.environ.items()}
    env.pop("CLAUDECODE", None)  # avoid nested-session error

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "opus", "--max-turns", "1"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        if result.returncode != 0:
            print(f"  ERROR: claude returned {result.returncode}: {result.stderr[:300]}")
            return ""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', result.stdout).strip()
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s (claude fallback)")
        return ""
    except Exception as e:
        print(f"  EXCEPTION (claude fallback): {e}")
        return ""


def validate_factcheck(report: str, slug: str) -> bool:
    """Validate a factcheck report meets quality requirements."""
    if len(report) < 300:
        print(f"  FAIL: {slug} report too short ({len(report)} chars)")
        return False

    # Must contain claim-related content
    has_claims = any(kw in report.lower() for kw in ["claim", "verdict", "verified", "factual"])
    if not has_claims:
        print(f"  FAIL: {slug} no claim/verdict content found")
        return False

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate fact-checker dataset")
    parser.add_argument("--dry-run", action="store_true", help="Process first 3 posts only")
    args = parser.parse_args()

    posts_to_process = SELECTED_POSTS[:3] if args.dry_run else SELECTED_POSTS

    training_examples = []
    holdout_examples = []
    failed = []

    for i, slug in enumerate(posts_to_process):
        post_path = POSTS_DIR / f"{slug}.md"
        if not post_path.exists():
            print(f"[{i+1}/{len(posts_to_process)}] SKIP: {slug} (file not found)")
            failed.append(slug)
            continue

        post_content = post_path.read_text()
        word_count = len(post_content.split())

        print(f"\n[{i+1}/{len(posts_to_process)}] {slug} ({word_count} words)")

        # Try existing report first
        report = find_existing_report(slug)
        if report:
            print(f"  Using existing factcheck report ({len(report)} chars)")
        else:
            print(f"  Generating factcheck via Codex (GPT-5.4)...")
            report = generate_factcheck_via_codex(post_path)
            if not report:
                print(f"  Codex failed, falling back to claude -p opus...")
                report = generate_factcheck_via_claude(post_path)

        if not report:
            print(f"  FAILED: Empty report")
            failed.append(slug)
            continue

        if not validate_factcheck(report, slug):
            failed.append(slug)
            continue

        print(f"  OK: {len(report)} chars, validated")

        example = {
            "input": post_content,
            "expected": report,
            "metadata": {
                "post_slug": slug,
                "word_count": word_count,
                "report_chars": len(report),
                "source": "existing" if slug in EXISTING_REPORTS else "generated",
            },
        }

        if slug in HOLDOUT_SLUGS:
            holdout_examples.append(example)
            print(f"  -> HOLDOUT")
        else:
            training_examples.append(example)
            print(f"  -> TRAINING")

        # Delay between Codex calls (not needed for existing reports)
        if slug not in EXISTING_REPORTS:
            time.sleep(3)

    # Write datasets
    train_path = DATASETS_DIR / "fact-checks.jsonl"
    holdout_path = DATASETS_DIR / "fact-checks-holdout.jsonl"

    with open(train_path, "w") as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + "\n")

    with open(holdout_path, "w") as f:
        for ex in holdout_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'='*50}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Training examples: {len(training_examples)} -> {train_path}")
    print(f"Holdout examples:  {len(holdout_examples)} -> {holdout_path}")
    print(f"Failed:            {len(failed)}")
    if failed:
        print(f"  Failed slugs: {failed}")


if __name__ == "__main__":
    main()
