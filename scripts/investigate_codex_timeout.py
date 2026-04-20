#!/usr/bin/env python3
"""
Investigate Codex timeout behavior on large blog posts.

Tests combinations of reasoning_effort (high vs xhigh) and model variant
(gpt-5.4 vs gpt-5.3-instant) on a representative large input to identify
a working configuration.

Usage:
    python3 scripts/investigate_codex_timeout.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

POSTS_DIR = Path.home() / "claudeworkspace/applications/ashitaorbis/shared/content/posts"

# Test posts by size: medium, large, very large
TEST_POSTS = [
    ("032-the-etymology-tax.md", "medium"),      # ~1.9K words
    ("025-the-logistics-gap.md", "large"),        # ~3K words
    ("037-the-model-generation-audit.md", "xlarge"),  # ~4K words
]

# Factcheck-style prompt (shortened)
PROMPT_TEMPLATE = """Read this blog post and extract all verifiable factual claims.
For each claim, provide a 1-line verdict (VERIFIED/INACCURATE/UNVERIFIABLE) with brief evidence.

Output format:
| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|

---

{post_content}
"""

# Configurations to test
CONFIGS = [
    {"name": "gpt-5.4_xhigh", "model": "gpt-5.4", "reasoning": "xhigh"},
    {"name": "gpt-5.4_high", "model": "gpt-5.4", "reasoning": "high"},
    {"name": "gpt-5.4_medium", "model": "gpt-5.4", "reasoning": "medium"},
]


def find_codex_binary():
    home = os.path.expanduser("~")
    nvm_dir = os.path.join(home, ".nvm/versions/node")
    if os.path.isdir(nvm_dir):
        for d in sorted(os.listdir(nvm_dir), reverse=True):
            candidate = os.path.join(nvm_dir, d, "bin/codex")
            if os.path.isfile(candidate):
                return candidate
    return "codex"


def get_mcp_disable_args():
    config_path = os.path.expanduser("~/.codex/config.toml")
    args = []
    if os.path.isfile(config_path):
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("[mcp_servers.") and line.endswith("]"):
                    name = line[len("[mcp_servers."):-1]
                    if name and "." not in name:
                        args.extend(["-c", f"mcp_servers.{name}.enabled=false"])
    return args


def run_codex(prompt: str, model: str, reasoning: str, timeout: int = 480):
    codex_bin = find_codex_binary()
    mcp_disable = get_mcp_disable_args()
    cmd = [
        codex_bin, "exec", "--full-auto",
        "-c", f'model="{model}"',
        "-c", f'model_reasoning_effort="{reasoning}"',
        *mcp_disable,
        "--ephemeral", "-",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return {"status": "error", "elapsed": elapsed, "error": result.stderr[:200]}
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        output = ansi_escape.sub('', result.stdout).strip()
        return {"status": "success", "elapsed": elapsed, "output_len": len(output)}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed": timeout, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"status": "exception", "elapsed": time.time() - t0, "error": str(e)}


def main():
    results = []

    for post_file, size_label in TEST_POSTS:
        post_path = POSTS_DIR / post_file
        if not post_path.exists():
            print(f"SKIP: {post_file} not found")
            continue
        post_content = post_path.read_text()
        word_count = len(post_content.split())
        prompt = PROMPT_TEMPLATE.format(post_content=post_content)
        prompt_chars = len(prompt)

        print(f"\n{'='*60}")
        print(f"Post: {post_file} ({size_label}, {word_count} words, {prompt_chars} chars)")
        print(f"{'='*60}")

        for cfg in CONFIGS:
            print(f"\n  [{cfg['name']}] running...")
            r = run_codex(prompt, cfg["model"], cfg["reasoning"], timeout=480)
            print(f"  [{cfg['name']}] {r['status']} in {r['elapsed']:.1f}s", end="")
            if r["status"] == "success":
                print(f" (output {r['output_len']} chars)")
            else:
                print(f" — {r.get('error', '')[:100]}")
            results.append({
                "post": post_file,
                "size": size_label,
                "words": word_count,
                "config": cfg["name"],
                **r,
            })
            time.sleep(5)  # gap between runs

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Post':<40} {'Config':<20} {'Status':<12} {'Time':>8}")
    for r in results:
        print(f"{r['post']:<40} {r['config']:<20} {r['status']:<12} {r['elapsed']:>7.1f}s")

    # Save results
    out_path = Path(__file__).parent.parent / "reports" / "codex-timeout-investigation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
