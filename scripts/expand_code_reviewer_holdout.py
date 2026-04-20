#!/usr/bin/env python3
"""
Expand code-reviewer holdout from 3 to 8 examples.

Generates 5 new holdout examples covering diverse categories (error_handling,
concurrency, memory, null_safety, best_practices). Uses Opus to produce both
the code snippet and the gold-standard review.

Usage:
    python3 scripts/expand_code_reviewer_holdout.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
HOLDOUT_PATH = DATASETS_DIR / "code-reviews-holdout.jsonl"

# New examples to generate — categories not yet in holdout
NEW_EXAMPLES = [
    {
        "category": "error_handling",
        "issue_type": "missing_error_handling",
        "severity": "high",
        "description": "A TypeScript async function that fetches user data from an API without any error handling (no try/catch, no response.ok check, no type validation). 10-15 lines.",
    },
    {
        "category": "concurrency",
        "issue_type": "race_condition",
        "severity": "high",
        "description": "A Python script with a race condition: two threads modify a shared counter without a lock. 15-20 lines with threading.",
    },
    {
        "category": "memory",
        "issue_type": "memory_leak",
        "severity": "medium",
        "description": "A JavaScript event listener registered in a React component useEffect without cleanup — classic memory leak. 15-20 lines.",
    },
    {
        "category": "null_safety",
        "issue_type": "null_dereference",
        "severity": "high",
        "description": "A Go function that dereferences a map value without checking if the key exists, leading to a nil pointer panic risk. 10-15 lines.",
    },
    {
        "category": "best_practices",
        "issue_type": "magic_numbers",
        "severity": "low",
        "description": "A Python function with multiple magic numbers (timeouts, retry counts, thresholds) hardcoded throughout. 15-20 lines.",
    },
]

PROMPT_TEMPLATE = """Generate a realistic code review holdout example for a code-reviewer agent.

Category: {category}
Issue type: {issue_type}
Severity: {severity}
Description: {description}

Output EXACTLY two sections separated by "===REVIEW===":

Section 1 — Code snippet that exhibits the issue. Use this exact format:

Review the following code change:

File: [appropriate file path and extension]
```[language]
[10-20 lines of realistic code with the issue]
```

Section 2 — Gold-standard code review (concise, ~800-1500 chars). Use this format:

## Code Review Summary

### Critical Issues / Warnings / Suggestions
1. **[Issue title]** (file:line)
   - Severity: [Critical/High/Medium/Low]
   - [Brief explanation]

   **Recommended fix**:
   ```[language]
   [fix code]
   ```

[Additional issues if any, but keep to 1-2 major findings and maybe 1-2 minor ones]

Remember: Section 1 goes first (the code), then "===REVIEW===", then Section 2 (the review).
Do NOT include any preamble or explanation outside these two sections.
"""


def generate_example(example):
    prompt = PROMPT_TEMPLATE.format(**example)
    env = {k: v for k, v in os.environ.items()}
    env.pop("CLAUDECODE", None)

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "opus", "--max-turns", "1"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if result.returncode != 0:
            print(f"  ERROR: claude returned {result.returncode}: {result.stderr[:200]}")
            return None
        output = result.stdout.strip()
        if "===REVIEW===" not in output:
            print(f"  ERROR: no ===REVIEW=== separator found")
            return None
        code_section, review_section = output.split("===REVIEW===", 1)
        code_section = code_section.strip()
        review_section = review_section.strip()

        if len(code_section) < 100 or len(review_section) < 300:
            print(f"  ERROR: sections too short (code={len(code_section)}, review={len(review_section)})")
            return None

        return {
            "input": code_section,
            "expected": review_section,
            "metadata": {
                "severity": example["severity"],
                "category": example["category"],
                "issue_type": example["issue_type"],
            },
        }
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 300s")
        return None
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return None


def main():
    # Load existing holdout
    existing = []
    with open(HOLDOUT_PATH) as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))
    print(f"Loaded {len(existing)} existing holdout examples")

    new_examples = []
    for i, ex_spec in enumerate(NEW_EXAMPLES):
        print(f"\n[{i+1}/{len(NEW_EXAMPLES)}] Generating {ex_spec['category']}/{ex_spec['issue_type']}")
        result = generate_example(ex_spec)
        if result:
            print(f"  OK: input={len(result['input'])} chars, expected={len(result['expected'])} chars")
            new_examples.append(result)
        else:
            print(f"  FAILED")
        time.sleep(3)

    all_examples = existing + new_examples
    # Write back
    with open(HOLDOUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'='*50}")
    print(f"Holdout expanded: {len(existing)} -> {len(all_examples)}")
    print(f"New examples added: {len(new_examples)}")
    print(f"Failed: {len(NEW_EXAMPLES) - len(new_examples)}")

    # Show category distribution
    from collections import Counter
    categories = Counter(ex["metadata"]["category"] for ex in all_examples if ex.get("metadata"))
    print(f"\nCategory distribution:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
