#!/usr/bin/env python3
"""
Training data generation helper for prompt optimization.

Provides utilities for creating, validating, and managing JSONL training datasets.

Usage:
    # Create a new dataset with template examples
    python create_training_data.py --create --type agent --name code-reviewer

    # Validate an existing dataset
    python create_training_data.py --validate datasets/code-reviews.jsonl

    # Extract examples from git history (for code-reviewer)
    python create_training_data.py --from-git --repo /path/to/repo --output datasets/git-reviews.jsonl

    # Convert markdown examples to JSONL
    python create_training_data.py --from-markdown examples.md --output datasets/examples.jsonl

Examples:
    # Create template for test-writer agent
    python create_training_data.py --create --type agent --name test-writer --count 10

    # Validate and show statistics
    python create_training_data.py --validate datasets/code-reviews.jsonl --stats
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Template structures for different types
AGENT_TEMPLATES = {
    "code-reviewer": {
        "input_template": "```{language}\n{code}\n```\n\nPlease review this code for issues.",
        "expected_template": "## Code Review\n\n### Issues Found\n\n**{severity}**: {issue_description}\n\n### Recommendation\n\n{recommendation}",
        "metadata_fields": ["severity", "category", "issue_type"],
        "examples": [
            {
                "input": "```python\ndef get_user(id):\n    query = f\"SELECT * FROM users WHERE id = {id}\"\n    return db.execute(query)\n```\n\nPlease review this code for issues.",
                "expected": "## Code Review\n\n### Issues Found\n\n**Critical**: SQL injection vulnerability. User input is directly interpolated into SQL query.\n\n### Recommendation\n\nUse parameterized queries: `db.execute(\"SELECT * FROM users WHERE id = ?\", [id])`",
                "metadata": {"severity": "critical", "category": "security", "issue_type": "sql_injection"}
            }
        ]
    },
    "test-writer": {
        "input_template": "Write tests for:\n```{language}\n{code}\n```",
        "expected_template": "```{language}\ndescribe('{function_name}', () => {{\n  it('should {test_case}', () => {{\n    {test_body}\n  }});\n}});\n```",
        "metadata_fields": ["test_type", "coverage", "framework"],
        "examples": [
            {
                "input": "Write tests for:\n```typescript\nfunction add(a: number, b: number): number {\n  return a + b;\n}\n```",
                "expected": "```typescript\ndescribe('add', () => {\n  it('should add two positive numbers', () => {\n    expect(add(2, 3)).toBe(5);\n  });\n  it('should handle negative numbers', () => {\n    expect(add(-1, 1)).toBe(0);\n  });\n  it('should handle zero', () => {\n    expect(add(0, 5)).toBe(5);\n  });\n});\n```",
                "metadata": {"test_type": "unit", "coverage": "basic", "framework": "jest"}
            }
        ]
    },
    "security-auditor": {
        "input_template": "Audit this code for security vulnerabilities:\n```{language}\n{code}\n```",
        "expected_template": "## Security Audit\n\n### Vulnerability: {cwe_id} - {vulnerability_name}\n\n**Severity**: {severity}\n**Location**: {location}\n\n### Description\n{description}\n\n### Remediation\n{remediation}",
        "metadata_fields": ["severity", "cwe_id", "vulnerability_type"],
        "examples": [
            {
                "input": "Audit this code for security vulnerabilities:\n```javascript\napp.get('/user', (req, res) => {\n  const html = `<h1>Hello ${req.query.name}</h1>`;\n  res.send(html);\n});\n```",
                "expected": "## Security Audit\n\n### Vulnerability: CWE-79 - Cross-Site Scripting (XSS)\n\n**Severity**: High\n**Location**: Line 2-3, template literal\n\n### Description\nUser-supplied input from query parameter is directly embedded in HTML without sanitization.\n\n### Remediation\nUse a templating engine with auto-escaping or sanitize input: `escapeHtml(req.query.name)`",
                "metadata": {"severity": "high", "cwe_id": "CWE-79", "vulnerability_type": "xss"}
            }
        ]
    },
    "performance-analyzer": {
        "input_template": "Analyze performance of:\n```{language}\n{code}\n```",
        "expected_template": "## Performance Analysis\n\n### Pattern Identified: {pattern}\n\n**Current Complexity**: {complexity}\n**Impact**: {impact}\n\n### Recommended Improvement\n{improvement}\n\n**Expected Improvement**: {expected_gain}",
        "metadata_fields": ["pattern", "complexity", "impact_level"],
        "examples": [
            {
                "input": "Analyze performance of:\n```python\ndef find_duplicates(items):\n    duplicates = []\n    for i in range(len(items)):\n        for j in range(i+1, len(items)):\n            if items[i] == items[j] and items[i] not in duplicates:\n                duplicates.append(items[i])\n    return duplicates\n```",
                "expected": "## Performance Analysis\n\n### Pattern Identified: Nested Loop with Linear Search\n\n**Current Complexity**: O(n²) comparisons + O(n) membership check per match\n**Impact**: High - scales poorly with input size\n\n### Recommended Improvement\nUse a set for O(1) lookups:\n```python\ndef find_duplicates(items):\n    seen = set()\n    duplicates = set()\n    for item in items:\n        if item in seen:\n            duplicates.add(item)\n        seen.add(item)\n    return list(duplicates)\n```\n\n**Expected Improvement**: O(n) total time complexity",
                "metadata": {"pattern": "nested_loop", "complexity": "O(n²)", "impact_level": "high"}
            }
        ]
    },
}

SKILL_TEMPLATES = {
    "mcp-search-framework": {
        "input_template": "Query: {query}",
        "expected_template": "Tool: {tool}\nReason: {reason}",
        "metadata_fields": ["tool", "category"],
        "examples": [
            {
                "input": "Query: restaurants near downtown Seattle",
                "expected": "Tool: brave_local_search\nReason: Location-based query for local businesses",
                "metadata": {"tool": "brave_local_search", "category": "local"}
            }
        ]
    },
    "mgrep-guide": {
        "input_template": "Search for: {query}\nContext: {context}",
        "expected_template": "Tool: {tool}\nReason: {reason}",
        "metadata_fields": ["tool", "search_type"],
        "examples": [
            {
                "input": "Search for: handleSubmit\nContext: Finding function definition",
                "expected": "Tool: Grep\nReason: Exact function name - use Grep for literal string matching",
                "metadata": {"tool": "Grep", "search_type": "exact"}
            }
        ]
    },
    "advanced-tool-use": {
        "input_template": "Task: {task}\nContext: {context}",
        "expected_template": "Tool Tier: {tier}\nTool: {tool}\nReason: {reason}",
        "metadata_fields": ["tier", "tool"],
        "examples": [
            {
                "input": "Task: Read the contents of config.json\nContext: Need to understand configuration",
                "expected": "Tool Tier: Core\nTool: Read\nReason: Single file read operation - use Read tool directly",
                "metadata": {"tier": "core", "tool": "read"}
            }
        ]
    },
}


def create_template_dataset(
    target_type: str,
    name: str,
    count: int = 5,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Create a template dataset with placeholder examples.

    Args:
        target_type: "agent" or "skill"
        name: Target name (e.g., "code-reviewer")
        count: Number of template examples to generate
        output_path: Optional path to save JSONL

    Returns:
        List of template examples
    """
    templates = AGENT_TEMPLATES if target_type == "agent" else SKILL_TEMPLATES

    if name not in templates:
        # Create generic template
        template = {
            "input_template": "Input: {input}",
            "expected_template": "Output: {output}",
            "metadata_fields": ["category"],
            "examples": [
                {
                    "input": f"Example input for {name}",
                    "expected": f"Example expected output for {name}",
                    "metadata": {"category": "example"}
                }
            ]
        }
    else:
        template = templates[name]

    # Generate examples
    examples = []

    # Add the provided example(s)
    examples.extend(template["examples"])

    # Generate additional placeholders
    for i in range(count - len(examples)):
        examples.append({
            "input": f"TODO: Add input example {i+1}",
            "expected": f"TODO: Add expected output {i+1}",
            "metadata": {field: "TODO" for field in template["metadata_fields"]}
        })

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"Created template dataset: {output_path}")
        print(f"  Examples: {len(examples)}")
        print(f"  TODO: Replace placeholder examples with real data")

    return examples


def validate_dataset(
    path: Path,
    show_stats: bool = False,
) -> Dict[str, Any]:
    """
    Validate a JSONL training dataset.

    Args:
        path: Path to JSONL file
        show_stats: Print statistics

    Returns:
        Validation result with any errors
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "total_examples": 0,
            "with_metadata": 0,
            "avg_input_length": 0,
            "avg_output_length": 0,
            "metadata_fields": set(),
        }
    }

    input_lengths = []
    output_lengths = []

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                result["valid"] = False
                result["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # Check required fields
            if "input" not in data:
                result["valid"] = False
                result["errors"].append(f"Line {line_num}: Missing 'input' field")

            if "expected" not in data:
                result["valid"] = False
                result["errors"].append(f"Line {line_num}: Missing 'expected' field")

            # Check for TODO placeholders
            if "TODO" in str(data.get("input", "")):
                result["warnings"].append(f"Line {line_num}: Contains TODO placeholder in input")

            if "TODO" in str(data.get("expected", "")):
                result["warnings"].append(f"Line {line_num}: Contains TODO placeholder in expected")

            # Collect stats
            result["stats"]["total_examples"] += 1

            if "metadata" in data:
                result["stats"]["with_metadata"] += 1
                if isinstance(data["metadata"], dict):
                    result["stats"]["metadata_fields"].update(data["metadata"].keys())

            if "input" in data:
                input_lengths.append(len(data["input"]))
            if "expected" in data:
                output_lengths.append(len(data["expected"]))

    # Calculate averages
    if input_lengths:
        result["stats"]["avg_input_length"] = sum(input_lengths) / len(input_lengths)
    if output_lengths:
        result["stats"]["avg_output_length"] = sum(output_lengths) / len(output_lengths)

    # Convert set to list for JSON serialization
    result["stats"]["metadata_fields"] = list(result["stats"]["metadata_fields"])

    if show_stats:
        print(f"Dataset: {path}")
        print(f"  Valid: {result['valid']}")
        print(f"  Total examples: {result['stats']['total_examples']}")
        print(f"  With metadata: {result['stats']['with_metadata']}")
        print(f"  Avg input length: {result['stats']['avg_input_length']:.0f} chars")
        print(f"  Avg output length: {result['stats']['avg_output_length']:.0f} chars")
        print(f"  Metadata fields: {result['stats']['metadata_fields']}")

        if result["errors"]:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result["errors"][:5]:
                print(f"  - {error}")
            if len(result["errors"]) > 5:
                print(f"  ... and {len(result['errors']) - 5} more")

        if result["warnings"]:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result["warnings"][:5]:
                print(f"  - {warning}")
            if len(result["warnings"]) > 5:
                print(f"  ... and {len(result['warnings']) - 5} more")

    return result


def extract_from_git(
    repo_path: Path,
    output_path: Path,
    max_commits: int = 50,
) -> List[Dict[str, Any]]:
    """
    Extract code review examples from git history.

    Analyzes commit diffs to create training data for code-reviewer.

    Args:
        repo_path: Path to git repository
        output_path: Path to save JSONL output
        max_commits: Maximum commits to analyze

    Returns:
        List of extracted examples
    """
    import subprocess

    examples = []

    # Get recent commits with changes
    result = subprocess.run(
        ["git", "log", f"-{max_commits}", "--oneline", "--diff-filter=M"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: Failed to get git log: {result.stderr}")
        return examples

    commits = result.stdout.strip().split("\n")

    for commit_line in commits[:max_commits]:
        if not commit_line:
            continue

        commit_hash = commit_line.split()[0]

        # Get diff for this commit
        diff_result = subprocess.run(
            ["git", "show", commit_hash, "--stat", "--format=%B"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        if diff_result.returncode != 0:
            continue

        commit_msg = diff_result.stdout.strip()

        # Simple heuristic: if commit message mentions fix/bug/security
        categories = []
        if "fix" in commit_msg.lower():
            categories.append("bugfix")
        if "security" in commit_msg.lower():
            categories.append("security")
        if "refactor" in commit_msg.lower():
            categories.append("refactoring")

        if categories:
            # This is a potential review example
            examples.append({
                "input": f"Review the following commit:\n\n{commit_msg[:500]}",
                "expected": f"This commit addresses: {', '.join(categories)}",
                "metadata": {
                    "commit": commit_hash,
                    "category": categories[0] if categories else "general",
                    "source": "git_history"
                }
            })

    if output_path and examples:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"Extracted {len(examples)} examples from git history")
        print(f"Note: These are raw extractions - manual review recommended")

    return examples


def convert_from_markdown(
    markdown_path: Path,
    output_path: Path,
) -> List[Dict[str, Any]]:
    """
    Convert markdown examples to JSONL format.

    Expects markdown with ## Example sections containing:
    - **Input**: ...
    - **Expected**: ...
    - **Category**: ... (optional)

    Args:
        markdown_path: Path to markdown file
        output_path: Path to save JSONL output

    Returns:
        List of converted examples
    """
    import re

    examples = []

    with open(markdown_path) as f:
        content = f.read()

    # Split by example sections
    example_sections = re.split(r"##\s+Example\s*\d*", content)[1:]

    for section in example_sections:
        example = {"metadata": {}}

        # Extract input
        input_match = re.search(r"\*\*Input\*\*:\s*(.+?)(?=\*\*|$)", section, re.DOTALL)
        if input_match:
            example["input"] = input_match.group(1).strip()

        # Extract expected
        expected_match = re.search(r"\*\*Expected\*\*:\s*(.+?)(?=\*\*|$)", section, re.DOTALL)
        if expected_match:
            example["expected"] = expected_match.group(1).strip()

        # Extract optional metadata
        category_match = re.search(r"\*\*Category\*\*:\s*(.+?)(?=\*\*|$)", section)
        if category_match:
            example["metadata"]["category"] = category_match.group(1).strip()

        if "input" in example and "expected" in example:
            examples.append(example)

    if output_path and examples:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"Converted {len(examples)} examples from markdown")

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Training data generation helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--create",
        action="store_true",
        help="Create a new template dataset",
    )
    mode_group.add_argument(
        "--validate",
        type=str,
        metavar="PATH",
        help="Validate an existing JSONL dataset",
    )
    mode_group.add_argument(
        "--from-git",
        action="store_true",
        help="Extract examples from git history",
    )
    mode_group.add_argument(
        "--from-markdown",
        type=str,
        metavar="PATH",
        help="Convert markdown examples to JSONL",
    )
    mode_group.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates",
    )

    # Create mode options
    parser.add_argument(
        "--type",
        choices=["agent", "skill"],
        help="Target type for --create",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Target name for --create",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of template examples (default: 5)",
    )

    # Git mode options
    parser.add_argument(
        "--repo",
        type=str,
        help="Repository path for --from-git",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=50,
        help="Maximum commits to analyze (default: 50)",
    )

    # Common options
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path for generated JSONL",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics (with --validate)",
    )

    args = parser.parse_args()

    # List templates mode
    if args.list_templates:
        print("Agent Templates:")
        for name, template in AGENT_TEMPLATES.items():
            print(f"  {name}")
            print(f"    Metadata fields: {template['metadata_fields']}")
            print(f"    Examples provided: {len(template['examples'])}")

        print("\nSkill Templates:")
        for name, template in SKILL_TEMPLATES.items():
            print(f"  {name}")
            print(f"    Metadata fields: {template['metadata_fields']}")
            print(f"    Examples provided: {len(template['examples'])}")
        return

    # Create mode
    if args.create:
        if not args.type or not args.name:
            parser.error("--create requires --type and --name")

        output_path = Path(args.output) if args.output else Path(f"datasets/{args.name}.jsonl")
        create_template_dataset(
            target_type=args.type,
            name=args.name,
            count=args.count,
            output_path=output_path,
        )
        return

    # Validate mode
    if args.validate:
        result = validate_dataset(Path(args.validate), show_stats=True)
        if not result["valid"]:
            sys.exit(1)
        return

    # Git extraction mode
    if args.from_git:
        if not args.repo:
            parser.error("--from-git requires --repo")
        if not args.output:
            parser.error("--from-git requires --output")

        extract_from_git(
            repo_path=Path(args.repo),
            output_path=Path(args.output),
            max_commits=args.max_commits,
        )
        return

    # Markdown conversion mode
    if args.from_markdown:
        if not args.output:
            parser.error("--from-markdown requires --output")

        convert_from_markdown(
            markdown_path=Path(args.from_markdown),
            output_path=Path(args.output),
        )
        return


if __name__ == "__main__":
    main()
