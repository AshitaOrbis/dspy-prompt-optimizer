"""
Format instructions for structured output during optimization.

These templates guide Claude to produce concise, extractable outputs
instead of verbose multi-paragraph responses.

This addresses the "format collapse" issue where Claude's natural
verbosity makes metric extraction unreliable.
"""

from typing import Optional

# Format templates for different metric types
FORMAT_TEMPLATES = {
    "routing": """
## Output Format

Respond with EXACTLY this format (2 lines only):

Tool: [tool_name]
Reason: [one sentence explanation]

Example:
Tool: brave_news_search
Reason: Query asks about current events which requires fresh news content.
""",

    "binary_decision": """
## Output Format

Respond with EXACTLY this format (2 lines only):

Decision: [grep OR mgrep]
Reason: [one sentence explanation]

Example:
Decision: mgrep
Reason: Query uses natural language requiring semantic understanding.
""",

    "tier_classification": """
## Output Format

Respond with EXACTLY this format (2 lines only):

Tier: [core OR specialized OR deferred]
Reason: [one sentence explanation]

Example:
Tier: core
Reason: File reading is a basic operation handled by the Read tool.
""",

    "evaluation_score": """
## Output Format

Respond with EXACTLY this format:

Score: [number]/100
Decision: [INTEGRATE OR SKIP OR RESEARCH_MORE]
Key Points:
- [point 1]
- [point 2]
- [point 3]
""",

    "severity_classification": """
## Output Format

Respond with a code review in this structure:

## Code Review Summary

### Critical Issues
[List each critical issue with file:line reference and description]
Example:
1. **SQL Injection** (api/users.ts:45)
   - User input passed directly to query without sanitization

### High Issues
[List each high severity issue in same format]

### Warnings
[List warnings]

### Summary
- X critical issues requiring immediate fix
- Y high issues to address
- Z warnings to review
""",

    "test_coverage": """
## Output Format

Respond with a test suite in this structure:

## Test Suite Summary

Total Tests: [count]

### Categories Covered
- Happy path tests
- Edge case tests
- Error handling tests

### Test Structure
Using describe/it/expect pattern

### Test List
1. should [behavior] when [condition]
2. should [behavior] when [condition]
""",

    "publication_review": """
## Output Format

Organize your findings into three priority tiers:

## MUST FIX
1. **[Brief title]** — [Why it's wrong and what to fix]
2. ...

## SHOULD FIX
1. **[Brief title]** — [Why it matters and suggested correction]
2. ...

## NICE TO HAVE
1. **[Brief title]** — [Suggested improvement]
2. ...

Rules:
- MUST FIX: Factual errors, unsupported claims, internal contradictions
- SHOULD FIX: Overclaims, hedging issues, consistency gaps, tone problems
- NICE TO HAVE: Sentence-level polish, minor redundancy, stylistic preferences
- Cite exact text when flagging issues
- If no issues in a tier, omit that section
""",

    "default": """
## Output Format

Be concise. Respond with a structured answer using clear labels.
Avoid lengthy explanations or decision trees.
""",
}

# Mapping from metric function names to format types
METRIC_FORMAT_MAP = {
    # Skill metrics
    "routing_accuracy": "routing",
    "binary_decision_match": "binary_decision",
    "tool_tier_classification": "tier_classification",
    # Agent metrics
    "evaluation_score_metric": "evaluation_score",
    "issue_severity_match": "severity_classification",
    "security_cwe_match": "severity_classification",
    "test_coverage_score": "test_coverage",
    "complexity_classification": "tier_classification",
    # Publication review
    "publication_review_match": "publication_review",
}


def get_format_instruction(metric_type: str) -> str:
    """
    Get format instruction template for a given metric type.

    Args:
        metric_type: Either a format type name (e.g., "routing") or
                    a metric function name (e.g., "routing_accuracy")

    Returns:
        Format instruction string to append to prompts
    """
    # Check if it's a direct format type
    if metric_type in FORMAT_TEMPLATES:
        return FORMAT_TEMPLATES[metric_type]

    # Check if it's a metric function name
    if metric_type in METRIC_FORMAT_MAP:
        format_type = METRIC_FORMAT_MAP[metric_type]
        return FORMAT_TEMPLATES[format_type]

    # Fall back to default
    return FORMAT_TEMPLATES["default"]


def get_format_type_for_target(target_name: str) -> str:
    """
    Get the format type for a skill/agent target.

    Args:
        target_name: Name of the skill or agent

    Returns:
        Format type string
    """
    target_format_map = {
        # Skills
        "mcp-search-framework": "routing",
        "mgrep-guide": "binary_decision",
        "advanced-tool-use": "tier_classification",
        "dispatching-parallel-agents": "binary_decision",
        # Agents
        "code-reviewer": "severity_classification",
        "security-auditor": "severity_classification",
        "test-writer": "test_coverage",
        "performance-analyzer": "tier_classification",
        "capability-evaluator": "evaluation_score",
        # Publication review targets
        "publication-review-gpt": "publication_review",
        "publication-review-gemini": "publication_review",
        "publication-review-opus": "publication_review",
    }

    return target_format_map.get(target_name, "default")


def append_format_instruction(prompt: str, metric_type: str) -> str:
    """
    Append format instruction to a prompt.

    Args:
        prompt: Original prompt text
        metric_type: Metric type or target name

    Returns:
        Prompt with format instruction appended
    """
    instruction = get_format_instruction(metric_type)

    # Check if prompt already has format section
    if "## Output Format" in prompt:
        return prompt

    return f"{prompt}\n\n{instruction}"


def get_all_format_types() -> list:
    """Return list of all available format types."""
    return list(FORMAT_TEMPLATES.keys())


def get_format_preview(metric_type: str) -> str:
    """
    Get a preview of what the expected output format looks like.

    Args:
        metric_type: Format type or metric name

    Returns:
        Preview string showing expected format
    """
    instruction = get_format_instruction(metric_type)

    # Extract example from instruction
    if "Example:" in instruction:
        example_start = instruction.index("Example:")
        return instruction[example_start:]

    return "No example available"
