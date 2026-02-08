"""
Demo transformers for post-processing captured optimization demos.

When Claude produces verbose outputs during optimization, these transformers
convert them into structured, concise demos that are more effective for
few-shot learning.

This addresses the "format collapse" issue where verbose demos (500+ words)
pollute the few-shot context and degrade performance.
"""

from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

from .extractors import (
    extract_tool_from_verbose,
    extract_binary_decision,
    extract_tier,
    normalize_tool_name,
    # Agent extractors
    extract_severity_details,
    extract_test_coverage_details,
    extract_evaluation_details,
)


@dataclass
class TransformedDemo:
    """A transformed demo with structured output."""
    input_text: str
    output_text: str
    original_output: str  # Keep original for debugging
    transformation_applied: str


def transform_routing_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform a verbose routing demo into structured format.

    Extracts the tool selection and creates a concise 2-line output.

    Args:
        input_text: Original input query
        output_text: Verbose output from Claude
        metadata: Optional metadata dict

    Returns:
        TransformedDemo with structured output
    """
    tool = extract_tool_from_verbose(output_text)

    if tool:
        # Try to extract a reason from the verbose output
        reason = _extract_reason(output_text, tool)
        structured = f"Tool: {tool}\nReason: {reason}"
    else:
        # Couldn't extract tool, keep original but truncate
        structured = output_text[:200] + "..." if len(output_text) > 200 else output_text

    return TransformedDemo(
        input_text=input_text,
        output_text=structured,
        original_output=output_text,
        transformation_applied="routing",
    )


def transform_binary_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform a verbose binary decision demo into structured format.

    Extracts the grep/mgrep decision and creates a concise 2-line output.

    Args:
        input_text: Original input query
        output_text: Verbose output from Claude
        metadata: Optional metadata dict

    Returns:
        TransformedDemo with structured output
    """
    decision = extract_binary_decision(output_text)

    if decision:
        reason = _extract_reason(output_text, decision)
        structured = f"Decision: {decision}\nReason: {reason}"
    else:
        # Couldn't extract decision, keep original but truncate
        structured = output_text[:200] + "..." if len(output_text) > 200 else output_text

    return TransformedDemo(
        input_text=input_text,
        output_text=structured,
        original_output=output_text,
        transformation_applied="binary_decision",
    )


def transform_tier_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform a verbose tier classification demo into structured format.

    Extracts the tier and creates a concise 2-line output.

    Args:
        input_text: Original input query
        output_text: Verbose output from Claude
        metadata: Optional metadata dict

    Returns:
        TransformedDemo with structured output
    """
    tier = extract_tier(output_text)

    if tier:
        reason = _extract_reason(output_text, tier)
        structured = f"Tier: {tier}\nReason: {reason}"
    else:
        # Couldn't extract tier, keep original but truncate
        structured = output_text[:200] + "..." if len(output_text) > 200 else output_text

    return TransformedDemo(
        input_text=input_text,
        output_text=structured,
        original_output=output_text,
        transformation_applied="tier_classification",
    )


def transform_passthrough(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Passthrough transformer that keeps output unchanged.

    Used when no transformation is appropriate.
    """
    return TransformedDemo(
        input_text=input_text,
        output_text=output_text,
        original_output=output_text,
        transformation_applied="passthrough",
    )


# =============================================================================
# Agent-Specific Transformers
# =============================================================================

def transform_severity_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform verbose code review demo into structured severity format.

    Extracts severity counts and top 3 issues, condensing to ~500 chars.
    Preserves: severity counts, issue titles, file locations.
    Drops: full code examples, detailed explanations.

    Args:
        input_text: Original input (code to review)
        output_text: Verbose review output
        metadata: Optional metadata dict

    Returns:
        TransformedDemo with condensed output (~500 chars)
    """
    details = extract_severity_details(output_text)

    # Build structured output
    lines = ["## Code Review Summary", ""]

    # Severity counts
    lines.append(f"Critical: {details.critical_count}")
    lines.append(f"High: {details.high_count}")
    lines.append(f"Medium: {details.medium_count}")
    lines.append(f"Low: {details.low_count}")

    # Top 3 issues with brief descriptions
    if details.issues:
        lines.append("")
        lines.append("### Key Issues")
        for i, issue in enumerate(details.issues[:3]):
            severity = issue.get('severity', 'unknown').capitalize()
            title = issue.get('title', '')[:80]
            file_ref = ""
            if issue.get('file'):
                file_ref = f" ({issue['file']}"
                if issue.get('line'):
                    file_ref += f":{issue['line']}"
                file_ref += ")"
            lines.append(f"{i+1}. **{severity}**{file_ref}: {title}")

    structured = "\n".join(lines)

    # Ensure we stay under ~500 chars
    if len(structured) > 500:
        structured = structured[:497] + "..."

    return TransformedDemo(
        input_text=input_text,
        output_text=structured,
        original_output=output_text,
        transformation_applied="severity_classification",
    )


def transform_test_coverage_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform verbose test suite demo into structured coverage format.

    Extracts test count and category breakdown, condensing to ~400 chars.
    Preserves: test names, categories, structure.
    Drops: full test code, detailed assertions.

    Args:
        input_text: Original input (function to test)
        output_text: Verbose test output
        metadata: Optional metadata dict

    Returns:
        TransformedDemo with condensed output (~400 chars)
    """
    details = extract_test_coverage_details(output_text)

    # Build structured output
    lines = ["## Test Suite Summary", ""]

    # Counts
    lines.append(f"Total Tests: {details.total_tests}")

    # Categories
    categories = []
    if details.happy_path_count > 0:
        categories.append("happy_path")
    if details.edge_case_count > 0:
        categories.append("edge_cases")
    if details.error_case_count > 0:
        categories.append("error_cases")
    if categories:
        lines.append(f"Categories: {', '.join(categories)}")

    # Structure
    structure_parts = []
    if details.has_describe:
        structure_parts.append("describe")
    if details.has_it:
        structure_parts.append("it/test")
    if details.has_expect:
        structure_parts.append("expect")
    if structure_parts:
        lines.append(f"Structure: {'/'.join(structure_parts)}")

    # Top 5 test names
    if details.tests:
        lines.append("")
        lines.append("### Tests")
        for test in details.tests[:5]:
            name = test.get('name', '')[:60]
            lines.append(f"- {name}")

    structured = "\n".join(lines)

    # Ensure we stay under ~400 chars
    if len(structured) > 400:
        structured = structured[:397] + "..."

    return TransformedDemo(
        input_text=input_text,
        output_text=structured,
        original_output=output_text,
        transformation_applied="test_coverage",
    )


def transform_evaluation_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform verbose evaluation demo into structured score format.

    Extracts score and decision, condensing to ~300 chars.
    Preserves: score, decision, top 3 criteria.
    Drops: full analysis, alternatives considered.

    Args:
        input_text: Original input (item to evaluate)
        output_text: Verbose evaluation output
        metadata: Optional metadata dict

    Returns:
        TransformedDemo with condensed output (~300 chars)
    """
    details = extract_evaluation_details(output_text)

    # Build structured output
    lines = ["## Evaluation Summary", ""]

    # Score
    if details.score is not None:
        lines.append(f"Score: {details.score:.0f}/100")
    else:
        lines.append("Score: [not extracted]")

    # Decision
    if details.decision:
        lines.append(f"Decision: {details.decision}")
    else:
        lines.append("Decision: [not extracted]")

    # Top 3 criteria
    if details.criteria:
        lines.append("")
        lines.append("### Key Criteria")
        for criterion in details.criteria[:3]:
            name = criterion.get('name', '')[:30]
            score = criterion.get('score', 0)
            lines.append(f"- {name}: {score:.0f}")

    structured = "\n".join(lines)

    # Ensure we stay under ~300 chars
    if len(structured) > 300:
        structured = structured[:297] + "..."

    return TransformedDemo(
        input_text=input_text,
        output_text=structured,
        original_output=output_text,
        transformation_applied="evaluation",
    )


def _extract_reason(text: str, selection: str) -> str:
    """
    Extract a brief reason from verbose output.

    Looks for explanation patterns near the selection.

    Args:
        text: Verbose output text
        selection: The tool/decision that was selected

    Returns:
        Brief reason string (max 100 chars)
    """
    import re

    text_lower = text.lower()
    selection_lower = selection.lower()

    # Look for reason patterns
    reason_patterns = [
        rf'{selection_lower}[^.]*because\s+([^.]+)',
        rf'{selection_lower}[^.]*since\s+([^.]+)',
        rf'{selection_lower}[^.]*as\s+([^.]+)',
        r'reason[:\s]+([^.\n]+)',
        r'because\s+([^.]+)',
        r'since\s+([^.]+)',
    ]

    for pattern in reason_patterns:
        match = re.search(pattern, text_lower)
        if match:
            reason = match.group(1).strip()
            # Capitalize and truncate
            reason = reason[0].upper() + reason[1:] if reason else ""
            if len(reason) > 100:
                reason = reason[:97] + "..."
            return reason

    # Fallback: use first sentence that mentions the selection
    sentences = text.split(".")
    for sentence in sentences:
        if selection_lower in sentence.lower():
            clean = sentence.strip()
            if clean and len(clean) > 10:
                if len(clean) > 100:
                    clean = clean[:97] + "..."
                return clean[0].upper() + clean[1:] if clean else ""

    # Ultimate fallback
    return f"Best match for the query type."


# Mapping from metric types to transformer functions
TRANSFORMER_MAP: Dict[str, Callable] = {
    # Skill transformers
    "routing": transform_routing_demo,
    "routing_accuracy": transform_routing_demo,
    "binary_decision": transform_binary_demo,
    "binary_decision_match": transform_binary_demo,
    "tier_classification": transform_tier_demo,
    "tool_tier_classification": transform_tier_demo,
    # Agent transformers
    "severity_classification": transform_severity_demo,
    "issue_severity_match": transform_severity_demo,
    "security_cwe_match": transform_severity_demo,
    "test_coverage": transform_test_coverage_demo,
    "test_coverage_score": transform_test_coverage_demo,
    "evaluation": transform_evaluation_demo,
    "evaluation_score": transform_evaluation_demo,
    "default": transform_passthrough,
}

# Mapping from target names to transformer functions
TARGET_TRANSFORMER_MAP: Dict[str, Callable] = {
    # Skills
    "mcp-search-framework": transform_routing_demo,
    "mgrep-guide": transform_binary_demo,
    "advanced-tool-use": transform_tier_demo,
    "dispatching-parallel-agents": transform_binary_demo,
    # Agents
    "code-reviewer": transform_severity_demo,
    "security-auditor": transform_severity_demo,
    "test-writer": transform_test_coverage_demo,
    "capability-evaluator": transform_evaluation_demo,
}


def get_demo_transformer(metric_type: str) -> Callable:
    """
    Get the appropriate transformer for a metric type.

    Args:
        metric_type: Either a format type name (e.g., "routing"),
                    a metric function name (e.g., "routing_accuracy"),
                    or a target name (e.g., "mcp-search-framework")

    Returns:
        Transformer function
    """
    # Check target names first
    if metric_type in TARGET_TRANSFORMER_MAP:
        return TARGET_TRANSFORMER_MAP[metric_type]

    # Check metric/format types
    if metric_type in TRANSFORMER_MAP:
        return TRANSFORMER_MAP[metric_type]

    # Fall back to passthrough
    return transform_passthrough


def transform_demo_list(
    demos: list,
    metric_type: str,
) -> list:
    """
    Transform a list of demos using the appropriate transformer.

    Args:
        demos: List of (input_text, output_text, score) tuples or Demo objects
        metric_type: Metric type for transformer selection

    Returns:
        List of TransformedDemo objects
    """
    transformer = get_demo_transformer(metric_type)
    transformed = []

    for demo in demos:
        if hasattr(demo, "input_text"):
            # Demo object
            result = transformer(
                demo.input_text,
                demo.output_text,
                getattr(demo, "metadata", None),
            )
        elif isinstance(demo, tuple) and len(demo) >= 2:
            # Tuple format
            result = transformer(demo[0], demo[1], None)
        else:
            continue

        transformed.append(result)

    return transformed
