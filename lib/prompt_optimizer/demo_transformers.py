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
    # Publication review extractor
    extract_review_findings,
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
            # severity can be None (not just missing) — extractor sets it
            # to None when it cannot infer a severity
            severity_raw = issue.get('severity') or 'unknown'
            severity = severity_raw.capitalize() if isinstance(severity_raw, str) else 'Unknown'
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


def transform_review_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform verbose publication review demo into structured format.

    Input side: Truncate blog post to first 500 words + [...truncated...]
    Output side: Condense to structured finding list (~200 words)
    Keeps each demo under ~1000 tokens, allowing 3 demos within budget.

    Args:
        input_text: Original input (blog post text)
        output_text: Verbose review output
        metadata: Optional metadata dict

    Returns:
        TransformedDemo with condensed input and output (~1000 tokens total)
    """
    # Truncate input to ~500 words
    words = input_text.split()
    if len(words) > 500:
        truncated_input = " ".join(words[:500]) + "\n\n[...truncated...]"
    else:
        truncated_input = input_text

    # Extract and condense findings
    findings = extract_review_findings(output_text)

    if findings:
        # Build condensed output
        sections = {"MUST": [], "SHOULD": [], "NICE": []}
        for f in findings:
            sections[f.tier].append(f.description[:100])

        lines = []
        if sections["MUST"]:
            lines.append("## MUST FIX")
            for i, desc in enumerate(sections["MUST"][:3], 1):  # Max 3 per tier
                lines.append(f"{i}. {desc}")
        if sections["SHOULD"]:
            lines.append("## SHOULD FIX")
            for i, desc in enumerate(sections["SHOULD"][:3], 1):
                lines.append(f"{i}. {desc}")
        if sections["NICE"]:
            lines.append("## NICE TO HAVE")
            for i, desc in enumerate(sections["NICE"][:2], 1):  # Max 2 for NICE
                lines.append(f"{i}. {desc}")

        structured = "\n".join(lines)
    else:
        # Couldn't extract findings, truncate original
        structured = output_text[:500] + "..." if len(output_text) > 500 else output_text

    # Ensure output stays under ~200 words
    output_words = structured.split()
    if len(output_words) > 200:
        structured = " ".join(output_words[:200]) + "..."

    return TransformedDemo(
        input_text=truncated_input,
        output_text=structured,
        original_output=output_text,
        transformation_applied="publication_review",
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


def transform_writing_review_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform verbose writing-review demo into condensed per-perspective summary.

    Input: truncated to first 500 words.
    Output: one condensed block per perspective (Editorial Critic, Target Reader,
    Tone Analyst, Technical Accuracy, Fact-Check Summary, Overall Assessment),
    keeping the first 3 lines of each section. Capped at ~300 words.
    """
    import re

    words = input_text.split()
    truncated_input = (
        " ".join(words[:500]) + "\n\n[...truncated...]"
        if len(words) > 500
        else input_text
    )

    perspective_headers = [
        "Editorial Critic", "Target Reader", "Tone Analyst",
        "Technical Accuracy", "Fact-Check Summary", "Overall Assessment",
    ]
    pattern = re.compile(
        r"##\s+(" + "|".join(re.escape(h) for h in perspective_headers) + r")[^\n]*\n(.*?)(?=\n##\s+|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    sections = []
    for m in pattern.finditer(output_text):
        header = m.group(1).strip()
        body = m.group(2).strip()
        first_block_lines = []
        for line in body.splitlines():
            s = line.strip()
            if not s:
                if first_block_lines:
                    break
                continue
            first_block_lines.append(s)
            if len(first_block_lines) >= 3:
                break
        condensed = " ".join(first_block_lines)
        if condensed:
            sections.append(f"## {header}\n{condensed}")

    structured = "\n\n".join(sections) if sections else output_text[:500]

    out_words = structured.split()
    if len(out_words) > 300:
        structured = " ".join(out_words[:300]) + "..."

    return TransformedDemo(
        input_text=truncated_input,
        output_text=structured,
        original_output=output_text,
        transformation_applied="writing_review",
    )


def transform_factcheck_demo(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TransformedDemo:
    """
    Transform verbose fact-check report into compact verdict summary.

    Input: truncated to first 500 words.
    Output: top 5 claim verdicts pulled from the Claims Extracted table (or
    fallback to "### Claim N" + "**Verdict**" pairs), each trimmed to one
    line, plus a brief Summary block. Capped at ~300 words.
    """
    import re

    words = input_text.split()
    truncated_input = (
        " ".join(words[:500]) + "\n\n[...truncated...]"
        if len(words) > 500
        else input_text
    )

    verdicts = []

    # 1) Try markdown table rows: | # | claim | type | para | verdict | ... |
    table_rows = re.findall(
        r"\|\s*\d+\s*\|([^|\n]+)\|[^|\n]*\|[^|\n]*\|\s*([A-Z][A-Z '\-/]+)\s*\|",
        output_text,
    )
    for claim, verdict in table_rows[:5]:
        claim = claim.strip()[:100]
        verdicts.append(f"- [{verdict.strip()}] {claim}")

    # 2) Fall back to "### Claim N:" / "**Verdict:**" pairs
    if not verdicts:
        claim_blocks = re.findall(
            r"###\s*Claim\s*\d+[:\s]*([^\n]+).*?\*\*Verdict\*\*[:\s]*([^\n]+)",
            output_text, re.DOTALL,
        )
        for claim, verdict in claim_blocks[:5]:
            verdicts.append(f"- [{verdict.strip()[:40]}] {claim.strip()[:100]}")

    # 3) Summary counts block as last resort
    summary_match = re.search(
        r"(?:##\s*Summary|Total claims)[^#]*", output_text, re.DOTALL,
    )
    summary_snippet = ""
    if summary_match:
        lines = [l.strip() for l in summary_match.group(0).splitlines()
                 if l.strip() and len(l.strip()) < 120]
        summary_snippet = "\n".join(lines[:6])

    parts = ["## Fact-Check Summary"]
    if summary_snippet:
        parts.append(summary_snippet)
    if verdicts:
        parts.append("### Top Claims")
        parts.extend(verdicts)

    structured = "\n".join(parts) if (verdicts or summary_snippet) else output_text[:500]

    out_words = structured.split()
    if len(out_words) > 300:
        structured = " ".join(out_words[:300]) + "..."

    return TransformedDemo(
        input_text=truncated_input,
        output_text=structured,
        original_output=output_text,
        transformation_applied="fact_check",
    )


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
    # Publication review
    "publication_review": transform_review_demo,
    "publication_review_match": transform_review_demo,
    # Writing review (Tier 3 skill)
    "writing_review": transform_writing_review_demo,
    "writing_review_quality_match": transform_writing_review_demo,
    # Fact-checker (Tier 2 agent)
    "fact_check": transform_factcheck_demo,
    "fact_check_quality_match": transform_factcheck_demo,
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
    # Publication review targets
    "publication-review-gpt": transform_review_demo,
    "publication-review-gemini": transform_review_demo,
    "publication-review-opus": transform_review_demo,
    # Writing review skill
    "writing-review": transform_writing_review_demo,
    # Fact-checker agent
    "fact-checker": transform_factcheck_demo,
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
