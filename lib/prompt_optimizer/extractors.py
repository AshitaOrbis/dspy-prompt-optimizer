"""
Robust tool extraction from verbose Claude outputs.

This module addresses the "format collapse" issue where Claude produces
verbose multi-paragraph responses during optimization but metrics expect
concise tool names. Uses 5 fallback strategies to maximize extraction success.

Reference: arXiv:2601.06151 on format collapse in LLM optimization
"""

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Canonical tool names for normalization
CANONICAL_TOOLS = {
    # Brave tools
    "brave_web_search": ["brave_web_search", "brave web search", "bravesearch"],
    "brave_news_search": ["brave_news_search", "brave news search", "brave_news"],
    "brave_local_search": ["brave_local_search", "brave local search", "brave_local"],
    "brave_image_search": ["brave_image_search", "brave image search", "brave_image"],
    "brave_video_search": ["brave_video_search", "brave video search", "brave_video"],
    # Exa tools
    "web_search_exa": ["web_search_exa", "exa_web_search", "exa search"],
    "get_code_context_exa": ["get_code_context_exa", "code_context_exa", "exa code"],
    "deep_researcher_start": ["deep_researcher_start", "deep_researcher", "exa deep research"],
    "company_research_exa": ["company_research_exa", "company_research", "exa company"],
    "crawling_exa": ["crawling_exa", "crawling", "exa crawl"],
    "linkedin_search_exa": ["linkedin_search_exa", "linkedin_search", "exa linkedin"],
    # Codex tools
    "codex": ["codex", "mcp__codex__codex", "codex researcher"],
    "codex-reply": ["codex-reply", "codex_reply", "mcp__codex__codex-reply"],
    # Local search tools
    "grep": ["grep", "built-in grep", "ripgrep", "rg"],
    "mgrep": ["mgrep", "semantic search", "mixedbread"],
    "glob": ["glob", "file glob", "pattern match"],
}

# Tool families for partial credit scoring
TOOL_FAMILIES = {
    "brave": ["brave_web_search", "brave_news_search", "brave_local_search",
              "brave_image_search", "brave_video_search"],
    "exa": ["web_search_exa", "get_code_context_exa", "deep_researcher_start",
            "company_research_exa", "crawling_exa", "linkedin_search_exa"],
    "codex": ["codex", "codex-reply"],
    "local_search": ["grep", "mgrep", "glob"],
}

# Related tool families for partial credit (0.3 score)
RELATED_FAMILIES = {
    "brave": ["exa"],  # Both are web search
    "exa": ["brave"],
    "codex": [],  # Codex is unique
    "local_search": [],
}


def normalize_tool_name(name: str) -> Optional[str]:
    """
    Normalize a tool name to its canonical form.

    Args:
        name: Raw tool name from text

    Returns:
        Canonical tool name or None if not recognized
    """
    if not name:
        return None

    name_lower = name.lower().strip()

    # Remove common prefixes
    for prefix in ["mcp__", "use ", "→ "]:
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix):]

    # Direct canonical match
    for canonical, aliases in CANONICAL_TOOLS.items():
        if name_lower == canonical:
            return canonical
        for alias in aliases:
            if name_lower == alias.lower():
                return canonical

    # Partial match (contains)
    for canonical, aliases in CANONICAL_TOOLS.items():
        if canonical in name_lower:
            return canonical
        for alias in aliases:
            if alias.lower() in name_lower:
                return canonical

    return None


def get_tool_family(tool: str) -> Optional[str]:
    """
    Get the family/category of a tool.

    Args:
        tool: Canonical tool name

    Returns:
        Family name ("brave", "exa", "codex", "local_search") or None
    """
    if not tool:
        return None

    tool_lower = tool.lower()

    for family, tools in TOOL_FAMILIES.items():
        if tool_lower in [t.lower() for t in tools]:
            return family

    return None


def _extract_explicit_header(text: str) -> Optional[str]:
    """
    Strategy 1: Look for explicit headers like **Tool Selection:** or **Tool:**

    Patterns:
    - **Tool Selection:** brave_web_search
    - **Tool:** web_search_exa
    - Tool: grep
    - Decision: mgrep
    """
    patterns = [
        r'\*\*Tool(?:\s*Selection)?[:\s]*\*\*[:\s]*([a-z_]+)',
        r'(?:^|\n)Tool(?:\s*Selection)?[:\s]+([a-z_]+)',
        r'\*\*Decision[:\s]*\*\*[:\s]*([a-z_]+)',
        r'(?:^|\n)Decision[:\s]+([a-z_]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return normalize_tool_name(match.group(1))

    return None


def _extract_use_pattern(text: str) -> Optional[str]:
    """
    Strategy 2: Look for "Use X" patterns.

    Patterns:
    - should use Brave: brave_local_search
    - Use Exa: web_search_exa
    - → Use brave_web_search
    - I recommend using mgrep
    """
    patterns = [
        r'(?:should|would|recommend)\s+use\s+(?:\w+:\s*)?([a-z_]+)',
        r'→\s*(?:Use\s+)?(?:\w+:\s*)?([a-z_]+)',
        r'use\s+(?:the\s+)?([a-z_]+(?:_[a-z]+)+)',  # Underscore-separated tool names
        r'recommend\s+(?:using\s+)?([a-z_]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            tool = normalize_tool_name(match.group(1))
            if tool:
                return tool

    return None


def _extract_bold_mention(text: str) -> Optional[str]:
    """
    Strategy 3: Look for bold tool mentions.

    Patterns:
    - **brave_local_search**
    - **get_code_context_exa**
    - **mgrep**
    """
    patterns = [
        r'\*\*([a-z_]+(?:_[a-z]+)*)\*\*',  # **tool_name**
        r'`([a-z_]+(?:_[a-z]+)*)`',  # `tool_name`
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            tool = normalize_tool_name(match)
            if tool:
                return tool

    return None


# Precompute (canonical, alias) pairs sorted by alias length descending so that
# longer, more specific names match before their substrings. This prevents
# "mgrep" being mis-extracted as "grep", or "codex-reply" as "codex".
_ALIAS_PAIRS_BY_LENGTH = sorted(
    [
        (canonical, alias)
        for canonical, aliases in CANONICAL_TOOLS.items()
        for alias in ([canonical] + aliases)
    ],
    key=lambda pair: len(pair[1]),
    reverse=True,
)


def _extract_direct_mention(text: str) -> Optional[str]:
    """
    Strategy 4: Look for direct tool name mentions in text.

    Scans for known tool names using word boundaries and longest-alias-first
    ordering so that substrings (grep within mgrep, codex within codex-reply)
    do not produce a wrong, shorter match.
    """
    text_lower = text.lower()

    for canonical, alias in _ALIAS_PAIRS_BY_LENGTH:
        alias_lower = alias.lower()
        # Boundary chars exclude the identifier characters used in tool names so
        # 'grep' won't match inside 'mgrep'.
        pattern = r"(?<![A-Za-z0-9_-])" + re.escape(alias_lower) + r"(?![A-Za-z0-9_-])"
        if re.search(pattern, text_lower):
            return canonical

    return None


def _extract_keyword_density(text: str) -> Optional[str]:
    """
    Strategy 5: Keyword density analysis (last resort).

    Counts keywords associated with each tool family and
    returns the tool with highest density.
    """
    text_lower = text.lower()

    # Keywords associated with each tool/family
    keyword_map = {
        # Brave tools
        "brave_web_search": ["brave", "keyword", "factual", "simple"],
        "brave_news_search": ["news", "current events", "breaking", "recent"],
        "brave_local_search": ["local", "nearby", "restaurant", "business"],
        "brave_image_search": ["image", "picture", "photo"],
        "brave_video_search": ["video", "youtube", "tutorial"],
        # Exa tools
        "web_search_exa": ["semantic", "neural", "exploratory", "complex query"],
        "get_code_context_exa": ["code", "api", "library", "programming", "implementation"],
        "deep_researcher_start": ["deep research", "comprehensive", "multi-source"],
        "company_research_exa": ["company", "competitor", "business research"],
        # Codex
        "codex": ["alternative ai", "cross-validation", "second opinion", "gpt"],
        # Local search
        "grep": ["exact", "literal", "function name", "import", "regex"],
        "mgrep": ["semantic", "natural language", "relevance", "exploratory"],
    }

    scores = {}
    for tool, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[tool] = score

    if scores:
        # Return tool with highest score
        best_tool = max(scores, key=scores.get)
        return best_tool

    return None


def extract_tool_from_verbose(text: str) -> Optional[str]:
    """
    Extract tool name from verbose Claude output using 5 fallback strategies.

    Strategies (in order of preference):
    1. Explicit headers (**Tool Selection:** X)
    2. "Use X" patterns (should use Brave: brave_local_search)
    3. Bold tool mentions (**brave_local_search**)
    4. Direct tool name detection
    5. Keyword density analysis

    Args:
        text: Verbose output from Claude

    Returns:
        Canonical tool name or None if no tool could be extracted
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Strategy 1: Explicit headers
    tool = _extract_explicit_header(text)
    if tool:
        return tool

    # Strategy 2: "Use X" patterns
    tool = _extract_use_pattern(text)
    if tool:
        return tool

    # Strategy 3: Bold mentions
    tool = _extract_bold_mention(text)
    if tool:
        return tool

    # Strategy 4: Direct mentions
    tool = _extract_direct_mention(text)
    if tool:
        return tool

    # Strategy 5: Keyword density (last resort)
    tool = _extract_keyword_density(text)
    if tool:
        return tool

    return None


def score_tool_match(expected: str, actual: str) -> float:
    """
    Score the match between expected and actual tool selections.

    Scoring:
    - 1.0: Exact match
    - 0.6: Same family (e.g., both are Brave tools)
    - 0.3: Related families (e.g., Brave vs Exa - both web search)
    - 0.0: Wrong family or no tool extracted

    Args:
        expected: Expected tool name (may need normalization)
        actual: Actual tool name from verbose output

    Returns:
        Score between 0.0 and 1.0
    """
    # Normalize both
    expected_norm = normalize_tool_name(expected)
    actual_norm = extract_tool_from_verbose(actual) if actual else None

    if not expected_norm:
        # Can't score if expected is invalid
        return 0.5

    if not actual_norm:
        # No tool extracted from actual
        return 0.0

    # Exact match
    if expected_norm == actual_norm:
        return 1.0

    # Same family
    expected_family = get_tool_family(expected_norm)
    actual_family = get_tool_family(actual_norm)

    if expected_family and expected_family == actual_family:
        return 0.6

    # Related families
    if expected_family and actual_family:
        related = RELATED_FAMILIES.get(expected_family, [])
        if actual_family in related:
            return 0.3

    return 0.0


def extract_binary_decision(text: str) -> Optional[str]:
    """
    Extract binary decision (grep vs mgrep) from verbose output.

    Args:
        text: Verbose output from Claude

    Returns:
        "grep" or "mgrep" or None
    """
    if not isinstance(text, str) or not text:
        return None

    text_lower = text.lower().strip()

    # Check for bare values (single word input)
    if text_lower == "grep":
        return "grep"
    if text_lower == "mgrep":
        return "mgrep"

    # Check for explicit decision patterns
    decision_patterns = [
        (r'\*\*Decision[:\s]*\*\*[:\s]*(grep|mgrep)', None),
        (r'(?:^|\n)Decision[:\s]+(grep|mgrep)', None),
        (r'use\s+(mgrep|grep)(?:\s|$)', None),
        (r'→\s*(mgrep|grep)', None),
    ]

    for pattern, _ in decision_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # Heuristic: count mentions
    mgrep_count = text_lower.count("mgrep")
    grep_count = text_lower.count("grep") - mgrep_count  # Don't double-count mgrep

    # Check semantic indicators
    semantic_keywords = ["semantic", "natural language", "exploratory", "meaning", "intent"]
    exact_keywords = ["exact", "literal", "function name", "import", "regex", "string"]

    semantic_score = mgrep_count + sum(1 for kw in semantic_keywords if kw in text_lower)
    exact_score = grep_count + sum(1 for kw in exact_keywords if kw in text_lower)

    if semantic_score > exact_score:
        return "mgrep"
    elif exact_score > semantic_score:
        return "grep"

    return None


def extract_tier(text: str) -> Optional[str]:
    """
    Extract tool tier classification from verbose output.

    Tiers:
    - core: Read, Edit, Write, Bash, Grep, Glob
    - specialized: Task, WebFetch, WebSearch, TodoWrite
    - deferred: MCP tools, Brave, Exa, Playwright, etc.

    Args:
        text: Verbose output from Claude

    Returns:
        "core", "specialized", or "deferred" or None
    """
    if not isinstance(text, str) or not text:
        return None

    text_lower = text.lower().strip()

    # Check for bare tier names (single word input)
    if text_lower in ("core", "specialized", "deferred"):
        return text_lower

    # Check for explicit tier mentions
    tier_patterns = [
        (r'\*\*Tier[:\s]*\*\*[:\s]*(core|specialized|deferred)', None),
        (r'(?:^|\n)Tier[:\s]+(core|specialized|deferred)', None),
        (r'(core|specialized|deferred)\s+tool', None),
    ]

    for pattern, _ in tier_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # Check for tool mentions to infer tier
    core_tools = ["read", "edit", "write", "bash", "grep", "glob"]
    specialized_tools = ["task", "webfetch", "websearch", "todowrite", "subagent"]
    deferred_tools = ["mcp", "brave", "exa", "playwright", "gemini", "notebook", "skill"]

    core_count = sum(1 for t in core_tools if t in text_lower)
    specialized_count = sum(1 for t in specialized_tools if t in text_lower)
    deferred_count = sum(1 for t in deferred_tools if t in text_lower)

    max_count = max(core_count, specialized_count, deferred_count)

    if max_count == 0:
        return None

    if core_count == max_count:
        return "core"
    elif specialized_count == max_count:
        return "specialized"
    else:
        return "deferred"


# =============================================================================
# Agent-Specific Extractors
# =============================================================================

@dataclass
class SeverityDetails:
    """Extracted severity information from code review output."""
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    warning_count: int = 0
    issues: List[dict] = None  # [{severity, title, file, line, description}]

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

    @property
    def total_count(self) -> int:
        return self.critical_count + self.high_count + self.medium_count + self.low_count + self.warning_count


@dataclass
class CoverageDetails:
    """Extracted test coverage information from test-writer output."""
    total_tests: int = 0
    happy_path_count: int = 0
    edge_case_count: int = 0
    error_case_count: int = 0
    tests: List[dict] = None  # [{name, type, description}]
    has_describe: bool = False
    has_it: bool = False
    has_expect: bool = False

    def __post_init__(self):
        if self.tests is None:
            self.tests = []


# Alias for backward compatibility with existing code
TestCoverageDetails = CoverageDetails


@dataclass
class EvaluationDetails:
    """Extracted evaluation information from capability-evaluator output."""
    score: Optional[float] = None
    decision: Optional[str] = None  # INTEGRATE, SKIP, RESEARCH_MORE
    criteria: List[dict] = None  # [{name, score, notes}]

    def __post_init__(self):
        if self.criteria is None:
            self.criteria = []


def _extract_severity_explicit_counts(text: str) -> Tuple[dict, bool]:
    """
    Strategy 1: Look for explicit count patterns.

    Patterns:
    - Critical: 2
    - High: 3 issues
    - 2 critical issues
    """
    counts = {}
    found_any = False

    severity_levels = ['critical', 'high', 'medium', 'low', 'warning']

    for level in severity_levels:
        patterns = [
            rf'{level}[:\s]*(\d+)',  # "Critical: 2" or "Critical 2"
            rf'(\d+)\s*{level}',  # "2 critical"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                counts[level] = int(match.group(1))
                found_any = True
                break

    return counts, found_any


def _extract_severity_markdown_headers(text: str) -> Tuple[dict, List[dict], bool]:
    """
    Strategy 2: Parse markdown headers like ### Critical Issues.

    Extracts counts by counting numbered items under each section.
    """
    counts = {}
    issues = []
    found_any = False

    # Map singular/plural variants to canonical names
    severity_variants = {
        'critical': ['critical', 'criticals'],
        'high': ['high', 'highs'],
        'medium': ['medium', 'mediums'],
        'low': ['low', 'lows'],
        'warning': ['warning', 'warnings'],
    }

    for level, variants in severity_variants.items():
        for variant in variants:
            # Match section header and content until next header or end
            # Match "### Warnings" or "### Warning" or "### Critical Issues"
            pattern = rf'###?\s*{variant}s?\s*(?:issues?)?\s*\n(.*?)(?=###?|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

            if match:
                section_content = match.group(1)

                # Count numbered items (1., 2., etc.)
                numbered_items = re.findall(r'^\s*(\d+)\.\s*(.+?)(?=^\s*\d+\.|\Z)',
                                           section_content, re.MULTILINE | re.DOTALL)

                if numbered_items:
                    counts[level] = len(numbered_items)
                    found_any = True

                    # Extract issue details
                    for _, item_text in numbered_items:
                        issue = _parse_issue_text(item_text, level)
                        if issue:
                            issues.append(issue)
                else:
                    # Check for bullet points
                    bullets = re.findall(r'^\s*[-*]\s*(.+)$', section_content, re.MULTILINE)
                    if bullets:
                        counts[level] = len(bullets)
                        found_any = True
                        for bullet in bullets:
                            issue = _parse_issue_text(bullet, level)
                            if issue:
                                issues.append(issue)
                break  # Found a match for this severity level

    return counts, issues, found_any


def _extract_severity_numbered_lists(text: str) -> Tuple[dict, List[dict], bool]:
    """
    Strategy 3: Parse numbered lists with severity indicators.

    Patterns:
    - 1. [Critical] SQL injection in...
    - 1. **Critical** - SQL injection...
    """
    counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'warning': 0}
    issues = []
    found_any = False

    # Pattern: numbered item with severity indicator
    patterns = [
        r'^\s*\d+\.\s*\[?(critical|high|medium|low|warning)\]?\s*[-:]\s*(.+)',
        r'^\s*\d+\.\s*\*\*(critical|high|medium|low|warning)\*\*\s*[-:]\s*(.+)',
        r'^\s*\d+\.\s*\((critical|high|medium|low|warning)\)\s*[-:]\s*(.+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for severity, description in matches:
            sev_lower = severity.lower()
            counts[sev_lower] = counts.get(sev_lower, 0) + 1
            found_any = True

            issue = _parse_issue_text(description, sev_lower)
            if issue:
                issues.append(issue)

    return counts, issues, found_any


def _extract_severity_keyword_density(text: str) -> Tuple[dict, bool]:
    """
    Strategy 4: Keyword density analysis.

    Counts severity-related keywords to estimate issue counts.
    """
    text_lower = text.lower()
    counts = {}

    # Severity indicators
    severity_keywords = {
        'critical': ['critical', 'severe', 'dangerous', 'must fix', 'immediate'],
        'high': ['high', 'serious', 'significant', 'important'],
        'medium': ['medium', 'moderate', 'should fix'],
        'low': ['low', 'minor', 'trivial', 'cosmetic'],
        'warning': ['warning', 'suggestion', 'consider', 'could'],
    }

    for level, keywords in severity_keywords.items():
        count = sum(text_lower.count(kw) for kw in keywords)
        if count > 0:
            # Normalize: assume ~3 keyword mentions per issue
            counts[level] = max(1, count // 3)

    return counts, bool(counts)


def _extract_severity_issue_patterns(text: str) -> Tuple[List[dict], bool]:
    """
    Strategy 5: Look for common issue description patterns.

    Extracts file:line references and issue descriptions.
    """
    issues = []

    # Pattern: file:line references with descriptions
    file_patterns = [
        r'`?([a-zA-Z0-9_/.-]+\.[a-z]+):(\d+)`?\s*[-:]\s*(.+)',
        r'\*\*([a-zA-Z0-9_/.-]+\.[a-z]+):(\d+)\*\*\s*(.+)',
        r'at\s+([a-zA-Z0-9_/.-]+\.[a-z]+):(\d+)\s*[-:]\s*(.+)',
    ]

    for pattern in file_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for file_path, line, description in matches:
            issues.append({
                'file': file_path,
                'line': int(line),
                'description': description.strip()[:200],
                'severity': None,  # Unknown without context
            })

    return issues, bool(issues)


def _parse_issue_text(text: str, severity: str) -> Optional[dict]:
    """Parse issue text to extract file, line, and description."""
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try to extract file:line
    file_match = re.search(r'`?([a-zA-Z0-9_/.-]+\.[a-z]+):(\d+)`?', text)

    issue = {
        'severity': severity,
        'title': text[:100].split('\n')[0],
        'description': text[:300],
    }

    if file_match:
        issue['file'] = file_match.group(1)
        issue['line'] = int(file_match.group(2))

    return issue


def extract_severity_details(text: str) -> SeverityDetails:
    """
    Extract severity classification details from code review output.

    Uses 5 fallback strategies:
    1. Explicit counts (Critical: 2, High: 3)
    2. Markdown headers (### Critical Issues)
    3. Numbered lists with severity ([Critical] - ...)
    4. Keyword density analysis
    5. Issue pattern extraction (file:line references)

    Args:
        text: Code review output (may be verbose markdown)

    Returns:
        SeverityDetails with counts and extracted issues
    """
    if not isinstance(text, str) or not text.strip():
        return SeverityDetails()

    all_issues = []
    final_counts = {}

    # Strategy 1: Explicit counts
    counts, found = _extract_severity_explicit_counts(text)
    if found:
        final_counts = counts

    # Strategy 2: Markdown headers (also extracts issues)
    # Always try markdown headers to extract issues, even if explicit counts found
    md_counts, md_issues, md_found = _extract_severity_markdown_headers(text)
    if md_found:
        if not final_counts:
            final_counts = md_counts
        all_issues.extend(md_issues)

    # Strategy 3: Numbered lists with severity
    if not final_counts:
        counts, issues, found = _extract_severity_numbered_lists(text)
        if found:
            final_counts = counts
            all_issues.extend(issues)

    # Strategy 4: Keyword density (last resort for counts)
    if not final_counts:
        counts, found = _extract_severity_keyword_density(text)
        if found:
            final_counts = counts

    # Strategy 5: Extract issues even if counts found elsewhere
    if not all_issues:
        issues, _ = _extract_severity_issue_patterns(text)
        all_issues.extend(issues)

    return SeverityDetails(
        critical_count=final_counts.get('critical', 0),
        high_count=final_counts.get('high', 0),
        medium_count=final_counts.get('medium', 0),
        low_count=final_counts.get('low', 0),
        warning_count=final_counts.get('warning', 0),
        issues=all_issues,
    )


def _extract_test_explicit_count(text: str) -> Tuple[int, bool]:
    """Strategy 1: Look for explicit test counts."""
    patterns = [
        r'total[:\s]*(\d+)\s*tests?',
        r'total\s+tests?[:\s]*(\d+)',  # matches "Total Tests: 12" (format instruction output)
        r'(\d+)\s*tests?\s*total',
        r'(\d+)\s*tests?\s*(?:written|generated|created)',
        r'test\s*count[:\s]*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1)), True

    return 0, False


def _extract_test_describe_it_blocks(text: str) -> Tuple[int, List[dict], bool]:
    """Strategy 2: Count describe/it/test blocks."""
    tests = []
    seen_names = set()  # Avoid duplicates

    # Count it() or test() blocks with quotes
    it_patterns = [
        r'\bit\s*\(\s*[\'"](.+?)[\'"]',
        r'\btest\s*\(\s*[\'"](.+?)[\'"]',
    ]

    for pattern in it_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            name = match[:100]
            if name not in seen_names:
                seen_names.add(name)
                tests.append({
                    'name': name,
                    'type': 'unknown',
                    'description': match[:200],
                })

    count = len(tests)
    return count, tests, count > 0


def _extract_test_function_names(text: str) -> Tuple[int, List[dict], bool]:
    """Strategy 3: Extract test function names."""
    tests = []
    count = 0

    # Test function patterns
    patterns = [
        r'def\s+(test_\w+)',
        r'async def\s+(test_\w+)',
        r'function\s+(test\w+)',
        r'const\s+(test\w+)\s*=',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            count += 1
            tests.append({
                'name': match,
                'type': 'function',
                'description': match.replace('_', ' ').replace('test', '').strip(),
            })

    return count, tests, count > 0


def _extract_test_categories(text: str) -> Tuple[dict, bool]:
    """Strategy 4: Identify test categories mentioned."""
    text_lower = text.lower()
    categories = {}

    category_keywords = {
        'happy_path': ['happy path', 'success', 'normal case', 'positive test', 'golden path'],
        'edge_cases': ['edge case', 'boundary', 'limit', 'empty', 'null', 'undefined', 'zero'],
        'error_cases': ['error case', 'exception', 'throw', 'reject', 'fail', 'invalid', 'negative test'],
    }

    for cat, keywords in category_keywords.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            categories[cat] = count

    return categories, bool(categories)


def _extract_test_structure(text: str) -> dict:
    """Strategy 5: Check for test structure elements."""
    return {
        'has_describe': bool(re.search(r'\bdescribe\s*\(', text)),
        'has_it': bool(re.search(r'\bit\s*\(', text)) or bool(re.search(r'\btest\s*\(', text)),
        'has_expect': bool(re.search(r'\bexpect\s*\(', text)) or bool(re.search(r'\bassert', text)),
    }


def extract_test_coverage_details(text: str) -> CoverageDetails:
    """
    Extract test coverage details from test-writer output.

    Uses 5 fallback strategies:
    1. Explicit test counts (Total: 15 tests)
    2. describe/it/test blocks
    3. Test function names (test_*, testFoo)
    4. Test category identification
    5. Structure element detection

    Args:
        text: Test suite output (may be verbose)

    Returns:
        TestCoverageDetails with counts and test list
    """
    if not isinstance(text, str) or not text.strip():
        return TestCoverageDetails()

    all_tests = []

    # Strategy 1: Explicit count
    total_count, found = _extract_test_explicit_count(text)

    # Strategy 2: describe/it blocks
    if not total_count:
        count, tests, found = _extract_test_describe_it_blocks(text)
        if found:
            total_count = count
            all_tests.extend(tests)

    # Strategy 3: Test function names
    if not total_count:
        count, tests, found = _extract_test_function_names(text)
        if found:
            total_count = count
            all_tests.extend(tests)

    # Strategy 4: Categories
    categories, _ = _extract_test_categories(text)

    # Strategy 5: Structure
    structure = _extract_test_structure(text)

    return TestCoverageDetails(
        total_tests=total_count,
        happy_path_count=categories.get('happy_path', 0),
        edge_case_count=categories.get('edge_cases', 0),
        error_case_count=categories.get('error_cases', 0),
        tests=all_tests,
        has_describe=structure['has_describe'],
        has_it=structure['has_it'],
        has_expect=structure['has_expect'],
    )


def _extract_eval_explicit_score(text: str) -> Tuple[Optional[float], bool]:
    """Strategy 1: Look for explicit score patterns."""
    patterns = [
        r'\*\*(?:final|overall|total|weighted)\s*score[:\*\s]*(\d+(?:\.\d+)?)',  # **Overall Score:** 88
        r'(?:final|overall|total|weighted)\s*score[:\s]*\*?\*?(\d+(?:\.\d+)?)',  # Overall Score: 88
        r'score[:\s]*(\d+(?:\.\d+)?)\s*/\s*100',  # Score: 88/100
        r'score[:\s]*(\d+(?:\.\d+)?)',  # Score: 88
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Normalize if it looks like a percentage > 1
            if score > 1 and score <= 100:
                return score, True
            elif score <= 1:
                return score * 100, True

    return None, False


def _extract_eval_bold_numbers(text: str) -> Tuple[Optional[float], bool]:
    """Strategy 2: Look for bold numbers (likely scores)."""
    patterns = [
        r'\*\*(\d+(?:\.\d+)?)\*\*\s*/\s*100',
        r'\*\*(\d+(?:\.\d+)?)/100\*\*',
        r'\*\*(\d+(?:\.\d+)?)\*\*\s*(?:points?|score)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1)), True

    return None, False


def _extract_eval_decision(text: str) -> Tuple[Optional[str], bool]:
    """Strategy 3: Extract decision (INTEGRATE/SKIP/RESEARCH_MORE)."""
    patterns = [
        r'\*\*decision[:\*\s]*(integrate|skip|research[_\s]*more)',  # **Decision:** INTEGRATE
        r'decision[:\s]*(integrate|skip|research[_\s]*more)',  # Decision: INTEGRATE
        r'\*\*(integrate|skip|research[_\s]*more)\*\*',  # **INTEGRATE**
        r'recommend(?:ation)?[:\s]*(integrate|skip|research[_\s]*more)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            decision = match.group(1).lower().replace(' ', '_').replace('-', '_')
            if decision == 'research_more':
                decision = 'RESEARCH_MORE'
            else:
                decision = decision.upper()
            return decision, True

    # Infer from context
    text_lower = text.lower()
    if 'should integrate' in text_lower or 'recommend integrating' in text_lower:
        return 'INTEGRATE', True
    if 'should skip' in text_lower or 'not recommend' in text_lower:
        return 'SKIP', True
    if 'more research' in text_lower or 'needs investigation' in text_lower:
        return 'RESEARCH_MORE', True

    return None, False


def _extract_eval_criteria(text: str) -> Tuple[List[dict], bool]:
    """Strategy 4: Extract evaluation criteria and sub-scores."""
    criteria = []

    # Look for criteria with scores
    patterns = [
        r'(?:^|\n)\s*[-*]\s*([A-Za-z\s]+)[:\s]*(\d+(?:\.\d+)?)\s*/\s*\d+',
        r'(?:^|\n)\s*\|?\s*([A-Za-z\s]+)\s*\|?\s*(\d+(?:\.\d+)?)\s*/?\s*\d*\s*\|?',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for name, score in matches:
            name = name.strip()
            if len(name) > 3 and len(name) < 50:  # Filter noise
                criteria.append({
                    'name': name,
                    'score': float(score),
                    'notes': '',
                })

    return criteria, bool(criteria)


def _extract_eval_percentage_patterns(text: str) -> Tuple[Optional[float], bool]:
    """Strategy 5: Look for percentage patterns."""
    patterns = [
        r'(\d+(?:\.\d+)?)\s*%\s*(?:match|accuracy|confidence)',
        r'confidence[:\s]*(\d+(?:\.\d+)?)\s*%?',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:score|rating)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1)), True

    return None, False


def extract_evaluation_details(text: str) -> EvaluationDetails:
    """
    Extract evaluation details from capability-evaluator output.

    Uses 5 fallback strategies:
    1. Explicit score patterns (Score: 85/100)
    2. Bold numbers (**85**/100)
    3. Decision extraction (INTEGRATE/SKIP)
    4. Criteria extraction
    5. Percentage patterns

    Args:
        text: Evaluation output (may be verbose)

    Returns:
        EvaluationDetails with score, decision, and criteria
    """
    if not isinstance(text, str) or not text.strip():
        return EvaluationDetails()

    # Strategy 1: Explicit score
    score, found = _extract_eval_explicit_score(text)

    # Strategy 2: Bold numbers
    if score is None:
        score, found = _extract_eval_bold_numbers(text)

    # Strategy 5: Percentage patterns (before decision, as fallback for score)
    if score is None:
        score, found = _extract_eval_percentage_patterns(text)

    # Strategy 3: Decision
    decision, _ = _extract_eval_decision(text)

    # Strategy 4: Criteria
    criteria, _ = _extract_eval_criteria(text)

    return EvaluationDetails(
        score=score,
        decision=decision,
        criteria=criteria,
    )


# =============================================================================
# Publication Review Extractor
# =============================================================================

@dataclass
class ReviewFinding:
    """A single finding from a publication review."""
    tier: str  # MUST, SHOULD, NICE
    description: str
    keywords: set = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = _extract_keywords(self.description)


def _extract_keywords(text: str) -> set:
    """Extract content word stems from text for matching.

    Uses crude stemming (first 5+ chars) to bridge vocabulary gaps between
    manifest text (quotes post) and model output (uses review language).
    """
    # Remove markdown formatting, quotes, code, URLs
    cleaned = re.sub(r'[*`#|"\'()\[\]{}]', ' ', text.lower())
    cleaned = re.sub(r'https?://\S+', ' ', cleaned)
    # Remove stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'as', 'into', 'through', 'during', 'before', 'after', 'above',
                 'below', 'between', 'under', 'about', 'this', 'that', 'these',
                 'those', 'it', 'its', 'and', 'but', 'or', 'nor', 'not', 'no',
                 'so', 'if', 'then', 'than', 'when', 'where', 'how', 'what',
                 'which', 'who', 'whom', 'whose', 'why', 'all', 'each', 'every',
                 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
                 'own', 'same', 'too', 'very', 'also', 'just', 'still', 'yet'}

    words = set()
    for w in cleaned.split():
        w = w.strip('.,;:!?-—/')
        if len(w) >= 3 and not w.isdigit() and w not in stopwords:
            words.add(w)
            # Add crude stem (first 5 chars) for longer words to bridge
            # vocabulary gaps like "methodology" vs "method", "performance" vs "perform"
            if len(w) >= 6:
                words.add(w[:5])
    return words


def extract_publication_review(text: str) -> str:
    """
    Normalize verbose review output into structured finding format.

    Handles various output formats models produce:
    - Standard ## MUST FIX / ## SHOULD FIX / ## NICE TO HAVE headers
    - Numbered lists without headers
    - Table format
    - Prose with embedded findings

    Args:
        text: Raw review output (may be verbose)

    Returns:
        Normalized text with ## MUST FIX / ## SHOULD FIX / ## NICE TO HAVE sections
    """
    if not text or not text.strip():
        return ""

    findings = extract_review_findings(text)
    if not findings:
        return text  # Return original if we can't parse

    return _format_findings(findings)


def extract_review_findings(text: str) -> List[ReviewFinding]:
    """
    Extract findings from review text using multiple strategies.

    Returns list of ReviewFinding objects with tier, description, and keywords.
    """
    if not text or not text.strip():
        return []

    # Strategy 1: Parse structured headers (## MUST FIX, etc.)
    findings = _extract_findings_from_headers(text)
    if findings:
        return findings

    # Strategy 2: Parse table format
    findings = _extract_findings_from_tables(text)
    if findings:
        return findings

    # Strategy 2.5: Parse ### section headers with bullets (natural model output)
    findings = _extract_findings_from_sections(text)
    if findings:
        return findings

    # Strategy 3: Parse numbered lists with tier indicators
    findings = _extract_findings_from_numbered_lists(text)
    if findings:
        return findings

    # Strategy 4: Keyword-based extraction from prose
    findings = _extract_findings_from_prose(text)
    return findings


def _extract_findings_from_headers(text: str) -> List[ReviewFinding]:
    """Strategy 1: Parse ## MUST FIX / ## SHOULD FIX / ## NICE TO HAVE sections."""
    findings = []

    tier_patterns = [
        ("MUST", r'##\s*MUST\s*FIX\s*\n(.*?)(?=\n##|\Z)'),
        ("SHOULD", r'##\s*SHOULD\s*FIX\s*\n(.*?)(?=\n##|\Z)'),
        ("NICE", r'##\s*NICE\s*(?:TO\s*)?HAVE\s*\n(.*?)(?=\n##|\Z)'),
    ]

    for tier, pattern in tier_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            continue

        section = match.group(1)

        # Extract numbered items
        items = re.findall(r'^\s*\d+\.\s*(.+?)(?=^\s*\d+\.|\Z)', section, re.MULTILINE | re.DOTALL)
        for item in items:
            desc = item.strip()
            if desc and len(desc) > 10:
                # Clean up: take first line if multi-line
                first_line = desc.split('\n')[0].strip()
                # Remove bold markers for keyword extraction
                findings.append(ReviewFinding(tier=tier, description=first_line))

        # Also check for bullet points if no numbered items
        if not items:
            bullets = re.findall(r'^\s*[-*]\s+(.+)$', section, re.MULTILINE)
            for bullet in bullets:
                desc = bullet.strip()
                if desc and len(desc) > 10:
                    findings.append(ReviewFinding(tier=tier, description=desc))

        # Also check for table rows (| # | Finding | ...)
        if not items:
            rows = re.findall(r'\|\s*\d+\s*\|([^|]+)', section)
            for row in rows:
                desc = row.strip()
                if desc and len(desc) > 10 and not desc.startswith("Finding") and not desc.startswith("---"):
                    findings.append(ReviewFinding(tier=tier, description=desc))

    return findings


def _extract_findings_from_tables(text: str) -> List[ReviewFinding]:
    """Strategy 2: Parse findings from markdown tables with tier columns."""
    findings = []

    # Look for tables with Priority/Tier column
    rows = re.findall(
        r'\|\s*\d+\s*\|\s*(MUST|SHOULD|NICE|must|should|nice)[^|]*\|([^|]+)',
        text, re.IGNORECASE
    )
    for tier_str, desc in rows:
        tier = tier_str.strip().upper()
        if tier == "NICE":
            tier = "NICE"
        desc = desc.strip()
        if desc and len(desc) > 10:
            findings.append(ReviewFinding(tier=tier, description=desc))

    return findings


def _extract_findings_from_sections(text: str) -> List[ReviewFinding]:
    """Strategy 2.5: Parse ### section headers with bullets (natural model output format).

    Models naturally produce output like:
        ### 1. Factual Errors & Technical Accuracy
        *   **Claim X:** is wrong because...
        *   **Claim Y:** needs citation...
        ### 2. Internal Consistency
        *   **Contradiction:** ...

    Map section headers to tiers by keyword:
        Factual/accuracy/error/incorrect/citation → MUST
        Consistency/contradiction/tone/fairness → SHOULD
        Style/polish/minor/suggestion/hedging → NICE
    """
    findings = []

    # Find all ### sections
    sections = re.split(r'\n###\s+', text)
    if len(sections) <= 1:
        # Try ## sections too
        sections = re.split(r'\n##\s+', text)
    if len(sections) <= 1:
        return findings

    must_headers = ['factual', 'accuracy', 'error', 'incorrect', 'citation',
                    'unsupported', 'verif', 'numerical', 'evidence']
    should_headers = ['consisten', 'contradict', 'tone', 'fairness', 'overclaim',
                      'hedg', 'internal', 'logic', 'coherence']
    nice_headers = ['style', 'polish', 'minor', 'suggest', 'redundan', 'sentence',
                    'clarity', 'readability', 'nitpick']

    for section in sections[1:]:  # Skip preamble before first header
        # Get header line
        header_line = section.split('\n')[0].lower()

        # Classify tier from header
        if any(kw in header_line for kw in must_headers):
            tier = "MUST"
        elif any(kw in header_line for kw in should_headers):
            tier = "SHOULD"
        elif any(kw in header_line for kw in nice_headers):
            tier = "NICE"
        else:
            tier = "SHOULD"  # Default to SHOULD for unclassifiable sections

        # Extract bullets from section
        bullets = re.findall(r'^\s*[*\-]\s+(.+)$', section, re.MULTILINE)
        for bullet in bullets:
            desc = bullet.strip()
            if desc and len(desc) > 15:
                # Clean leading bold markers
                desc = re.sub(r'^\*\*([^*]+)\*\*[:\s]*', r'\1: ', desc)
                findings.append(ReviewFinding(tier=tier, description=desc[:300]))

        # Also numbered items in section
        if not bullets:
            items = re.findall(r'^\s*\d+\.\s+(.+)$', section, re.MULTILINE)
            for item in items:
                desc = item.strip()
                if desc and len(desc) > 15:
                    desc = re.sub(r'^\*\*([^*]+)\*\*[:\s]*', r'\1: ', desc)
                    findings.append(ReviewFinding(tier=tier, description=desc[:300]))

    return findings


def _extract_findings_from_numbered_lists(text: str) -> List[ReviewFinding]:
    """Strategy 3: Parse numbered lists with severity/tier indicators."""
    findings = []

    patterns = [
        r'^\s*\d+\.\s*\*?\*?\[?(MUST|SHOULD|NICE)\s*(?:FIX|TO HAVE)?\]?\*?\*?\s*[-:]\s*(.+)',
        r'^\s*\d+\.\s*\*?\*?\((MUST|SHOULD|NICE)\)\*?\*?\s*[-:]\s*(.+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for tier_str, desc in matches:
            tier = tier_str.strip().upper()
            desc = desc.strip()
            if desc and len(desc) > 10:
                findings.append(ReviewFinding(tier=tier, description=desc))

    return findings


def _extract_findings_from_prose(text: str) -> List[ReviewFinding]:
    """Strategy 4: Extract from prose using severity keywords."""
    findings = []
    text_lower = text.lower()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    must_keywords = ['factual error', 'incorrect', 'wrong', 'false', 'contradicts',
                     'unsupported claim', 'no evidence', 'citation needed', 'misleading']
    should_keywords = ['overclaim', 'overstated', 'hedge', 'qualification needed',
                       'tone', 'could be clearer', 'ambiguous']

    for sentence in sentences:
        s_lower = sentence.lower()
        if any(kw in s_lower for kw in must_keywords):
            findings.append(ReviewFinding(tier="MUST", description=sentence.strip()[:200]))
        elif any(kw in s_lower for kw in should_keywords):
            findings.append(ReviewFinding(tier="SHOULD", description=sentence.strip()[:200]))

    return findings


def _format_findings(findings: List[ReviewFinding]) -> str:
    """Format findings into standard ## MUST FIX / ## SHOULD FIX / ## NICE TO HAVE structure."""
    sections = {"MUST": [], "SHOULD": [], "NICE": []}
    for f in findings:
        sections[f.tier].append(f.description)

    parts = []
    if sections["MUST"]:
        parts.append("## MUST FIX")
        for i, desc in enumerate(sections["MUST"], 1):
            parts.append(f"{i}. {desc}")

    if sections["SHOULD"]:
        parts.append("\n## SHOULD FIX")
        for i, desc in enumerate(sections["SHOULD"], 1):
            parts.append(f"{i}. {desc}")

    if sections["NICE"]:
        parts.append("\n## NICE TO HAVE")
        for i, desc in enumerate(sections["NICE"], 1):
            parts.append(f"{i}. {desc}")

    return "\n".join(parts)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# Known technical terms for anchor extraction (lowercase)
_KNOWN_TECH_TERMS = frozenset([
    'next.js', 'astro', 'react', 'vue', 'angular', 'svelte', 'nuxt',
    'cloudflare', 'vercel', 'aws', 'docker', 'kubernetes', 'postgresql',
    'mongodb', 'redis', 'typescript', 'javascript', 'python', 'rust', 'go',
    'css', 'html', 'dns', 'ci/cd', 'jwt', 'oauth', 'cors', 'xss', 'sql',
    'graphql', 'websocket', 'ssr', 'csr', 'spa', 'seo', 'rlhf', 'llm',
    'gpt', 'claude', 'gemini', 'opus', 'anthropic', 'openai', 'changshu',
    'webvan', 'instacart', 'facebook', 'myspace', 'friendster',
    'cb insights', 'khan academy', 'youtube', 'google',
])


def _extract_anchors(text: str) -> set:
    """Extract high-signal anchor tokens: entities, numbers, tech terms, quoted phrases.

    These tokens are preserved across paraphrases because both the manifest
    and the model output reference the same blog post. Entities like "Next.js",
    numbers like "40%", and proper nouns like "Changshu" appear in both
    regardless of whether the surrounding language differs.
    """
    anchors = set()
    text_lower = text.lower()

    # 1. Numbers with optional units (40%, 2025, 0.34s, 8KB, 90%, 42%)
    for m in re.findall(r'\b(\d+(?:\.\d+)?(?:%|kb|mb|ms|s|k)?)\b', text_lower):
        anchors.add(m)

    # 2. Capitalized words (proper nouns, tech terms)
    for m in re.findall(r'\b([A-Z][a-z]+(?:\.?[a-z]+)*)\b', text):
        if len(m) >= 3:
            anchors.add(m.lower())
    # ALL-CAPS acronyms (DNS, CI/CD, RLHF, etc.)
    for m in re.findall(r'\b([A-Z]{2,}(?:/[A-Z]+)?)\b', text):
        anchors.add(m.lower())

    # 3. Known tech terms in any case context
    for term in _KNOWN_TECH_TERMS:
        if term in text_lower:
            anchors.add(term)

    # 4. Significant words from quoted phrases
    for phrase in re.findall(r'"([^"]+)"', text):
        for w in phrase.lower().split():
            w = w.strip('.,;:!?-')
            if len(w) >= 4 and w not in {'that', 'this', 'with', 'from', 'they',
                                          'have', 'been', 'were', 'does', 'also',
                                          'more', 'than', 'very', 'much', 'some',
                                          'what', 'when', 'where', 'which', 'their'}:
                anchors.add(w)

    return anchors


def _char_ngrams(text: str, ns: tuple = (3, 4)) -> set:
    """Extract character n-grams from text (lowercase, whitespace-normalized).

    Character n-grams catch partial word overlaps that stem-based Jaccard misses.
    E.g., "methodology" and "methodological" share 3-grams and 4-grams.
    """
    cleaned = re.sub(r'\s+', ' ', text.lower().strip())
    grams = set()
    for n in ns:
        for i in range(len(cleaned) - n + 1):
            grams.add(cleaned[i:i + n])
    return grams


def hybrid_finding_similarity(text_a: str, text_b: str) -> float:
    """Compute similarity between two finding descriptions using three signals.

    Combines:
    - 50% anchor overlap (Szymkiewicz-Simpson on entities/numbers/tech terms)
    - 30% character n-gram coverage (catches partial word overlaps across paraphrases)
    - 20% keyword Jaccard (content word stems, weakest but broadest signal)

    Szymkiewicz-Simpson (overlap / min set size) handles asymmetric lengths —
    a short model description matching against a long manifest description.
    """
    # Anchor component (50%): entities, numbers, tech terms
    anchors_a = _extract_anchors(text_a)
    anchors_b = _extract_anchors(text_b)

    if anchors_a and anchors_b:
        overlap = anchors_a & anchors_b
        denom = min(len(anchors_a), len(anchors_b))
        anchor_score = len(overlap) / denom if denom > 0 else 0.0
    else:
        anchor_score = 0.0

    # Character n-gram component (30%): catches partial word overlaps
    ngrams_a = _char_ngrams(text_a)
    ngrams_b = _char_ngrams(text_b)
    if ngrams_a and ngrams_b:
        ngram_overlap = ngrams_a & ngrams_b
        # Szymkiewicz-Simpson: overlap relative to the shorter text's n-grams
        ngram_denom = min(len(ngrams_a), len(ngrams_b))
        ngram_score = len(ngram_overlap) / ngram_denom if ngram_denom > 0 else 0.0
    else:
        ngram_score = 0.0

    # Content keyword component (20%): Jaccard on stems
    kw_a = _extract_keywords(text_a)
    kw_b = _extract_keywords(text_b)
    kw_score = jaccard_similarity(kw_a, kw_b)

    return 0.50 * anchor_score + 0.30 * ngram_score + 0.20 * kw_score


def match_review_findings(
    expected: List[ReviewFinding],
    actual: List[ReviewFinding],
    threshold: float = 0.15,
) -> List[Tuple[ReviewFinding, Optional[ReviewFinding], float]]:
    """
    Bipartite matching of expected to actual findings using hybrid similarity.

    Uses anchor-based matching (entities, numbers, tech terms) combined with
    keyword Jaccard. This bridges the vocabulary gap between manifest descriptions
    (which quote the post) and model output (which uses review language).

    Returns list of (expected_finding, matched_actual_or_None, similarity_score).
    """
    if not expected:
        return []
    if not actual:
        return [(e, None, 0.0) for e in expected]

    used_actual = set()
    matches = []

    for exp in expected:
        best_score = 0.0
        best_idx = -1

        for i, act in enumerate(actual):
            if i in used_actual:
                continue
            sim = hybrid_finding_similarity(exp.description, act.description)
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_score >= threshold and best_idx >= 0:
            used_actual.add(best_idx)
            matches.append((exp, actual[best_idx], best_score))
        else:
            matches.append((exp, None, 0.0))

    return matches
