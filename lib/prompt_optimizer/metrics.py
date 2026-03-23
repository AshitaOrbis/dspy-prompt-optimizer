"""
Metric functions for evaluating prompt outputs.

Each metric returns a score between 0.0 and 1.0, where:
- 1.0 = perfect match
- 0.0 = no match
"""

import re
from typing import Callable, List, Optional, Dict, Any, Set
from difflib import SequenceMatcher


# =============================================================================
# Semantic Equivalence Data Structures
# =============================================================================

# Change type equivalence groups for pr_quality_match
CHANGE_TYPE_EQUIVALENCES: Dict[str, Set[str]] = {
    'additive': {'feature', 'enhancement', 'new_feature', 'feature_addition', 'adds', 'implement'},
    'corrective': {'bugfix', 'bug_fix', 'fix', 'patch', 'security', 'hotfix', 'fixes'},
    'structural': {'refactor', 'cleanup', 'reorganize', 'migration', 'restructure'},
    'meta': {'docs', 'documentation', 'test', 'testing', 'config', 'ci', 'chore', 'build'},
}

# Related change type groups (for partial credit)
CHANGE_TYPE_RELATED: Dict[str, List[str]] = {
    'additive': ['structural'],
    'corrective': ['additive'],
    'structural': ['additive', 'corrective'],
    'meta': [],
}

# Technique equivalence groups for refactoring_match (expanded)
TECHNIQUE_EQUIVALENCES: Dict[str, Set[str]] = {
    'decomposition': {
        'extract_method', 'extract_function', 'extract_class',
        'pipeline_pattern', 'compose_method', 'split_phase',
        'break_into', 'smaller_function',
    },
    'polymorphism': {
        'strategy_pattern', 'state_pattern', 'command_pattern',
        'replace_conditional_with_polymorphism', 'renderer_registry',
    },
    'data_organization': {
        'extract_class', 'extract_configuration', 'lookup_table',
        'introduce_parameter_object', 'value_object', 'builder_pattern',
        'config_driven_events',
    },
    'orchestration': {
        'facade_and_events', 'saga_pattern', 'saga_orchestrator',
        'event_sourcing', 'command_handler',
    },
    'abstraction': {
        'extract_generic_base', 'template_method', 'generic_helper',
        'abstract_factory', 'introduce_interface',
    },
    'efficiency': {
        'batch_fetching', 'dataloader_pattern', 'parallel_fetch',
        'caching', 'memoization',
    },
}

# Smell-technique affinity (which technique groups are valid for which smells)
SMELL_TECHNIQUE_AFFINITY: Dict[str, List[str]] = {
    'long_method': ['decomposition', 'orchestration'],
    'duplicate_code': ['abstraction', 'decomposition'],
    'conditional_complexity': ['polymorphism', 'data_organization'],
    'switch_statement': ['polymorphism', 'data_organization'],
    'type_conditional': ['polymorphism', 'data_organization'],
    'god_class': ['decomposition', 'orchestration'],
    'feature_envy': ['decomposition', 'abstraction'],
    'data_clumps': ['data_organization', 'abstraction'],
    'primitive_obsession': ['data_organization'],
    'hardcoded_rules': ['data_organization', 'abstraction'],
    'n_plus_one_queries': ['efficiency'],
    'orchestration_in_service': ['orchestration', 'decomposition'],
}

# Code smell equivalence groups (for semantic matching)
SMELL_EQUIVALENCES: Dict[str, Set[str]] = {
    'complexity': {
        'long_method', 'god_class', 'orchestration_in_service',
        'complex_method', 'too_many_lines', 'multiple_responsibilities',
    },
    'duplication': {
        'duplicate_code', 'shotgun_surgery', 'dry_violation',
        'repeated_code', 'copy_paste',
    },
    'conditionals': {
        'conditional_complexity', 'switch_statement', 'type_conditional',
        'nested_if', 'cyclomatic',
    },
    'data_issues': {
        'data_clumps', 'primitive_obsession', 'hardcoded_rules',
        'magic_number', 'magic_string',
    },
    'performance': {
        'n_plus_one_queries', 'inefficient_data_access',
        'sequential_query',
    },
    'coupling': {
        'feature_envy', 'inappropriate_intimacy',
        'knows_too_much',
    },
}


# =============================================================================
# Semantic Matching Helpers
# =============================================================================

def _normalize_categorical_value(value: str) -> str:
    """Normalize a categorical value for comparison."""
    if not value:
        return ''
    return value.strip().lower().replace('-', '_').replace(' ', '_')


def _find_group_for_value(
    value: str,
    equivalences: Dict[str, Set[str]],
) -> Optional[str]:
    """Find the equivalence group that contains the given value."""
    value_norm = _normalize_categorical_value(value)
    if not value_norm:
        return None

    for group_name, members in equivalences.items():
        # Check direct membership
        if value_norm in members:
            return group_name
        # Check normalized membership
        normalized_members = {_normalize_categorical_value(m) for m in members}
        if value_norm in normalized_members:
            return group_name

    return None


def _score_categorical_match(
    expected: Optional[str],
    actual: Optional[str],
    equivalences: Dict[str, Set[str]],
    related_groups: Optional[Dict[str, List[str]]] = None,
) -> float:
    """
    Score categorical match with semantic similarity.

    Scoring levels:
    - 1.0: Exact match or same equivalence group
    - 0.6: Related group (via related_groups mapping)
    - 0.3: Both have a group but unrelated
    - 0.2: Only one has a group (partial information)
    - 0.0: Missing actual value

    Args:
        expected: Expected categorical value
        actual: Actual categorical value
        equivalences: Dict mapping group names to sets of equivalent values
        related_groups: Optional dict mapping group names to related group names

    Returns:
        Score between 0.0 and 1.0
    """
    if not expected:
        return 0.5  # No expectation to match against

    if not actual:
        return 0.0  # Missing actual value

    expected_norm = _normalize_categorical_value(expected)
    actual_norm = _normalize_categorical_value(actual)

    # Exact match (normalized)
    if expected_norm == actual_norm:
        return 1.0

    # Find groups for both values
    expected_group = _find_group_for_value(expected, equivalences)
    actual_group = _find_group_for_value(actual, equivalences)

    # Same equivalence group
    if expected_group and expected_group == actual_group:
        return 1.0

    # Check for related groups
    if expected_group and actual_group and related_groups:
        related = related_groups.get(expected_group, [])
        if actual_group in related:
            return 0.6

    # Both have groups but not related
    if expected_group and actual_group:
        return 0.3

    # Only one has a group (partial information)
    if expected_group or actual_group:
        return 0.2

    # Neither has a group - fall back to simple substring check
    if actual_norm in expected_norm or expected_norm in actual_norm:
        return 0.4

    return 0.1  # Both present but no semantic overlap


def _score_technique_for_smell(
    smell: Optional[str],
    technique: Optional[str],
    smell_equivalences: Dict[str, Set[str]],
    technique_equivalences: Dict[str, Set[str]],
    affinity: Dict[str, List[str]],
) -> float:
    """
    Score whether a technique is valid for addressing a detected smell.

    Uses smell-technique affinity to determine if the suggested technique
    is appropriate for the identified code smell.

    Args:
        smell: Detected code smell
        technique: Suggested refactoring technique
        smell_equivalences: Smell groupings
        technique_equivalences: Technique groupings
        affinity: Mapping of smells to valid technique groups

    Returns:
        Score between 0.0 and 1.0
    """
    if not smell or not technique:
        return 0.0 if not technique else 0.3  # Technique without smell = partial credit

    smell_norm = _normalize_categorical_value(smell)
    technique_norm = _normalize_categorical_value(technique)

    # Find the canonical smell (may be a member of a group)
    canonical_smell = None
    for smell_group, members in smell_equivalences.items():
        normalized_members = {_normalize_categorical_value(m) for m in members}
        if smell_norm in normalized_members or smell_norm == smell_group:
            # Find first affinity key that matches this group
            for affinity_key in affinity.keys():
                if _normalize_categorical_value(affinity_key) in normalized_members:
                    canonical_smell = affinity_key
                    break
            if canonical_smell:
                break

    # If no canonical smell found, try direct lookup
    if not canonical_smell:
        for affinity_key in affinity.keys():
            if _normalize_categorical_value(affinity_key) == smell_norm:
                canonical_smell = affinity_key
                break

    if not canonical_smell:
        return 0.3  # Unknown smell, give partial credit for any technique

    # Find the technique's group
    technique_group = _find_group_for_value(technique, technique_equivalences)

    # Check if technique group is valid for this smell
    valid_groups = affinity.get(canonical_smell, [])

    if technique_group and technique_group in valid_groups:
        return 1.0  # Valid technique for this smell

    if technique_group:
        return 0.4  # Known technique but not ideal for this smell

    return 0.2  # Unknown technique


def exact_match(expected: str, actual: str) -> float:
    """
    Check if output exactly matches expected (case-insensitive, whitespace-normalized).

    Args:
        expected: Expected output string
        actual: Actual output from model

    Returns:
        1.0 if match, 0.0 otherwise
    """
    norm_expected = " ".join(expected.lower().split())
    norm_actual = " ".join(actual.lower().split())
    return 1.0 if norm_expected == norm_actual else 0.0


def contains_keywords(keywords: List[str], case_sensitive: bool = False) -> Callable[[str, str], float]:
    """
    Factory that returns a metric checking for keyword presence.

    Args:
        keywords: List of required keywords
        case_sensitive: Whether to do case-sensitive matching

    Returns:
        Metric function that scores based on keyword presence
    """
    def metric(expected: str, actual: str) -> float:
        if not keywords:
            return 1.0

        check_actual = actual if case_sensitive else actual.lower()
        matches = sum(
            1 for kw in keywords
            if (kw if case_sensitive else kw.lower()) in check_actual
        )
        return matches / len(keywords)

    return metric


def score_similarity(expected: str, actual: str) -> float:
    """
    Compare string similarity using SequenceMatcher.

    Good for evaluating free-form text outputs where exact match is too strict.

    Args:
        expected: Expected output
        actual: Actual output

    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()


def numeric_score_match(expected: str, actual: str, tolerance: float = 0.1) -> float:
    """
    Extract and compare numeric scores from outputs.

    Useful for evaluations that produce scores like "Score: 85/100".

    Args:
        expected: Expected output with numeric score
        actual: Actual output with numeric score
        tolerance: Acceptable deviation as fraction (0.1 = 10%)

    Returns:
        1.0 if scores match within tolerance, scaled down otherwise
    """
    def extract_score(text: str) -> Optional[float]:
        # Try various score patterns
        patterns = [
            r'(?:score|rating|total)[:\s]*(\d+(?:\.\d+)?)\s*[/out of]*\s*(\d+)?',
            r'(\d+(?:\.\d+)?)\s*/\s*(\d+)',
            r'(\d+(?:\.\d+)?)\s*%',
            r'(?:^|\s)(\d{1,3}(?:\.\d+)?)\s*(?:points?|pts?)?(?:\s|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-100 if there's a denominator
                if match.lastindex >= 2 and match.group(2):
                    denom = float(match.group(2))
                    if denom > 0:
                        score = (score / denom) * 100
                return score
        return None

    expected_score = extract_score(expected)
    actual_score = extract_score(actual)

    if expected_score is None or actual_score is None:
        # Fall back to string similarity if no scores found
        return score_similarity(expected, actual) * 0.5

    # Calculate how close the scores are
    max_score = max(expected_score, actual_score, 1)
    diff = abs(expected_score - actual_score)
    acceptable_diff = max_score * tolerance

    if diff <= acceptable_diff:
        return 1.0
    else:
        # Linear decay beyond tolerance
        excess = diff - acceptable_diff
        return max(0.0, 1.0 - (excess / max_score))


def composite_metric(metrics: List[Callable], weights: Optional[List[float]] = None) -> Callable[[str, str], float]:
    """
    Combine multiple metrics with optional weighting.

    Args:
        metrics: List of metric functions
        weights: Optional weights (must sum to 1.0 or will be normalized)

    Returns:
        Combined metric function

    Raises:
        ValueError: If metrics is empty or weights length doesn't match metrics
    """
    if not metrics:
        raise ValueError("metrics list cannot be empty")

    if weights is None:
        weights = [1.0 / len(metrics)] * len(metrics)
    else:
        if len(weights) != len(metrics):
            raise ValueError(f"weights length ({len(weights)}) must match metrics length ({len(metrics)})")
        # Normalize weights
        total = sum(weights)
        if total == 0:
            raise ValueError("weights cannot sum to zero")
        weights = [w / total for w in weights]

    def combined(expected: str, actual: str) -> float:
        scores = [m(expected, actual) for m in metrics]
        return sum(s * w for s, w in zip(scores, weights))

    return combined


def structured_output_match(expected: Dict[str, Any], actual: str) -> float:
    """
    Check if actual output contains expected structured fields.

    Useful for evaluating outputs that should contain specific key-value pairs.

    Args:
        expected: Dict of field -> expected value patterns
        actual: Actual output text

    Returns:
        Proportion of fields matched
    """
    if not expected:
        return 1.0

    matches = 0
    for field, pattern in expected.items():
        # Look for field: value pattern
        field_pattern = rf'{re.escape(field)}[:\s]+(.+?)(?:\n|$)'
        match = re.search(field_pattern, actual, re.IGNORECASE)

        if match:
            found_value = match.group(1).strip()
            if isinstance(pattern, str):
                if pattern.lower() in found_value.lower():
                    matches += 1
            elif isinstance(pattern, (int, float)):
                try:
                    extracted = float(re.search(r'[\d.]+', found_value).group())
                    if abs(extracted - pattern) < pattern * 0.15:  # 15% tolerance
                        matches += 1
                except (ValueError, AttributeError):
                    pass

    return matches / len(expected)


def evaluation_score_metric(threshold: float = 70.0) -> Callable[[str, str], float]:
    """
    Specific metric for capability evaluation scores.

    Designed for evaluator outputs that follow our scoring format.

    Args:
        threshold: Score threshold for "good" evaluations

    Returns:
        Metric function for evaluation scoring
    """
    def metric(expected: str, actual: str) -> float:
        # Extract the overall score
        score_pattern = r'(?:initial|final|overall|weighted|total)\s*score[:\s]*(\d+(?:\.\d+)?)'
        expected_match = re.search(score_pattern, expected, re.IGNORECASE)
        actual_match = re.search(score_pattern, actual, re.IGNORECASE)

        if not expected_match or not actual_match:
            # Fall back to looking for any score
            return numeric_score_match(expected, actual)

        expected_score = float(expected_match.group(1))
        actual_score = float(actual_match.group(1))

        # Check if both agree on pass/fail relative to threshold
        expected_pass = expected_score >= threshold
        actual_pass = actual_score >= threshold

        if expected_pass == actual_pass:
            # Agree on outcome, score based on proximity
            diff = abs(expected_score - actual_score)
            return max(0.5, 1.0 - diff / 100)
        else:
            # Disagree on outcome - significant penalty
            return 0.2

    return metric


# =============================================================================
# Phase 2A Metrics: Agent-Specific Metrics
# =============================================================================

def issue_severity_match(expected: str, actual: str) -> float:
    """
    Match issue severity classifications for code-reviewer and security-auditor.

    Uses semantic comparison via robust extractors that handle both:
    - Simple format: "Critical: 2"
    - Rich markdown: "### Critical Issues\n1. SQL injection..."

    Scoring components (weighted):
    - 40%: Severity count accuracy (exact match on critical/high counts)
    - 40%: Issue detection (F1 on issue types/categories)
    - 20%: File coverage (mentioned same files)

    Args:
        expected: Expected review output
        actual: Actual review output

    Returns:
        Score between 0.0 and 1.0
    """
    from .extractors import extract_severity_details

    expected_details = extract_severity_details(expected)
    actual_details = extract_severity_details(actual)

    # If no severity data in either, fall back to string similarity
    if expected_details.total_count == 0 and actual_details.total_count == 0:
        return score_similarity(expected, actual)

    score = 0.0

    # Component 1: Severity count accuracy (40%)
    count_score = _compare_severity_counts(expected_details, actual_details)
    score += 0.4 * count_score

    # Component 2: Issue detection F1 (40%)
    issue_score = _compute_issue_f1(expected_details, actual_details)
    score += 0.4 * issue_score

    # Component 3: File coverage (20%)
    file_score = _compute_file_coverage(expected_details, actual_details)
    score += 0.2 * file_score

    return min(1.0, score)


def _compare_severity_counts(expected, actual) -> float:
    """Compare severity counts with weighted importance."""
    weights = {
        'critical': 3.0,
        'high': 2.0,
        'medium': 1.0,
        'low': 0.5,
        'warning': 1.0,
    }

    total_score = 0.0
    total_weight = 0.0

    comparisons = [
        ('critical', expected.critical_count, actual.critical_count),
        ('high', expected.high_count, actual.high_count),
        ('medium', expected.medium_count, actual.medium_count),
        ('low', expected.low_count, actual.low_count),
        ('warning', expected.warning_count, actual.warning_count),
    ]

    for level, exp_count, act_count in comparisons:
        if exp_count > 0 or act_count > 0:
            weight = weights[level]
            total_weight += weight

            if exp_count == act_count:
                total_score += weight
            elif abs(exp_count - act_count) == 1:
                total_score += weight * 0.7  # Off by one
            else:
                # More significant difference
                max_count = max(exp_count, act_count, 1)
                diff_ratio = abs(exp_count - act_count) / max_count
                total_score += weight * max(0, 1 - diff_ratio)

    if total_weight == 0:
        return 0.5

    return total_score / total_weight


def _compute_issue_f1(expected, actual) -> float:
    """Compute F1 score on issue detection based on titles/descriptions."""
    if not expected.issues and not actual.issues:
        return 0.5  # Neither has issues

    if not expected.issues or not actual.issues:
        return 0.0  # One has issues, other doesn't

    # Extract issue keywords from titles
    def get_keywords(issues):
        keywords = set()
        for issue in issues:
            title = issue.get('title', '').lower()
            # Extract significant words
            for word in re.findall(r'\b[a-z]{4,}\b', title):
                keywords.add(word)
        return keywords

    expected_kw = get_keywords(expected.issues)
    actual_kw = get_keywords(actual.issues)

    if not expected_kw:
        return 0.5

    # F1 score
    intersection = len(expected_kw & actual_kw)
    precision = intersection / len(actual_kw) if actual_kw else 0
    recall = intersection / len(expected_kw)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def _compute_file_coverage(expected, actual) -> float:
    """Compute how many expected files were mentioned in actual."""
    def get_files(details):
        files = set()
        for issue in details.issues:
            if issue.get('file'):
                # Normalize: just the filename, not full path
                filename = issue['file'].split('/')[-1]
                files.add(filename.lower())
        return files

    expected_files = get_files(expected)
    actual_files = get_files(actual)

    if not expected_files:
        return 0.5  # No files expected

    if not actual_files:
        return 0.0  # Expected files but none found

    coverage = len(expected_files & actual_files) / len(expected_files)
    return coverage


def test_coverage_score(expected: str, actual: str) -> float:
    """
    Evaluate test suite quality for test-writer agent.

    Uses semantic comparison via robust extractors that handle both:
    - Simple format: "Total Tests: 15"
    - Rich code: describe/it blocks, test functions

    Scoring components (weighted):
    - 30%: Test count accuracy
    - 30%: Category coverage (happy path, edge cases, error cases)
    - 20%: Structure match (describe/it/expect)
    - 20%: Test name overlap

    Args:
        expected: Expected test output
        actual: Actual test output

    Returns:
        Score between 0.0 and 1.0
    """
    from .extractors import extract_test_coverage_details

    expected_details = extract_test_coverage_details(expected)
    actual_details = extract_test_coverage_details(actual)

    score = 0.0
    weights = {
        'test_count': 0.3,
        'categories': 0.3,
        'structure': 0.2,
        'test_names': 0.2,
    }

    # Component 1: Test count comparison (30%)
    if expected_details.total_tests > 0:
        count_diff = abs(expected_details.total_tests - actual_details.total_tests)
        count_ratio = count_diff / max(expected_details.total_tests, 1)
        score += weights['test_count'] * max(0, 1 - count_ratio)
    else:
        score += weights['test_count'] * (0.5 if actual_details.total_tests > 0 else 0)

    # Component 2: Category coverage (30%)
    expected_cats = set()
    actual_cats = set()
    if expected_details.happy_path_count > 0:
        expected_cats.add('happy_path')
    if expected_details.edge_case_count > 0:
        expected_cats.add('edge_cases')
    if expected_details.error_case_count > 0:
        expected_cats.add('error_cases')

    if actual_details.happy_path_count > 0:
        actual_cats.add('happy_path')
    if actual_details.edge_case_count > 0:
        actual_cats.add('edge_cases')
    if actual_details.error_case_count > 0:
        actual_cats.add('error_cases')

    if expected_cats:
        overlap = len(expected_cats & actual_cats)
        score += weights['categories'] * (overlap / len(expected_cats))
    else:
        score += weights['categories'] * (0.5 if actual_cats else 0)

    # Component 3: Structure match (20%)
    structure_score = 0
    if expected_details.has_describe == actual_details.has_describe:
        structure_score += 0.33
    if expected_details.has_it == actual_details.has_it:
        structure_score += 0.33
    if expected_details.has_expect == actual_details.has_expect:
        structure_score += 0.34
    score += weights['structure'] * structure_score

    # Component 4: Test name overlap (20%)
    if expected_details.tests and actual_details.tests:
        expected_names = {t.get('name', '').lower() for t in expected_details.tests}
        actual_names = {t.get('name', '').lower() for t in actual_details.tests}

        # Remove empty names
        expected_names.discard('')
        actual_names.discard('')

        if expected_names:
            overlap = len(expected_names & actual_names)
            score += weights['test_names'] * (overlap / len(expected_names))
        else:
            score += weights['test_names'] * 0.5
    else:
        score += weights['test_names'] * 0.5

    return min(1.0, score)


def complexity_classification(expected: str, actual: str) -> float:
    """
    Evaluate performance analysis accuracy for performance-analyzer agent.

    Checks:
    - Issue classification (Critical/High/Medium/Low)
    - Pattern identification (N+1, O(n^2), etc.)
    - Improvement percentage estimates

    Args:
        expected: Expected analysis output
        actual: Actual analysis output

    Returns:
        Score between 0.0 and 1.0
    """
    def extract_analysis(text: str) -> Dict[str, Any]:
        analysis = {
            'patterns': set(),
            'impact_level': None,
            'improvement_pct': None,
            'has_fix': False,
            'metrics_mentioned': False,
        }

        text_lower = text.lower()

        # Extract performance patterns
        pattern_keywords = {
            'n_plus_one': ['n+1', 'n + 1', 'n+1 query', 'sequential fetch'],
            'nested_loop': ['o(n^2)', 'o(n²)', 'nested loop', 'quadratic'],
            'sequential_io': ['sequential', 'parallel possible', 'promise.all'],
            'memory': ['memory leak', 'memory bloat', 'memory usage'],
            'string_concat': ['string concatenation', 'string concat'],
            'client_side': ['client-side', 'filtering in application'],
            'cache': ['cache', 'stampede', 'caching'],
        }

        for pattern, keywords in pattern_keywords.items():
            if any(kw in text_lower for kw in keywords):
                analysis['patterns'].add(pattern)

        # Extract impact level
        impact_patterns = [
            (r'impact[:\s]*(high|critical)', 'high'),
            (r'impact[:\s]*(medium)', 'medium'),
            (r'impact[:\s]*(low)', 'low'),
            (r'critical\s*issues?[:\s]*(\d+)', 'high'),
        ]
        for pattern, level in impact_patterns:
            if re.search(pattern, text_lower):
                analysis['impact_level'] = level
                break

        # Extract improvement percentage
        pct_match = re.search(r'(\d+)\s*%\s*(?:reduction|improvement|saved)', text_lower)
        if pct_match:
            analysis['improvement_pct'] = int(pct_match.group(1))

        # Check for recommended fix
        analysis['has_fix'] = any(kw in text_lower for kw in
            ['recommended fix', 'fix:', 'solution:', 'after:', 'improved:'])

        # Check for metrics
        analysis['metrics_mentioned'] = any(kw in text_lower for kw in
            ['metrics to monitor', 'response time', 'memory usage', 'target'])

        return analysis

    expected_analysis = extract_analysis(expected)
    actual_analysis = extract_analysis(actual)

    score = 0.0
    weights = {
        'patterns': 0.35,
        'impact': 0.2,
        'improvement': 0.15,
        'fix': 0.15,
        'metrics': 0.15,
    }

    # Pattern matching
    if expected_analysis['patterns']:
        overlap = len(expected_analysis['patterns'] & actual_analysis['patterns'])
        total = len(expected_analysis['patterns'])
        score += weights['patterns'] * (overlap / total)
    else:
        score += weights['patterns'] * (0.5 if actual_analysis['patterns'] else 0)

    # Impact level
    if expected_analysis['impact_level']:
        if expected_analysis['impact_level'] == actual_analysis['impact_level']:
            score += weights['impact']
        elif actual_analysis['impact_level']:
            # Adjacent levels get partial credit
            levels = ['low', 'medium', 'high']
            if all(l in levels for l in [expected_analysis['impact_level'], actual_analysis['impact_level']]):
                idx_exp = levels.index(expected_analysis['impact_level'])
                idx_act = levels.index(actual_analysis['impact_level'])
                if abs(idx_exp - idx_act) == 1:
                    score += weights['impact'] * 0.5
    else:
        score += weights['impact'] * 0.5

    # Improvement percentage
    if expected_analysis['improvement_pct'] and actual_analysis['improvement_pct']:
        pct_diff = abs(expected_analysis['improvement_pct'] - actual_analysis['improvement_pct'])
        score += weights['improvement'] * max(0, 1 - pct_diff / 100)
    elif expected_analysis['improvement_pct'] or actual_analysis['improvement_pct']:
        score += weights['improvement'] * 0.3
    else:
        score += weights['improvement'] * 0.5

    # Has fix
    if expected_analysis['has_fix'] == actual_analysis['has_fix']:
        score += weights['fix']
    elif actual_analysis['has_fix']:
        score += weights['fix'] * 0.5

    # Metrics mentioned
    if expected_analysis['metrics_mentioned'] == actual_analysis['metrics_mentioned']:
        score += weights['metrics']
    elif actual_analysis['metrics_mentioned']:
        score += weights['metrics'] * 0.5

    return min(1.0, score)


def security_cwe_match(expected: str, actual: str) -> float:
    """
    Specialized metric for security-auditor CWE classification.

    Compares CWE identifiers and vulnerability types.

    Args:
        expected: Expected security audit output
        actual: Actual security audit output

    Returns:
        Score between 0.0 and 1.0
    """
    def extract_cwes(text: str) -> set:
        # Extract CWE-XXX patterns
        cwes = set(re.findall(r'CWE-(\d+)', text, re.IGNORECASE))
        return cwes

    def extract_vuln_types(text: str) -> set:
        text_lower = text.lower()
        vuln_keywords = {
            'sql_injection': ['sql injection', 'sqli'],
            'xss': ['xss', 'cross-site scripting', 'script injection'],
            'command_injection': ['command injection', 'os command', 'shell injection'],
            'path_traversal': ['path traversal', 'directory traversal', 'lfi'],
            'auth_bypass': ['authentication bypass', 'auth bypass'],
            'hardcoded_creds': ['hardcoded', 'hard-coded', 'credential'],
            'weak_crypto': ['weak hash', 'md5', 'sha1', 'weak encryption'],
            'missing_auth': ['missing authorization', 'broken access'],
            'cors': ['cors', 'cross-origin'],
            'csrf': ['csrf', 'cross-site request'],
        }

        found = set()
        for vuln_type, keywords in vuln_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found.add(vuln_type)
        return found

    expected_cwes = extract_cwes(expected)
    actual_cwes = extract_cwes(actual)
    expected_vulns = extract_vuln_types(expected)
    actual_vulns = extract_vuln_types(actual)

    cwe_score = 0.0
    if expected_cwes:
        overlap = len(expected_cwes & actual_cwes)
        cwe_score = overlap / len(expected_cwes)
    elif actual_cwes:
        cwe_score = 0.3  # Found CWEs when none expected
    else:
        cwe_score = 0.5  # Neither has CWEs

    vuln_score = 0.0
    if expected_vulns:
        overlap = len(expected_vulns & actual_vulns)
        vuln_score = overlap / len(expected_vulns)
    elif actual_vulns:
        vuln_score = 0.3
    else:
        vuln_score = 0.5

    # Also check severity match
    severity_score = issue_severity_match(expected, actual)

    # Combine: CWE 30%, Vuln types 30%, Severity 40%
    return 0.3 * cwe_score + 0.3 * vuln_score + 0.4 * severity_score


# =============================================================================
# Phase 2B Metrics: Skill-Specific Metrics
# =============================================================================

def routing_accuracy(expected: str, actual: str) -> float:
    """
    Evaluate tool routing accuracy for mcp-search-framework skill.

    Compares selected tool against expected tool using robust extraction
    that handles verbose Claude outputs.

    Args:
        expected: Expected tool name/output
        actual: Actual tool name/output (may be verbose)

    Returns:
        1.0 if correct tool, 0.6 for same family, 0.3 for related, 0.0 otherwise
    """
    # Import robust extractors
    from .extractors import (
        extract_tool_from_verbose,
        normalize_tool_name,
        get_tool_family,
        RELATED_FAMILIES,
    )

    # Extract tools from both expected and actual
    expected_tool = normalize_tool_name(expected)
    if not expected_tool:
        # Try extraction if not a simple tool name
        expected_tool = extract_tool_from_verbose(expected)

    actual_tool = extract_tool_from_verbose(actual)

    if not expected_tool:
        # Can't determine expected tool, fall back to similarity
        return score_similarity(expected, actual)

    if not actual_tool:
        # No tool extracted from verbose output
        return 0.0

    # Exact match
    if expected_tool == actual_tool:
        return 1.0

    # Same family (e.g., both Brave tools)
    expected_family = get_tool_family(expected_tool)
    actual_family = get_tool_family(actual_tool)

    if expected_family and expected_family == actual_family:
        return 0.6

    # Related families (e.g., Brave and Exa are both web search)
    if expected_family and actual_family:
        related = RELATED_FAMILIES.get(expected_family, [])
        if actual_family in related:
            return 0.3

    return 0.0


def binary_decision_match(expected: str, actual: str) -> float:
    """
    Evaluate binary decision accuracy for mgrep-guide skill.

    Compares whether output recommends Grep vs mgrep correctly.
    Uses robust extraction that handles verbose outputs.

    Args:
        expected: Expected decision
        actual: Actual decision (may be verbose)

    Returns:
        1.0 if correct decision, 0.0 otherwise
    """
    from .extractors import extract_binary_decision

    # Extract decisions using robust extractor
    expected_decision = extract_binary_decision(expected)
    actual_decision = extract_binary_decision(actual)

    if not expected_decision:
        # Can't determine expected decision, fall back to similarity
        return score_similarity(expected, actual)

    if not actual_decision:
        # No decision extracted from verbose output
        return 0.0

    if expected_decision == actual_decision:
        return 1.0

    return 0.0


# =============================================================================
# Phase 5 Metrics: Regularization
# =============================================================================

def length_regularized_score(
    base_score: float,
    prompt_tokens: int,
    baseline_tokens: int = 2000,
    penalty_factor: float = 0.0001,
) -> float:
    """
    Apply length penalty to encourage concise prompts.

    Penalizes prompts that exceed a baseline token count to prevent
    verbose, overfit prompts.

    Args:
        base_score: The original evaluation score (0.0-1.0)
        prompt_tokens: Number of tokens in the prompt
        baseline_tokens: Token count threshold (default: 2000)
        penalty_factor: Penalty per excess token (default: 0.0001)

    Returns:
        Length-penalized score, floored at 0.0
    """
    excess = max(0, prompt_tokens - baseline_tokens)
    penalty = excess * penalty_factor
    return max(0.0, base_score - penalty)


def tool_tier_classification(expected: str, actual: str) -> float:
    """
    Evaluate tool tier classification for advanced-tool-use skill.

    Classifies tools into Core/Specialized/Deferred tiers.
    Uses robust extraction that handles verbose outputs.

    Args:
        expected: Expected classification
        actual: Actual classification (may be verbose)

    Returns:
        Score between 0.0 and 1.0
    """
    from .extractors import extract_tier

    # Extract tiers using robust extractor
    expected_tier = extract_tier(expected)
    actual_tier = extract_tier(actual)

    if not expected_tier:
        # Can't determine expected tier, fall back to similarity
        return score_similarity(expected, actual)

    if not actual_tier:
        # No tier extracted from verbose output
        return 0.0

    if expected_tier == actual_tier:
        return 1.0

    # Adjacent tiers get partial credit
    tier_order = ['core', 'specialized', 'deferred']
    if expected_tier in tier_order and actual_tier in tier_order:
        idx_exp = tier_order.index(expected_tier)
        idx_act = tier_order.index(actual_tier)
        if abs(idx_exp - idx_act) == 1:
            return 0.5

    return 0.2


# =============================================================================
# Phase 3 Metrics: High-Priority Agent Metrics
# =============================================================================

def root_cause_match(expected: str, actual: str) -> float:
    """
    Evaluate debugging analysis for debugger agent.

    Compares:
    - Root cause category identification (40%)
    - Fix type appropriateness (30%)
    - Verification steps presence (15%)
    - Code solution quality (15%)

    Args:
        expected: Expected debugging analysis
        actual: Actual debugging analysis

    Returns:
        Score between 0.0 and 1.0
    """
    def extract_root_cause(text: str) -> Optional[str]:
        """Extract root cause category from debugging output."""
        text_lower = text.lower()

        # Map common patterns to root cause categories
        root_cause_patterns = {
            'context_outside_provider': ['outside provider', 'not wrapped', 'usecontext.*undefined', 'provider missing'],
            'floating_point_precision': ['floating point', 'precision', 'tobecloseto', 'decimal.js', 'rounding'],
            'memory_storage_overflow': ['memory', 'enomem', 'memorystorage', 'buffer', 'heap'],
            'event_listener_leak': ['event listener', 'cleanup', 'removeEventListener', 'multiple times'],
            'docker_networking': ['localhost', 'docker', 'container', 'host.docker.internal', 'service name'],
            'hydration_mismatch': ['hydration', 'server', 'client', 'ssr', 'time-dependent'],
            'undefined_prop': ['undefined', 'cannot read', 'null', 'optional chaining', 'default value'],
            'async_race_condition': ['race condition', 'async', 'timing', 'stale closure'],
            'import_export_mismatch': ['import', 'export', 'module', 'named export', 'default export'],
            'type_mismatch': ['type error', 'type mismatch', 'typescript', 'generic'],
            'missing_dependency': ['dependency', 'useeffect', 'exhaustive-deps', 'stale'],
            'authentication_error': ['auth', '401', '403', 'token', 'credential'],
            'parsing_error': ['parse', 'json', 'syntax', 'malformed'],
        }

        for category, keywords in root_cause_patterns.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return None

    def extract_fix_type(text: str) -> Optional[str]:
        """Extract fix type from debugging output."""
        text_lower = text.lower()

        fix_type_patterns = {
            'context_guard': ['throw new error', 'if.*undefined', 'guard', 'check context'],
            'defensive_coding': ['default', 'optional', '??', '||', 'fallback'],
            'disk_storage': ['diskstorage', 'createreadstream', 'stream', 'file system'],
            'cleanup_listener': ['cleanup', 'return.*remove', 'useeffect.*return'],
            'environment_config': ['environment', 'env', 'config', 'os.environ'],
            'client_only': ['useeffect', 'mounted', 'use client', 'suppresshydrationwarning'],
            'precision_handling': ['tobecloseto', 'math.round', 'decimal', 'cents'],
            'error_boundary': ['errorboundary', 'catch', 'try.*catch'],
            'type_annotation': ['type', 'interface', 'generic', 'as const'],
            'dependency_array': ['dependency array', 'usecallback', 'usememo'],
        }

        for fix_type, keywords in fix_type_patterns.items():
            if any(kw in text_lower for kw in keywords):
                return fix_type
        return None

    def has_verification_steps(text: str) -> bool:
        """Check if output includes verification steps."""
        text_lower = text.lower()
        verification_indicators = [
            'verification', 'verify', 'test', 'confirm', 'check',
            '- [ ]', '- [x]', 'ensure', 'validate'
        ]
        return any(ind in text_lower for ind in verification_indicators)

    def has_code_solution(text: str) -> bool:
        """Check if output includes code solution."""
        return '```' in text and any(kw in text.lower() for kw in
            ['function', 'const', 'class', 'import', 'async', 'return', 'export'])

    score = 0.0

    # Component 1: Root cause match (40%)
    expected_cause = extract_root_cause(expected)
    actual_cause = extract_root_cause(actual)

    if expected_cause and actual_cause:
        if expected_cause == actual_cause:
            score += 0.4
        else:
            # Partial credit for related categories
            related_groups = [
                {'context_outside_provider', 'undefined_prop'},
                {'memory_storage_overflow', 'event_listener_leak'},
                {'async_race_condition', 'missing_dependency'},
            ]
            for group in related_groups:
                if expected_cause in group and actual_cause in group:
                    score += 0.2
                    break
    elif expected_cause is None and actual_cause is None:
        score += 0.2  # Neither identified - baseline

    # Component 2: Fix type match (30%)
    expected_fix = extract_fix_type(expected)
    actual_fix = extract_fix_type(actual)

    if expected_fix and actual_fix:
        if expected_fix == actual_fix:
            score += 0.3
        else:
            score += 0.1  # Partial credit for attempting fix
    elif actual_fix:
        score += 0.15  # Some fix suggested even if different

    # Component 3: Verification steps (15%)
    if has_verification_steps(expected) == has_verification_steps(actual):
        score += 0.15
    elif has_verification_steps(actual):
        score += 0.07  # Partial credit for including verification

    # Component 4: Code solution (15%)
    if has_code_solution(expected) and has_code_solution(actual):
        score += 0.15
    elif has_code_solution(actual):
        score += 0.07

    return min(1.0, score)


def pr_quality_match(expected: str, actual: str) -> float:
    """
    Evaluate PR description quality for pr-preparer agent.

    Revised weights (v3) with expanded header detection and content coherence:
    - Change type classification (15%) - uses equivalence groups
    - Risk level assessment (15%) - uses adjacent-level credit
    - Required sections presence (30%) - expanded header aliases, content fallback
    - Content coherence (10%) - file/function overlap between expected and actual
    - Testing checklist (20%) - unchanged
    - Quality indicators (10%) - migration notes, screenshots, etc.

    Args:
        expected: Expected PR description
        actual: Actual PR description

    Returns:
        Score between 0.0 and 1.0
    """
    def extract_change_type(text: str) -> Optional[str]:
        """Extract change type from PR description."""
        text_lower = text.lower()

        change_types = {
            'feature': ['new feature', 'feature', 'adds', 'implement'],
            'enhancement': ['enhancement', 'improves', 'enhance', 'update'],
            'bugfix': ['bug fix', 'bugfix', 'fix', 'fixes', 'patch'],
            'security': ['security', 'vulnerability', 'auth', 'cors'],
            'refactor': ['refactor', 'cleanup', 'reorganize'],
            'docs': ['documentation', 'docs', 'readme'],
            'test': ['test', 'testing', 'coverage'],
            'migration': ['migration', 'database', 'schema'],
            'config': ['config', 'configuration', 'ci', 'build'],
        }

        # Check type section first
        type_pattern = r'##?\s*type[:\s]*([^\n]+)'
        match = re.search(type_pattern, text_lower)
        if match:
            type_text = match.group(1).strip()
            for change_type, keywords in change_types.items():
                if any(kw in type_text for kw in keywords):
                    return change_type

        # Fall back to content analysis
        for change_type, keywords in change_types.items():
            if any(kw in text_lower for kw in keywords):
                return change_type
        return None

    def extract_risk_level(text: str) -> Optional[str]:
        """Extract risk level from PR description."""
        text_lower = text.lower()

        # Look for explicit risk section - handle both inline and multiline formats
        risk_patterns = [
            r'##?\s*risk[^:\n]*:\s*([^\n]+)',  # Inline with colon
            r'##?\s*risk[^\n]*\n\s*([^\n#]+)',  # Value on next line
        ]
        for pattern in risk_patterns:
            match = re.search(pattern, text_lower)
            if match:
                risk_text = match.group(1).strip()
                if 'high' in risk_text:
                    return 'high'
                elif 'medium' in risk_text:
                    return 'medium'
                elif 'low' in risk_text:
                    return 'low'

        # Infer from content
        high_risk_indicators = ['database', 'migration', 'auth', 'security', 'payment', 'breaking']
        medium_risk_indicators = ['api', 'multiple files', 'refactor', 'behavior change']
        low_risk_indicators = ['new file', 'additive', 'no existing', 'utility', 'docs']

        if any(ind in text_lower for ind in high_risk_indicators):
            return 'high'
        elif any(ind in text_lower for ind in medium_risk_indicators):
            return 'medium'
        elif any(ind in text_lower for ind in low_risk_indicators):
            return 'low'
        return None

    def check_required_sections(text: str) -> Dict[str, bool]:
        """Check presence of required PR sections with content validation."""
        text_lower = text.lower()

        sections = {
            'summary': bool(re.search(r'##?\s*(summary|overview|description|what|about)', text_lower)),
            'changes': bool(re.search(r'##?\s*(changes|what.?changed|modifications|diff|files)', text_lower)),
            'type': bool(re.search(r'##?\s*(type|category|kind|classification)', text_lower)),
            'risk': bool(re.search(r'##?\s*(risk|impact|safety|concern)', text_lower)),
            'testing': bool(re.search(r'##?\s*(test|testing|verification|how.?to.?test|qa)', text_lower)),
        }
        return sections

    def check_section_content(text: str) -> Dict[str, float]:
        """Check if sections have meaningful content (not just headers)."""
        scores = {}

        # Summary should have at least 20 chars of content
        summary_match = re.search(r'##?\s*(?:summary|overview|description|what|about)[:\s]*\n(.+?)(?=##|\Z)', text, re.IGNORECASE | re.DOTALL)
        if summary_match:
            content = summary_match.group(1).strip()
            scores['summary'] = min(1.0, len(content) / 50)  # Full credit at 50+ chars
        else:
            scores['summary'] = 0.0

        # Changes should have bullet points or numbered items
        changes_match = re.search(r'##?\s*(?:changes|what.?changed|modifications|diff|files)[:\s]*\n(.+?)(?=##|\Z)', text, re.IGNORECASE | re.DOTALL)
        if changes_match:
            content = changes_match.group(1)
            bullet_count = len(re.findall(r'^\s*[-*]\s', content, re.MULTILINE))
            numbered_count = len(re.findall(r'^\s*\d+\.\s', content, re.MULTILINE))
            scores['changes'] = min(1.0, (bullet_count + numbered_count) / 3)  # Full credit at 3+ items
        else:
            scores['changes'] = 0.0

        return scores

    def count_test_items(text: str) -> int:
        """Count testing checklist items."""
        test_items = re.findall(r'-\s*\[[ x]\]', text)
        return len(test_items)

    def check_quality_indicators(text: str) -> float:
        """Check for quality indicators in PR description."""
        text_lower = text.lower()
        indicators_found = 0
        total_indicators = 5

        # Check for screenshots/images
        if re.search(r'!\[.*?\]\(.*?\)|<img|screenshot', text_lower):
            indicators_found += 1

        # Check for migration notes (for structural changes)
        if re.search(r'migration|backward.?compat|breaking.?change', text_lower):
            indicators_found += 1

        # Check for links to related issues/PRs
        if re.search(r'#\d+|fixes|closes|related', text_lower):
            indicators_found += 1

        # Check for deployment notes
        if re.search(r'deploy|rollback|feature.?flag', text_lower):
            indicators_found += 1

        # Check for documentation updates mentioned
        if re.search(r'readme|documentation|docs.?updated', text_lower):
            indicators_found += 1

        return indicators_found / total_indicators

    score = 0.0

    # Component 1: Change type match (15%) - uses semantic equivalence
    expected_type = extract_change_type(expected)
    actual_type = extract_change_type(actual)

    change_type_score = _score_categorical_match(
        expected_type,
        actual_type,
        CHANGE_TYPE_EQUIVALENCES,
        CHANGE_TYPE_RELATED,
    )
    score += 0.15 * change_type_score

    # Component 2: Risk level match (15%) - uses adjacent-level credit
    expected_risk = extract_risk_level(expected)
    actual_risk = extract_risk_level(actual)

    if expected_risk and actual_risk:
        if expected_risk == actual_risk:
            score += 0.15
        else:
            # Adjacent risk levels get 70% credit
            risk_order = ['low', 'medium', 'high']
            if expected_risk in risk_order and actual_risk in risk_order:
                idx_exp = risk_order.index(expected_risk)
                idx_act = risk_order.index(actual_risk)
                if abs(idx_exp - idx_act) == 1:
                    score += 0.15 * 0.7  # 70% for adjacent levels
                else:
                    score += 0.15 * 0.3  # 30% for opposite ends
    elif expected_risk is None:
        score += 0.15 * 0.5  # No expected risk to match

    # Component 3: Required sections (30%) - reduced from 40%
    expected_sections = check_required_sections(expected)
    actual_sections = check_required_sections(actual)
    content_scores = check_section_content(actual)

    section_score = 0.0
    section_count = 0
    sections_found = sum(1 for v in actual_sections.values() if v)
    for section, expected_present in expected_sections.items():
        if expected_present:
            section_count += 1
            if actual_sections.get(section, False):
                # Section header present, add content quality bonus
                content_quality = content_scores.get(section, 0.5)
                section_score += 0.7 + (0.3 * content_quality)  # 70% for header, 30% for content
            # Missing section = 0
        else:
            section_count += 1
            section_score += 0.5  # Section not required, neutral score

    # Content-based fallback: if no headers match but text is substantive,
    # give partial credit (indicates good content without formal structure)
    if sections_found == 0 and len(actual) > 200:
        section_score = max(section_score, section_count * 0.3)

    if section_count > 0:
        score += 0.30 * (section_score / section_count)
    else:
        score += 0.15  # No sections to check

    # Component 3b: Content coherence (10%) - NEW
    # Check if PR description references files/functions from expected output
    expected_paths = set(re.findall(r'`([^`]*(?:\.\w+|/\w+)[^`]*)`', expected))
    actual_paths = set(re.findall(r'`([^`]*(?:\.\w+|/\w+)[^`]*)`', actual))
    if expected_paths:
        overlap = len(expected_paths & actual_paths)
        coherence_score = min(1.0, overlap / max(1, len(expected_paths) * 0.5))
    else:
        # Fallback: use SequenceMatcher on extracted nouns/identifiers
        coherence_score = SequenceMatcher(None, expected[:500].lower(), actual[:500].lower()).ratio()
    score += 0.10 * coherence_score

    # Component 4: Testing checklist (20%)
    expected_tests = count_test_items(expected)
    actual_tests = count_test_items(actual)

    if expected_tests > 0:
        if actual_tests >= expected_tests:
            score += 0.20
        elif actual_tests > 0:
            score += 0.20 * (actual_tests / expected_tests)
    elif actual_tests > 0:
        score += 0.15  # Added tests even when not explicitly required
    else:
        score += 0.10  # Neither has test items

    # Component 5: Quality indicators (10%) - NEW
    quality_score = check_quality_indicators(actual)
    score += 0.10 * quality_score

    return min(1.0, score)


def implementation_completeness(expected: str, actual: str) -> float:
    """
    Evaluate feature implementation for feature-implementer agent.

    Compares:
    - Acceptance criteria coverage (35%)
    - Files identified (20%)
    - Code changes included (25%)
    - Tests included (20%)

    Args:
        expected: Expected implementation plan
        actual: Actual implementation plan

    Returns:
        Score between 0.0 and 1.0
    """
    def extract_acceptance_criteria(text: str) -> List[str]:
        """Extract acceptance criteria items."""
        criteria = []

        # Look for AC section
        ac_pattern = r'(?:acceptance criteria|checklist)[:\s]*\n((?:[-*\d.]\s*\[?[x ]?\]?[^\n]+\n?)+)'
        match = re.search(ac_pattern, text, re.IGNORECASE)
        if match:
            lines = match.group(1).strip().split('\n')
            for line in lines:
                cleaned = re.sub(r'^[-*\d.]\s*\[?[x ]?\]?\s*', '', line).strip()
                if cleaned:
                    criteria.append(cleaned.lower())

        # Also check for [x] AC items anywhere
        checkbox_items = re.findall(r'\[[ x]\]\s*ac\d*[:\s]*([^\n]+)', text, re.IGNORECASE)
        criteria.extend([item.lower().strip() for item in checkbox_items])

        return criteria

    def extract_files(text: str) -> List[str]:
        """Extract file paths from implementation plan."""
        # Match file paths like src/components/Button.tsx
        file_pattern = r'(?:^|\s|`)((?:src|lib|app|pages|components|services|utils|hooks)[/\\][^\s`\n]+\.[a-z]+)'
        files = re.findall(file_pattern, text, re.IGNORECASE)
        return [f.lower() for f in files]

    def has_code_changes(text: str) -> bool:
        """Check if implementation includes code snippets."""
        return '```' in text and len(re.findall(r'```[a-z]*\n', text)) > 0

    def has_tests(text: str) -> bool:
        """Check if implementation includes test code or test plan."""
        text_lower = text.lower()
        test_indicators = [
            'it(', 'test(', 'describe(', 'expect(',
            'unit test', 'integration test',
            '```.*test', 'test.ts', '.test.', '.spec.'
        ]
        return any(ind in text_lower for ind in test_indicators)

    def calculate_ac_coverage(expected_ac: List[str], actual_text: str) -> float:
        """Calculate how many ACs are addressed in actual output."""
        if not expected_ac:
            return 0.5  # No ACs to check

        actual_lower = actual_text.lower()
        matched = 0

        for ac in expected_ac:
            # Extract key words from AC
            keywords = [w for w in ac.split() if len(w) > 3]
            if keywords:
                # Check if most keywords appear in actual
                found = sum(1 for kw in keywords if kw in actual_lower)
                if found / len(keywords) > 0.5:
                    matched += 1

        return matched / len(expected_ac)

    score = 0.0

    # Component 1: AC coverage (35%)
    expected_ac = extract_acceptance_criteria(expected)
    if expected_ac:
        ac_coverage = calculate_ac_coverage(expected_ac, actual)
        score += 0.35 * ac_coverage
    else:
        # No explicit ACs, check general completeness
        score += 0.17

    # Component 2: Files identified (20%)
    expected_files = extract_files(expected)
    actual_files = extract_files(actual)

    if expected_files:
        if actual_files:
            # Check overlap
            expected_set = set(expected_files)
            actual_set = set(actual_files)
            overlap = len(expected_set & actual_set)
            score += 0.2 * (overlap / len(expected_set))
        # No files identified = 0 for this component
    else:
        # No files expected
        score += 0.1 if actual_files else 0.05

    # Component 3: Code changes (25%)
    if has_code_changes(expected):
        if has_code_changes(actual):
            score += 0.25
        # No code = 0 for this component
    else:
        score += 0.12 if has_code_changes(actual) else 0.05

    # Component 4: Tests (20%)
    if has_tests(expected):
        if has_tests(actual):
            score += 0.2
    else:
        score += 0.1 if has_tests(actual) else 0.05

    return min(1.0, score)


# =============================================================================
# Plan Quality Metrics
# =============================================================================

def _extract_plan_files(text: str) -> List[str]:
    """Extract file paths from a plan output."""
    # Match common file path patterns
    patterns = [
        r'(?:^|\s|`)((?:src|lib|app|pages|components|services|utils|hooks|config|test|tests|scripts|public|api)[/\\][^\s`\n,)]+\.[a-z]+)',
        r'(?:^|\s|`)((?:\.?/)?[\w.-]+/[\w.-]+(?:/[\w.-]+)*\.[a-z]{1,4})',
        r'`([^`\s]+\.[a-z]{1,4})`',
    ]
    files = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            path = match.group(1).strip().lower()
            # Filter out obviously non-file matches
            if not any(path.endswith(ext) for ext in ['.com', '.org', '.net', '.io']):
                files.add(path)
    return list(files)


def _extract_edge_cases(text: str) -> int:
    """Count edge case / failure mode mentions in plan text."""
    text_lower = text.lower()
    edge_case_patterns = [
        r'edge\s*case',
        r'failure\s*mode',
        r'what\s+if',
        r'corner\s*case',
        r'error\s+(?:handling|case|condition|scenario)',
        r'fallback',
        r'empty\s+(?:state|list|array|input|result)',
        r'null\s+(?:check|case|value)',
        r'boundary\s+(?:case|condition)',
        r'race\s+condition',
        r'timeout',
        r'invalid\s+(?:input|data|state)',
        r'concurrent',
        r'(?:no|zero|missing)\s+(?:data|results|items)',
    ]
    count = 0
    for pattern in edge_case_patterns:
        count += len(re.findall(pattern, text_lower))
    return count


def _score_step_specificity(text: str) -> float:
    """
    Parse plan steps and score each for specificity.

    A specific step has: file path + what changes + why.
    A vague step is a single sentence without file references.
    """
    # Find numbered or bulleted steps
    step_patterns = [
        r'^\s*\d+\.\s+(.+?)(?=\n\s*\d+\.|\n\s*$|\Z)',
        r'^\s*[-*]\s+(?:\*\*)?Step\s*\d*[:\s]*(?:\*\*)?\s*(.+?)(?=\n\s*[-*]\s+(?:\*\*)?Step|\n\s*$|\Z)',
        r'^\s*###\s+Step\s*\d*[:\s]*(.+?)(?=\n\s*###|\Z)',
    ]

    steps = []
    for pattern in step_patterns:
        found = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if found:
            steps = found
            break

    if not steps:
        # Try splitting by numbered lines
        lines = text.split('\n')
        for line in lines:
            stripped = line.strip()
            if re.match(r'^\d+[\.\)]\s+', stripped):
                steps.append(stripped)

    if not steps:
        return 0.3  # No discernible steps

    step_scores = []
    for step in steps:
        score = 0.0
        step_lower = step.lower()

        # Has file path reference?
        has_file = bool(re.search(r'[\w/\\]+\.[a-z]{1,4}', step))
        if has_file:
            score += 0.4

        # Has what changes (action verb)?
        action_verbs = ['add', 'create', 'modify', 'update', 'remove', 'delete',
                       'refactor', 'rename', 'move', 'extract', 'implement',
                       'configure', 'install', 'migrate', 'wrap', 'convert']
        has_action = any(verb in step_lower for verb in action_verbs)
        if has_action:
            score += 0.3

        # Has why / rationale?
        rationale_signals = ['because', 'since', 'to ensure', 'so that', 'in order to',
                            'this allows', 'this enables', 'for', 'needed for',
                            'required by', 'to support', 'to handle', 'to fix']
        has_rationale = any(signal in step_lower for signal in rationale_signals)
        if has_rationale:
            score += 0.3

        # Penalize very short steps
        if len(step.strip()) < 30:
            score *= 0.5

        step_scores.append(min(1.0, score))

    return sum(step_scores) / len(step_scores) if step_scores else 0.0


def _detect_dependency_tracing(text: str) -> float:
    """Check for upstream/downstream dependency analysis in plan."""
    text_lower = text.lower()
    dependency_signals = [
        'import', 'imports', 'imported by', 'depends on',
        'upstream', 'downstream', 'callers', 'called by',
        'used by', 'uses', 'references', 'referenced by',
        'test impact', 'affected tests', 'test coverage',
        'consumers', 'dependencies', 'dependency chain',
        'breaking change', 'backward compat',
    ]
    found = sum(1 for signal in dependency_signals if signal in text_lower)
    # Normalize: 3+ signals = full credit, 1-2 = partial
    if found >= 3:
        return 1.0
    elif found >= 1:
        return found / 3.0
    return 0.0


def _detect_risk_mentions(text: str) -> float:
    """Check for risk/gotcha/caveat mentions in plan."""
    text_lower = text.lower()
    risk_signals = [
        'risk', 'gotcha', 'caveat', 'careful', 'note that',
        'watch out', 'pitfall', 'danger', 'warning', 'caution',
        'tricky', 'subtle', 'non-obvious', 'important to note',
        'potential issue', 'could break', 'might fail',
        'edge case', 'known issue', 'limitation',
    ]
    found = sum(1 for signal in risk_signals if signal in text_lower)
    # 1+ mention = full credit
    if found >= 1:
        return min(1.0, found / 2.0)
    return 0.0


def plan_quality_match(expected: str, actual: str) -> float:
    """
    Evaluate plan output quality on 5 weighted dimensions.

    Scoring components:
    - File coverage (25%): file paths extracted, compared with expected
    - Edge cases (20%): count of edge case / failure mode mentions
    - Step specificity (25%): each step scored for file path + what + why
    - Dependency tracing (15%): upstream/downstream analysis present
    - Risk identification (15%): risk/gotcha/caveat mentions

    Args:
        expected: Expected plan output (gold standard)
        actual: Actual plan output from model

    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0

    # Component 1: File coverage (25%)
    expected_files = set(_extract_plan_files(expected))
    actual_files = set(_extract_plan_files(actual))

    if expected_files:
        if actual_files:
            # Measure both recall (coverage) and precision
            recall = len(expected_files & actual_files) / len(expected_files)
            # Also credit additional relevant files (don't penalize finding more)
            file_score = recall
            # Bonus for finding extra files (up to 20% bonus)
            if len(actual_files) > len(expected_files):
                file_score = min(1.0, recall + 0.1)
            score += 0.25 * file_score
        # No files in actual = 0
    else:
        # No expected files; give partial credit if actual has any
        score += 0.25 * (0.5 if actual_files else 0.0)

    # Component 2: Edge cases (20%)
    expected_edge_count = _extract_edge_cases(expected)
    actual_edge_count = _extract_edge_cases(actual)

    if expected_edge_count > 0:
        # Score based on meeting or exceeding expected count
        edge_ratio = min(1.0, actual_edge_count / max(expected_edge_count, 1))
        score += 0.20 * edge_ratio
    else:
        # No expected edge cases; threshold is 2+ for full credit
        if actual_edge_count >= 2:
            score += 0.20
        elif actual_edge_count == 1:
            score += 0.10
        # 0 edge cases = 0

    # Component 3: Step specificity (25%)
    actual_specificity = _score_step_specificity(actual)
    expected_specificity = _score_step_specificity(expected)

    if expected_specificity > 0:
        # Compare actual vs expected specificity
        specificity_ratio = min(1.0, actual_specificity / max(expected_specificity, 0.01))
        score += 0.25 * specificity_ratio
    else:
        score += 0.25 * actual_specificity

    # Component 4: Dependency tracing (15%)
    expected_deps = _detect_dependency_tracing(expected)
    actual_deps = _detect_dependency_tracing(actual)

    if expected_deps > 0:
        dep_ratio = min(1.0, actual_deps / max(expected_deps, 0.01))
        score += 0.15 * dep_ratio
    else:
        score += 0.15 * actual_deps

    # Component 5: Risk identification (15%)
    expected_risks = _detect_risk_mentions(expected)
    actual_risks = _detect_risk_mentions(actual)

    if expected_risks > 0:
        risk_ratio = min(1.0, actual_risks / max(expected_risks, 0.01))
        score += 0.15 * risk_ratio
    else:
        score += 0.15 * actual_risks

    return min(1.0, score)


def refactoring_match(expected: str, actual: str) -> float:
    """
    Evaluate refactoring analysis for refactoring-advisor agent.

    Revised weights (v3) with expanded paraphrases and SequenceMatcher fallback:
    - Primary code smell detection (25%) - expanded keywords, SequenceMatcher fallback
    - Refactoring technique selection (25%) - expanded keywords, softened affinity cascade
    - Risk assessment (15%) - unchanged
    - Code example inclusion (15%) - unchanged
    - Rationale quality (20%) - why technique fixes smell

    Args:
        expected: Expected refactoring analysis
        actual: Actual refactoring analysis

    Returns:
        Score between 0.0 and 1.0
    """
    def extract_code_smell(text: str) -> Optional[str]:
        """Extract primary code smell from analysis."""
        text_lower = text.lower()

        smell_patterns = {
            'long_method': ['long method', 'too many lines', 'multiple responsibilities', 'single responsibility',
                           'oversized', 'bloated', 'too complex', 'doing too much', 'too long'],
            'duplicate_code': ['massive duplication', 'duplicate code', 'duplicated code', 'repeated code', 'dry violation',
                              'copy paste', 'copy-paste', 'similar code', 'shared logic', 'identical pattern'],
            'conditional_complexity': ['conditional complexity', 'nested if', 'cyclomatic',
                                      'deeply nested', 'complex branching', 'too many conditions'],
            'feature_envy': ['feature envy', 'knows too much'],
            'data_clumps': ['data clump', 'parameter list', 'grouped data'],
            'god_class': ['god class', 'too many dependencies', 'mixed concerns', 'does everything',
                         'monolithic', 'kitchen sink', 'do-it-all', 'too many responsibilities'],
            'hardcoded_rules': ['hardcoded', 'magic number', 'business rules in code'],
            'n_plus_one_queries': ['n+1', 'n + 1', 'n+1 query', 'sequential query'],
            'primitive_obsession': ['primitive obsession', 'magic string'],
            'switch_statement': ['switch statement', 'large switch', 'massive switch'],
            'type_conditional': ['type-based conditional', 'type conditional'],
            'orchestration_in_service': ['orchestration', 'complex rollback', 'business process in service'],
        }

        for smell, keywords in smell_patterns.items():
            if any(kw in text_lower for kw in keywords):
                return smell
        return None

    def extract_technique(text: str) -> Optional[str]:
        """Extract refactoring technique from analysis."""
        text_lower = text.lower()

        technique_patterns = {
            'extract_method': ['extract method', 'break into', 'smaller function',
                              'pull out', 'separate function', 'isolate logic', 'focused function'],
            'extract_class': ['extract class', 'value object', 'data class',
                             'separate class', 'dedicated class', 'new class'],
            'extract_function': ['extract function'],
            'strategy_pattern': ['strategy pattern', 'polymorphism',
                                'replace conditional', 'dynamic dispatch'],
            'move_method': ['move method', 'move to', 'belongs in'],
            'extract_configuration': ['extract config', 'configuration file'],
            'template_method': ['template method', 'base method', 'hook method'],
            'facade_and_events': ['facade', 'event-driven'],
            'builder_pattern': ['builder pattern', 'fluent'],
            'rules_engine': ['rules engine'],
            'pipeline_pattern': ['pipeline', 'compose'],
            'validation_schema': ['validation schema', 'declarative validation'],
            'saga_pattern': ['saga pattern'],
            'batch_fetching': ['batch fetch', 'dataloader', 'parallel fetch', 'promise.all'],
            'lookup_table': ['lookup table', 'lookup map', 'operation lookup'],
            'config_driven_events': ['configurable event', 'event config', 'event hierarchy', 'configureevent'],
            'saga_orchestrator': ['saga orchestrator', 'orchestrator pattern'],
            'renderer_registry': ['renderer registry', 'noderenderer'],
            'generic_helper': ['generic request', 'request helper', 'generic base', 'base service',
                              'reusable crud', 'abstract base'],
            'state_pattern': ['state pattern'],
            'command_pattern': ['command pattern'],
            'introduce_parameter_object': ['parameter object', 'dto'],
        }

        for technique, keywords in technique_patterns.items():
            if any(kw in text_lower for kw in keywords):
                return technique
        return None

    def extract_risk_level(text: str) -> Optional[str]:
        """Extract risk level from analysis."""
        text_lower = text.lower()

        risk_pattern = r'risk[^:]*[:\s]*(low|medium|high)'
        match = re.search(risk_pattern, text_lower)
        if match:
            return match.group(1)

        # Infer from content
        if any(ind in text_lower for ind in ['critical', 'high risk', 'business critical', 'payment', 'auth']):
            return 'high'
        elif any(ind in text_lower for ind in ['medium risk', 'test carefully', 'multiple file']):
            return 'medium'
        elif any(ind in text_lower for ind in ['low risk', 'additive', 'simple', 'no behavior change']):
            return 'low'
        return None

    def has_code_example(text: str) -> bool:
        """Check if analysis includes refactored code example."""
        return '```' in text and any(kw in text.lower() for kw in
            ['class', 'function', 'interface', 'const', 'type', 'import'])

    def check_rationale_quality(text: str, smell: Optional[str], technique: Optional[str]) -> float:
        """
        Check if the analysis explains WHY the technique addresses the smell.

        Looks for:
        - Explicit connection between smell and technique
        - Before/after comparison
        - Benefits explanation
        - Trade-offs mentioned
        """
        text_lower = text.lower()
        rationale_score = 0.0

        # Check for explicit connection keywords
        connection_phrases = [
            'this addresses', 'this fixes', 'this solves', 'this reduces',
            'will eliminate', 'will reduce', 'will simplify',
            'because', 'since', 'therefore', 'which means',
        ]
        if any(phrase in text_lower for phrase in connection_phrases):
            rationale_score += 0.3

        # Check for before/after comparison
        if re.search(r'before[:\s]|after[:\s]|current[:\s]|proposed[:\s]', text_lower):
            rationale_score += 0.25

        # Check for benefits explanation
        benefit_phrases = [
            'benefit', 'advantage', 'improve', 'easier to', 'simpler',
            'testable', 'maintainable', 'readable', 'reusable',
        ]
        if any(phrase in text_lower for phrase in benefit_phrases):
            rationale_score += 0.25

        # Check for trade-offs or considerations
        tradeoff_phrases = [
            'trade-off', 'tradeoff', 'consider', 'note that', 'however',
            'caveat', 'limitation', 'careful', 'ensure',
        ]
        if any(phrase in text_lower for phrase in tradeoff_phrases):
            rationale_score += 0.2

        return min(1.0, rationale_score)

    score = 0.0

    # Component 1: Code smell detection (25%) - uses semantic equivalence
    expected_smell = extract_code_smell(expected)
    actual_smell = extract_code_smell(actual)

    if expected_smell is None and actual_smell is None:
        # Neither detected - use SequenceMatcher on smell-related text
        smell_score = SequenceMatcher(None, expected[:400].lower(), actual[:400].lower()).ratio() * 0.5
    elif expected_smell is not None or actual_smell is not None:
        smell_score = _score_categorical_match(
            expected_smell,
            actual_smell,
            SMELL_EQUIVALENCES,
            related_groups=None,  # Use group membership only
        )
    else:
        smell_score = 0.0
    score += 0.25 * smell_score

    # Component 2: Technique selection (25%) - uses smell-technique affinity
    expected_technique = extract_technique(expected)
    actual_technique = extract_technique(actual)

    # If we have both smell and technique, check if technique is valid for smell
    if actual_smell and actual_technique:
        # Use affinity-based scoring
        technique_score = _score_technique_for_smell(
            actual_smell,
            actual_technique,
            SMELL_EQUIVALENCES,
            TECHNIQUE_EQUIVALENCES,
            SMELL_TECHNIQUE_AFFINITY,
        )

        # Also give credit if technique matches expected technique
        if expected_technique:
            expected_technique_score = _score_categorical_match(
                expected_technique,
                actual_technique,
                TECHNIQUE_EQUIVALENCES,
                related_groups=None,
            )
            # Blend: prioritize affinity score but reward exact match
            technique_score = max(technique_score, expected_technique_score)

        score += 0.25 * technique_score
    elif actual_technique:
        # No smell detected, just check technique equivalence
        technique_score = _score_categorical_match(
            expected_technique,
            actual_technique,
            TECHNIQUE_EQUIVALENCES,
            related_groups=None,
        )
        score += 0.25 * technique_score * 0.8  # Softened discount (was 0.7)
    else:
        # No technique suggested - use SequenceMatcher fallback
        # Extract technique-related text and compare semantically
        if expected_technique:
            similarity = SequenceMatcher(None, expected[:300].lower(), actual[:300].lower()).ratio()
            score += 0.25 * similarity * 0.5  # Partial credit for semantic similarity
        else:
            score += 0.25 * 0.15  # Minimal base credit

    # Component 3: Risk assessment (15%) - uses adjacent-level credit
    expected_risk = extract_risk_level(expected)
    actual_risk = extract_risk_level(actual)

    if expected_risk and actual_risk:
        if expected_risk == actual_risk:
            score += 0.15
        else:
            risk_order = ['low', 'medium', 'high']
            if expected_risk in risk_order and actual_risk in risk_order:
                idx_exp = risk_order.index(expected_risk)
                idx_act = risk_order.index(actual_risk)
                if abs(idx_exp - idx_act) == 1:
                    score += 0.15 * 0.6  # 60% for adjacent
                else:
                    score += 0.15 * 0.2  # 20% for opposite ends
    elif actual_risk:
        score += 0.15 * 0.5  # Some risk assessment provided

    # Component 4: Code example (15%)
    if has_code_example(expected):
        if has_code_example(actual):
            score += 0.15
    else:
        score += 0.15 * 0.5 if has_code_example(actual) else 0.15 * 0.2

    # Component 5: Rationale quality (20%) - NEW
    rationale_score = check_rationale_quality(actual, actual_smell, actual_technique)
    score += 0.20 * rationale_score

    return min(1.0, score)


# =============================================================================
# Tier 2: Agent Metrics
# =============================================================================

def discovery_quality_match(expected: str, actual: str) -> float:
    """
    Evaluate discovery report quality for capability-discoverer agent.

    Components:
    - Category accuracy (20%) - correct classification
    - Redundancy detection (25%) - correctly identifies duplicates vs novel
    - Source quality (15%) - sources cited with URLs
    - Score calibration (20%) - preliminary score within ±15 of expected
    - Report structure (20%) - has required sections
    """
    score = 0.0
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    # Component 1: Category accuracy (20%)
    categories = ['mcp', 'skill', 'technique', 'agent', 'tool', 'pattern', 'cli', 'workflow']
    expected_cat = None
    actual_cat = None
    for cat in categories:
        cat_pattern = rf'(?:category|type)[:\s]*{cat}'
        if re.search(cat_pattern, expected_lower):
            expected_cat = cat
        if re.search(cat_pattern, actual_lower):
            actual_cat = cat
    # Fallback: check for category mentioned anywhere
    if not expected_cat:
        for cat in categories:
            if cat in expected_lower:
                expected_cat = cat
                break
    if not actual_cat:
        for cat in categories:
            if cat in actual_lower:
                actual_cat = cat
                break

    if expected_cat and actual_cat:
        if expected_cat == actual_cat:
            score += 0.20
        else:
            # Related categories get partial credit
            related = {
                'mcp': {'tool'}, 'tool': {'mcp'},
                'skill': {'technique', 'workflow', 'pattern'},
                'technique': {'skill', 'pattern', 'workflow'},
                'pattern': {'technique', 'workflow', 'skill'},
                'workflow': {'technique', 'pattern', 'skill'},
                'agent': set(), 'cli': set(),
            }
            if actual_cat in related.get(expected_cat, set()):
                score += 0.20 * 0.6
    elif actual_cat:
        score += 0.20 * 0.3  # Some classification provided

    # Component 2: Redundancy detection (25%)
    redundancy_labels = ['novel', 'duplicate', 'improvement']
    expected_redundancy = None
    actual_redundancy = None
    for label in redundancy_labels:
        if re.search(rf'(?:result|status|check)[:\s]*{label}', expected_lower):
            expected_redundancy = label
        elif label in expected_lower:
            expected_redundancy = label
        if re.search(rf'(?:result|status|check)[:\s]*{label}', actual_lower):
            actual_redundancy = label
        elif label in actual_lower:
            actual_redundancy = label

    if expected_redundancy and actual_redundancy:
        if expected_redundancy == actual_redundancy:
            score += 0.25
        else:
            # Confusing NOVEL with IMPROVEMENT is less bad than with DUPLICATE
            confusion_credit = {
                ('novel', 'improvement'): 0.5,
                ('improvement', 'novel'): 0.5,
                ('novel', 'duplicate'): 0.0,
                ('duplicate', 'novel'): 0.0,
                ('improvement', 'duplicate'): 0.2,
                ('duplicate', 'improvement'): 0.2,
            }
            credit = confusion_credit.get((expected_redundancy, actual_redundancy), 0.1)
            score += 0.25 * credit
    elif actual_redundancy:
        score += 0.25 * 0.3

    # Component 3: Source quality (15%)
    actual_urls = re.findall(r'https?://[^\s\)]+', actual)
    if actual_urls:
        url_score = min(1.0, len(actual_urls) / 2)  # Full credit at 2+ URLs
        score += 0.15 * url_score
    elif re.search(r'source|reference|link', actual_lower):
        score += 0.15 * 0.2  # Mentioned but no URLs

    # Component 4: Score calibration (20%)
    expected_scores = re.findall(r'(?:score|rating)[:\s]*(\d{1,3})', expected_lower)
    actual_scores = re.findall(r'(?:score|rating)[:\s]*(\d{1,3})', actual_lower)
    # Also try X/100 format
    expected_scores += re.findall(r'(\d{1,3})\s*/\s*100', expected_lower)
    actual_scores += re.findall(r'(\d{1,3})\s*/\s*100', actual_lower)

    if expected_scores and actual_scores:
        exp_score = int(expected_scores[0])
        act_score = int(actual_scores[0])
        diff = abs(exp_score - act_score)
        if diff <= 5:
            score += 0.20
        elif diff <= 15:
            score += 0.20 * (1.0 - (diff - 5) / 20)  # Linear decay
        elif diff <= 30:
            score += 0.20 * 0.2
    elif actual_scores:
        score += 0.20 * 0.4  # Score present but can't compare

    # Component 5: Report structure (20%)
    structure_checks = {
        'title': bool(re.search(r'(?:^#|discovery|title)[:\s]', actual_lower, re.MULTILINE)),
        'summary': bool(re.search(r'(?:summary|overview|description)', actual_lower)),
        'research_questions': bool(re.search(r'(?:research|question|investigate)', actual_lower)),
        'redundancy_section': bool(re.search(r'(?:redundancy|registry|existing)', actual_lower)),
    }
    structure_score = sum(structure_checks.values()) / len(structure_checks)
    score += 0.20 * structure_score

    return min(1.0, score)


def fact_check_quality_match(expected: str, actual: str) -> float:
    """
    Evaluate fact-check report quality for fact-checker agent.

    Components:
    - Claim extraction (25%) - identifies same verifiable claims
    - Verdict accuracy (30%) - correct verdicts
    - Evidence citation (20%) - sources cited for each verdict
    - Verdict diversity (10%) - uses appropriate verdict types
    - Report structure (15%) - has required sections
    """
    score = 0.0
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    # Component 1: Claim extraction (25%)
    # Extract claim-like sentences from both
    expected_claims = set()
    actual_claims = set()
    # Look for claims in tables or bullet points
    for match in re.finditer(r'(?:claim|statement)[:\s]*["\']?([^"\'|\n]+)', expected_lower):
        words = set(match.group(1).strip().split())
        expected_claims.update(w for w in words if len(w) > 3)
    for match in re.finditer(r'(?:claim|statement)[:\s]*["\']?([^"\'|\n]+)', actual_lower):
        words = set(match.group(1).strip().split())
        actual_claims.update(w for w in words if len(w) > 3)

    # Fallback: extract significant content words
    if not expected_claims:
        expected_claims = set(w for w in re.findall(r'\b\w{4,}\b', expected_lower)
                            if w not in {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 'about', 'which', 'would', 'could', 'should', 'there', 'these', 'those', 'other', 'some', 'more', 'also', 'than'})
    if not actual_claims:
        actual_claims = set(w for w in re.findall(r'\b\w{4,}\b', actual_lower)
                          if w not in {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 'about', 'which', 'would', 'could', 'should', 'there', 'these', 'those', 'other', 'some', 'more', 'also', 'than'})

    if expected_claims and actual_claims:
        overlap = len(expected_claims & actual_claims)
        precision = overlap / len(actual_claims) if actual_claims else 0
        recall = overlap / len(expected_claims) if expected_claims else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        score += 0.25 * min(1.0, f1 * 1.5)  # Scale up slightly since word overlap is strict

    # Component 2: Verdict accuracy (30%)
    verdict_types = ['verified', 'inaccurate', 'contested', 'oversimplified', "author's data",
                     'accurate', 'false', 'misleading', 'unverifiable', 'partially true', 'correct']
    # Normalize verdict aliases
    verdict_normalize = {
        'accurate': 'verified', 'correct': 'verified', 'true': 'verified', 'confirmed': 'verified',
        'false': 'inaccurate', 'wrong': 'inaccurate', 'incorrect': 'inaccurate',
        'misleading': 'oversimplified', 'partially true': 'oversimplified',
        'unverifiable': "author's data",
    }

    expected_verdicts = []
    actual_verdicts = []
    for vt in verdict_types:
        expected_verdicts.extend([vt] * len(re.findall(rf'\b{re.escape(vt)}\b', expected_lower)))
        actual_verdicts.extend([vt] * len(re.findall(rf'\b{re.escape(vt)}\b', actual_lower)))

    # Normalize
    expected_normalized = [verdict_normalize.get(v, v) for v in expected_verdicts]
    actual_normalized = [verdict_normalize.get(v, v) for v in actual_verdicts]

    if expected_normalized and actual_normalized:
        # Count matching verdict types
        exp_counts = {}
        act_counts = {}
        for v in expected_normalized:
            exp_counts[v] = exp_counts.get(v, 0) + 1
        for v in actual_normalized:
            act_counts[v] = act_counts.get(v, 0) + 1

        matched = sum(min(exp_counts.get(v, 0), act_counts.get(v, 0)) for v in set(exp_counts) | set(act_counts))
        total = max(sum(exp_counts.values()), sum(act_counts.values()))
        score += 0.30 * (matched / total if total > 0 else 0)
    elif actual_normalized:
        score += 0.30 * 0.2  # Some verdicts present

    # Component 3: Evidence citation (20%)
    actual_urls = re.findall(r'https?://[^\s\)]+', actual)
    evidence_phrases = len(re.findall(
        r'(?:according to|source|evidence|per|cited|reference|confirms|states)',
        actual_lower
    ))
    citation_score = min(1.0, (len(actual_urls) * 0.3 + evidence_phrases * 0.15))
    score += 0.20 * citation_score

    # Component 4: Verdict diversity (10%)
    unique_verdicts = set(actual_normalized)
    if len(unique_verdicts) >= 3:
        score += 0.10
    elif len(unique_verdicts) == 2:
        score += 0.10 * 0.6
    elif len(unique_verdicts) == 1:
        score += 0.10 * 0.3

    # Component 5: Report structure (15%)
    structure_checks = {
        'claims_section': bool(re.search(r'(?:claim|extracted|identified)', actual_lower)),
        'verdict_section': bool(re.search(r'(?:verdict|finding|assessment|result)', actual_lower)),
        'confidence': bool(re.search(r'(?:confidence|certainty|reliability)', actual_lower)),
        'summary': bool(re.search(r'(?:summary|overall|conclusion)', actual_lower)),
    }
    structure_score = sum(structure_checks.values()) / len(structure_checks)
    score += 0.15 * structure_score

    return min(1.0, score)


def research_quality_match(expected: str, actual: str) -> float:
    """
    Evaluate research report quality for web-researcher agent.

    Components:
    - Topic coverage (25%) - addresses all key themes
    - Source diversity (20%) - multiple sources cited
    - Synthesis quality (25%) - organized by theme, has executive summary
    - Citation integrity (15%) - claims attributed to sources
    - Confidence calibration (15%) - has explicit confidence assessment
    """
    score = 0.0
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    # Component 1: Topic coverage (25%)
    # Extract significant topic words from expected
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their',
                  'about', 'which', 'would', 'could', 'should', 'there', 'these',
                  'those', 'other', 'some', 'more', 'also', 'than', 'they', 'will',
                  'into', 'when', 'what', 'each', 'most', 'such', 'like', 'over',
                  'both', 'only', 'very', 'just', 'make', 'well', 'back', 'even',
                  'good', 'much', 'then', 'them', 'many', 'made', 'after', 'being',
                  'does', 'used', 'using', 'based', 'provides', 'including', 'while'}
    expected_words = set(w for w in re.findall(r'\b\w{4,}\b', expected_lower) if w not in stop_words)
    actual_words = set(w for w in re.findall(r'\b\w{4,}\b', actual_lower) if w not in stop_words)

    if expected_words:
        overlap = len(expected_words & actual_words)
        coverage = overlap / len(expected_words)
        score += 0.25 * min(1.0, coverage * 1.5)  # Scale up since exact word match is strict

    # Component 2: Source diversity (20%)
    actual_urls = set(re.findall(r'https?://[^\s\)]+', actual))
    # Also count source references
    source_refs = len(re.findall(r'(?:according to|source:|per |cited in|reference:)', actual_lower))
    unique_domains = set()
    for url in actual_urls:
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            unique_domains.add(domain_match.group(1))

    source_count = max(len(unique_domains), source_refs)
    if source_count >= 4:
        score += 0.20
    elif source_count >= 2:
        score += 0.20 * (source_count / 4)
    elif source_count == 1:
        score += 0.20 * 0.3

    # Component 3: Synthesis quality (25%)
    synthesis_score = 0.0

    # Has executive summary?
    if re.search(r'(?:executive\s+summary|overview|key\s+takeaway|tl;?dr)', actual_lower):
        synthesis_score += 0.35

    # Organized by theme (not by source)?
    # Check for thematic headers (## Topic Name) vs source-oriented headers
    headers = re.findall(r'^##?\s+(.+)', actual, re.MULTILINE)
    source_oriented = sum(1 for h in headers if re.search(r'source|reference|article|blog|post', h.lower()))
    theme_oriented = len(headers) - source_oriented
    if headers:
        if theme_oriented > source_oriented:
            synthesis_score += 0.35
        elif theme_oriented == source_oriented:
            synthesis_score += 0.15

    # Has structured findings (bullets, numbered lists)?
    bullet_count = len(re.findall(r'^\s*[-*]\s', actual, re.MULTILINE))
    if bullet_count >= 5:
        synthesis_score += 0.3
    elif bullet_count >= 2:
        synthesis_score += 0.15

    score += 0.25 * min(1.0, synthesis_score)

    # Component 4: Citation integrity (15%)
    # Check if claims are attributed
    attribution_phrases = len(re.findall(
        r'(?:according to|reported by|found that|states that|shows that|per |noted by)',
        actual_lower
    ))
    inline_citations = len(re.findall(r'\[\d+\]|\[source\]|\(source:', actual_lower))
    citation_score = min(1.0, (attribution_phrases + inline_citations) / 4)
    score += 0.15 * citation_score

    # Component 5: Confidence calibration (15%)
    confidence_score = 0.0
    if re.search(r'(?:confidence|certainty|reliability|caveat|limitation)', actual_lower):
        confidence_score += 0.5
    if re.search(r'(?:high confidence|medium confidence|low confidence|uncertain|likely|probably)', actual_lower):
        confidence_score += 0.5
    score += 0.15 * min(1.0, confidence_score)

    return min(1.0, score)


# =============================================================================
# Tier 3: Skill Metrics
# =============================================================================

def writing_review_quality_match(expected: str, actual: str) -> float:
    """
    Evaluate writing review quality for writing-review skill. (v3)

    Components:
    - Text similarity (20%) - overall text overlap via SequenceMatcher
    - Topic-specific overlap (15%) - blog-specific terms match (not structural)
    - Perspective alignment (15%) - correct perspective applied
    - Evidence grounding (20%) - verdicts supported by direct quotes
    - Dimension coverage (15%) - expected dimensions present in actual
    - Actionable feedback (15%) - concrete suggestions present
    """
    score = 0.0
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    # Component 1: Text similarity (20%)
    # SequenceMatcher on substantial portions for overall content matching
    # Use middle section (skip headers/metadata, capture analysis body)
    exp_body = expected_lower[200:1200] if len(expected_lower) > 200 else expected_lower
    act_body = actual_lower[200:1200] if len(actual_lower) > 200 else actual_lower
    text_sim = SequenceMatcher(None, exp_body, act_body).ratio()
    score += 0.20 * text_sim

    # Component 2: Topic-specific overlap (15%)
    # Filter OUT known structural terms shared across all reviews
    structural_terms = {'date', 'draft', 'model', 'word count', 'has claude sections',
                        'evidence', 'verdict', 'assessment', 'overall', 'score',
                        'editorial critic', 'target reader', 'tone analyst', 'fact checker',
                        'writing review', 'excerpt', 'dimension', 'criterion'}

    expected_terms = set()
    for term in re.findall(r'\*\*([^*]{3,30})\*\*', expected):
        t = term.lower().strip()
        if t not in structural_terms and len(t) > 5:
            expected_terms.add(t)
    for term in re.findall(r'^##?\s+(.{3,40})$', expected, re.MULTILINE):
        t = term.lower().strip()
        if not any(s in t for s in structural_terms) and len(t) > 8:
            expected_terms.add(t)
    # Extract quoted content (unique to each review's blog post)
    for term in re.findall(r'["\u201c]([^"\u201d]{15,60})["\u201d]', expected):
        expected_terms.add(term.lower().strip())

    if expected_terms:
        matched = sum(1 for t in expected_terms if t in actual_lower)
        overlap_ratio = matched / len(expected_terms)
        score += 0.15 * min(1.0, overlap_ratio * 1.5)
    else:
        score += 0.15 * text_sim * 0.5  # Fallback

    # Component 3: Perspective alignment (15%)
    perspectives = ['editorial_critic', 'editorial critic', 'target_reader', 'target reader',
                     'tone_analyst', 'tone analyst', 'fact_checker', 'fact checker',
                     'critic', 'reader', 'tone', 'factcheck']

    expected_perspective = None
    actual_perspective = None
    for p in perspectives:
        if p in expected_lower:
            expected_perspective = p.replace('_', ' ').split()[0]
            break
    for p in perspectives:
        if p in actual_lower:
            actual_perspective = p.replace('_', ' ').split()[0]
            break

    if expected_perspective and actual_perspective:
        if expected_perspective == actual_perspective:
            score += 0.15
        else:
            score += 0.15 * 0.2
    elif actual_perspective:
        score += 0.15 * 0.5

    # Component 4: Evidence grounding (20%)
    quotes = re.findall(r'["\u201c]([^"\u201d]{10,})["\u201d]|>\s*(.{10,})', actual)
    quote_count = len(quotes)
    if quote_count >= 3:
        score += 0.20
    elif quote_count >= 1:
        score += 0.20 * (quote_count / 3)
    elif re.search(r'(?:excerpt|quote|passage|text says|writes)', actual_lower):
        score += 0.20 * 0.2

    # Component 5: Dimension coverage (15%)
    expected_dims = set(re.findall(r'(?:dimension|criterion|aspect)[:\s]*([^\n,;]+)', expected_lower))
    actual_dims = set(re.findall(r'(?:dimension|criterion|aspect)[:\s]*([^\n,;]+)', actual_lower))

    expected_scores_count = len(re.findall(r'\b[1-5]\s*/\s*5\b|\b[1-9]\s*/\s*10\b|\bscore[:\s]*\d', expected_lower))
    actual_scores_count = len(re.findall(r'\b[1-5]\s*/\s*5\b|\b[1-9]\s*/\s*10\b|\bscore[:\s]*\d', actual_lower))

    if expected_scores_count > 0:
        dim_ratio = min(1.0, actual_scores_count / expected_scores_count)
        score += 0.15 * dim_ratio
    elif expected_dims:
        if actual_dims:
            overlap = len(expected_dims & actual_dims)
            score += 0.15 * (overlap / len(expected_dims))
        else:
            score += 0.15 * 0.1
    else:
        score += 0.15 * text_sim * 0.5  # Fallback

    # Component 6: Actionable feedback (15%)
    suggestion_patterns = [
        r'(?:suggest|recommend|consider|should|could|try|improve|revise|add|remove|clarify)',
        r'(?:action|feedback|next step|improvement)',
    ]
    suggestion_count = sum(len(re.findall(p, actual_lower)) for p in suggestion_patterns)
    verdict_present = bool(re.search(r'(?:verdict|overall|assessment|rating)[:\s]', actual_lower))
    feedback_score = min(1.0, suggestion_count / 4 + (0.3 if verdict_present else 0))
    score += 0.15 * feedback_score

    return min(1.0, score)


def handoff_quality_match(expected: str, actual: str) -> float:
    """
    Evaluate session handoff quality for session-handoff skill. (v2)

    Components:
    - Content overlap (20%) - actual references same entities/topics as expected
    - Next step specificity (25%) - specific enough to resume without questions
    - File path overlap (15%) - references same files as expected
    - State completeness (15%) - covers current state, decisions, blockers
    - Verification commands (15%) - includes commands to verify state
    - Structure quality (10%) - organized with clear sections
    """
    score = 0.0
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    # Component 1: Content overlap (20%)
    # Extract key terms from expected and check overlap with actual
    # Key terms: multi-word technical phrases, identifiers, specific nouns
    expected_terms = set()
    # Extract backtick-enclosed terms
    for term in re.findall(r'`([^`]{3,40})`', expected):
        expected_terms.add(term.lower().strip())
    # Extract CamelCase or snake_case identifiers
    for term in re.findall(r'\b([A-Z][a-z]+[A-Z]\w+|[a-z]+_[a-z_]+)\b', expected):
        expected_terms.add(term.lower())
    # Extract technology/framework names (capitalized words 3+ chars)
    for term in re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b', expected):
        term_lower = term.lower()
        # Skip common structural words
        if term_lower not in {'session', 'handoff', 'current', 'state', 'completed', 'work',
                               'next', 'steps', 'files', 'read', 'first', 'key', 'decisions',
                               'verification', 'commands', 'note', 'summary', 'changes'}:
            expected_terms.add(term_lower)

    if expected_terms:
        matched = sum(1 for t in expected_terms if t in actual_lower)
        overlap_ratio = matched / len(expected_terms)
        score += 0.20 * min(1.0, overlap_ratio * 1.5)  # Boost: 67% overlap = full credit
    else:
        # Fallback to SequenceMatcher
        score += 0.20 * SequenceMatcher(None, expected_lower[:500], actual_lower[:500]).ratio()

    # Component 2: Next step specificity (25%)
    specific_markers = [
        r'(?:complete|finish|implement|add|fix|create|update|remove|test)\s+\w+',
        r'(?:POST|GET|PUT|DELETE)\s+/\w+',
        r'(?:function|method|class|component)\s+\w+',
    ]
    vague_markers = ['continue working', 'finish up', 'keep going', 'do more', 'work on it']

    specific_count = sum(len(re.findall(p, actual_lower)) for p in specific_markers)
    vague_count = sum(1 for v in vague_markers if v in actual_lower)

    if specific_count >= 3 and vague_count == 0:
        score += 0.25
    elif specific_count >= 2:
        score += 0.25 * 0.7
    elif specific_count >= 1:
        score += 0.25 * 0.4
    elif vague_count > 0:
        score += 0.25 * 0.1

    # Component 3: File path overlap (15%)
    # Extract file paths from expected AND actual, then compare
    def extract_file_paths(text):
        paths = set()
        for p in re.findall(r'`([^`]*(?:\.\w{2,4}|/\w+/)[^`]*)`', text):
            paths.add(p.lower().strip())
        for p in re.findall(r'(?:file|path|module)[:\s]*`?([^\s,`]+\.\w{2,4})', text.lower()):
            paths.add(p.lower().strip())
        return paths

    expected_files = extract_file_paths(expected)
    actual_files = extract_file_paths(actual)

    if expected_files:
        if actual_files:
            overlap = len(expected_files & actual_files)
            score += 0.15 * min(1.0, overlap / len(expected_files))
        # No credit if expected has files but actual doesn't
    else:
        # No expected files: credit based on actual having any files
        if actual_files:
            score += 0.15 * min(1.0, len(actual_files) / 3)

    # Component 4: State completeness (15%)
    state_checks = {
        'current_state': bool(re.search(r'(?:current|status|state|phase|progress)', actual_lower)),
        'completed': bool(re.search(r'(?:completed|done|finished|implemented)', actual_lower)),
        'pending': bool(re.search(r'(?:pending|remaining|todo|next|still need)', actual_lower)),
        'decisions': bool(re.search(r'(?:decision|chose|rationale|because|why)', actual_lower)),
        'blockers': bool(re.search(r'(?:blocker|blocked|issue|problem|depends on)', actual_lower)),
    }
    state_score = sum(state_checks.values()) / len(state_checks)
    score += 0.15 * state_score

    # Component 5: Verification commands (15%)
    code_blocks = re.findall(r'```(?:bash|sh|shell)?\n(.+?)```', actual, re.DOTALL)
    inline_commands = re.findall(r'`((?:npm|pnpm|yarn|python|pytest|curl|git|make)\s+[^`]+)`', actual)
    command_count = len(code_blocks) + len(inline_commands)

    if command_count >= 2:
        score += 0.15
    elif command_count == 1:
        score += 0.15 * 0.6
    elif re.search(r'(?:run|execute|verify|check|test)', actual_lower):
        score += 0.15 * 0.2

    # Component 6: Structure quality (10%)
    headers = re.findall(r'^##?\s+.+', actual, re.MULTILINE)
    has_bullets = bool(re.findall(r'^\s*[-*]\s', actual, re.MULTILINE))
    structure_score = min(1.0, len(headers) / 3 + (0.3 if has_bullets else 0))
    score += 0.10 * structure_score

    return min(1.0, score)


# =============================================================================
# Publication Review Metrics
# =============================================================================

def publication_review_match(expected: str, actual: str) -> float:
    """
    Evaluate publication review output quality.

    Three weighted components:
    - 50%: Finding recall (weighted by tier: MUST=3x, SHOULD=1.5x, NICE=0.5x)
    - 30%: Tier accuracy (exact=1.0, off-by-one=0.5, off-by-two=0.2)
    - 20%: Precision (1 - unmatched_actual/total_actual, clamped [0,1])

    Uses keyword-based Jaccard similarity (threshold 0.25) for finding matching,
    since models describe the same issue differently.

    Args:
        expected: Gold standard review output (from fix-manifest triage)
        actual: Model-generated review output

    Returns:
        Score between 0.0 and 1.0
    """
    from .extractors import (
        extract_review_findings,
        match_review_findings,
        ReviewFinding,
    )

    expected_findings = extract_review_findings(expected)
    actual_findings = extract_review_findings(actual)

    if not expected_findings:
        # No expected findings — score based on actual being empty too
        return 1.0 if not actual_findings else 0.5

    if not actual_findings:
        return 0.0

    # Match expected findings to actual
    # Hybrid matching: anchor entities (60%) + keyword Jaccard (40%)
    # Threshold 0.15 catches entity-anchored matches while filtering noise
    matches = match_review_findings(expected_findings, actual_findings, threshold=0.15)

    # Tier weights for recall
    tier_weights = {"MUST": 3.0, "SHOULD": 1.5, "NICE": 0.5}

    # Component 1: Finding recall (50%) — weighted by tier
    total_weight = sum(tier_weights.get(e.tier, 1.0) for e in expected_findings)
    recalled_weight = 0.0

    for exp, matched, sim in matches:
        if matched is not None:
            recalled_weight += tier_weights.get(exp.tier, 1.0)

    recall = recalled_weight / total_weight if total_weight > 0 else 0.0

    # Component 2: Tier accuracy (30%) — for matched findings
    matched_pairs = [(exp, act, sim) for exp, act, sim in matches if act is not None]
    if matched_pairs:
        tier_order = ["MUST", "SHOULD", "NICE"]

        tier_accuracy_sum = 0.0
        for exp, act, sim in matched_pairs:
            if exp.tier == act.tier:
                tier_accuracy_sum += 1.0
            elif exp.tier in tier_order and act.tier in tier_order:
                distance = abs(tier_order.index(exp.tier) - tier_order.index(act.tier))
                if distance == 1:
                    tier_accuracy_sum += 0.5
                else:
                    tier_accuracy_sum += 0.2
            else:
                tier_accuracy_sum += 0.2

        tier_accuracy = tier_accuracy_sum / len(matched_pairs)
    else:
        tier_accuracy = 0.0

    # Component 3: Precision (20%) — penalize unmatched actual findings
    matched_actual_count = sum(1 for _, act, _ in matches if act is not None)
    total_actual = len(actual_findings)
    unmatched_actual = total_actual - matched_actual_count

    precision = max(0.0, 1.0 - (unmatched_actual / total_actual)) if total_actual > 0 else 1.0

    # Weighted combination
    score = 0.50 * recall + 0.30 * tier_accuracy + 0.20 * precision

    return min(1.0, max(0.0, score))
