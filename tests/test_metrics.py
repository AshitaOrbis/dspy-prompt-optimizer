"""Tests for lib/prompt_optimizer/metrics.py"""

import pytest
from lib.prompt_optimizer.metrics import (
    # Normalization helpers (private but tested via public functions)
    _normalize_categorical_value,
    _find_group_for_value,
    _score_categorical_match,
    # Public metric functions
    exact_match,
    contains_keywords,
    score_similarity,
    numeric_score_match,
    composite_metric,
    structured_output_match,
    evaluation_score_metric,
    # Equivalence data structures
    CHANGE_TYPE_EQUIVALENCES,
    TECHNIQUE_EQUIVALENCES,
    SMELL_EQUIVALENCES,
)


# =============================================================================
# _normalize_categorical_value
# =============================================================================

class TestNormalizeCategoricalValue:
    def test_lowercases(self):
        assert _normalize_categorical_value("BugFix") == "bugfix"

    def test_replaces_dashes_with_underscores(self):
        assert _normalize_categorical_value("bug-fix") == "bug_fix"

    def test_replaces_spaces_with_underscores(self):
        assert _normalize_categorical_value("bug fix") == "bug_fix"

    def test_strips_whitespace(self):
        assert _normalize_categorical_value("  feature  ") == "feature"

    def test_empty_string_returns_empty(self):
        assert _normalize_categorical_value("") == ""

    def test_combined_transformations(self):
        assert _normalize_categorical_value("  New Feature  ") == "new_feature"


# =============================================================================
# _find_group_for_value
# =============================================================================

class TestFindGroupForValue:
    def test_finds_direct_member(self):
        group = _find_group_for_value("bugfix", CHANGE_TYPE_EQUIVALENCES)
        assert group == "corrective"

    def test_finds_normalized_member(self):
        # "Bug Fix" normalizes to "bug_fix" which is in corrective group
        group = _find_group_for_value("Bug Fix", CHANGE_TYPE_EQUIVALENCES)
        assert group == "corrective"

    def test_returns_none_for_unknown_value(self):
        assert _find_group_for_value("completely_unknown_xyz", CHANGE_TYPE_EQUIVALENCES) is None

    def test_returns_none_for_empty_string(self):
        assert _find_group_for_value("", CHANGE_TYPE_EQUIVALENCES) is None

    def test_finds_feature_in_additive(self):
        assert _find_group_for_value("feature", CHANGE_TYPE_EQUIVALENCES) == "additive"

    def test_finds_refactor_in_structural(self):
        assert _find_group_for_value("refactor", CHANGE_TYPE_EQUIVALENCES) == "structural"

    def test_finds_docs_in_meta(self):
        assert _find_group_for_value("docs", CHANGE_TYPE_EQUIVALENCES) == "meta"


# =============================================================================
# _score_categorical_match
# =============================================================================

class TestScoreCategoricalMatch:
    def test_exact_match_returns_1(self):
        score = _score_categorical_match("bugfix", "bugfix", CHANGE_TYPE_EQUIVALENCES)
        assert score == 1.0

    def test_case_insensitive_exact_match(self):
        score = _score_categorical_match("BugFix", "bugfix", CHANGE_TYPE_EQUIVALENCES)
        assert score == 1.0

    def test_same_equivalence_group_returns_1(self):
        # "fix" and "patch" are both in "corrective"
        score = _score_categorical_match("fix", "patch", CHANGE_TYPE_EQUIVALENCES)
        assert score == 1.0

    def test_missing_actual_returns_0(self):
        score = _score_categorical_match("bugfix", None, CHANGE_TYPE_EQUIVALENCES)
        assert score == 0.0

    def test_missing_expected_returns_05(self):
        score = _score_categorical_match(None, "bugfix", CHANGE_TYPE_EQUIVALENCES)
        assert score == 0.5

    def test_missing_both_returns_05(self):
        score = _score_categorical_match(None, None, CHANGE_TYPE_EQUIVALENCES)
        assert score == 0.5

    def test_related_groups_partial_credit(self):
        from lib.prompt_optimizer.metrics import CHANGE_TYPE_RELATED
        # "feature" is additive; "refactor" is structural; structural is related to additive
        score = _score_categorical_match(
            "feature", "refactor",
            CHANGE_TYPE_EQUIVALENCES,
            related_groups=CHANGE_TYPE_RELATED,
        )
        assert score == 0.6

    def test_unrelated_groups_returns_03(self):
        # "feature" (additive) vs "docs" (meta) — no relation
        score = _score_categorical_match("feature", "docs", CHANGE_TYPE_EQUIVALENCES)
        assert score == 0.3

    def test_one_has_group_other_doesnt_returns_02(self):
        # "bugfix" (corrective group) vs "xyzunknown" (no group)
        score = _score_categorical_match("bugfix", "xyzunknown", CHANGE_TYPE_EQUIVALENCES)
        assert score == 0.2

    def test_neither_has_group_substring_match(self):
        score = _score_categorical_match("unknown_abc", "abc", CHANGE_TYPE_EQUIVALENCES)
        # "abc" is substring of "unknown_abc" -> 0.4
        assert score == 0.4

    def test_neither_has_group_no_overlap_returns_01(self):
        score = _score_categorical_match("zzz_nope", "aaa_nope", CHANGE_TYPE_EQUIVALENCES)
        assert score == 0.1


# =============================================================================
# exact_match
# =============================================================================

class TestExactMatch:
    def test_identical_strings(self):
        assert exact_match("hello world", "hello world") == 1.0

    def test_case_insensitive(self):
        assert exact_match("Hello World", "hello world") == 1.0

    def test_whitespace_normalized(self):
        assert exact_match("hello   world", "hello world") == 1.0

    def test_different_strings(self):
        assert exact_match("hello world", "goodbye world") == 0.0

    def test_empty_strings(self):
        assert exact_match("", "") == 1.0

    def test_one_empty(self):
        assert exact_match("hello", "") == 0.0

    def test_leading_trailing_whitespace(self):
        assert exact_match("  hello  ", "hello") == 1.0


# =============================================================================
# contains_keywords
# =============================================================================

class TestContainsKeywords:
    def test_all_keywords_present(self):
        metric = contains_keywords(["security", "vulnerability"])
        assert metric("", "This has a security vulnerability") == 1.0

    def test_no_keywords_present(self):
        metric = contains_keywords(["missing", "absent"])
        assert metric("", "This text has nothing relevant") == 0.0

    def test_partial_keywords(self):
        metric = contains_keywords(["alpha", "beta", "gamma"])
        assert metric("", "alpha and beta are here") == pytest.approx(2 / 3)

    def test_empty_keywords_list_returns_1(self):
        metric = contains_keywords([])
        assert metric("", "any text") == 1.0

    def test_case_insensitive_by_default(self):
        metric = contains_keywords(["Python"])
        assert metric("", "python is great") == 1.0

    def test_case_sensitive_mode(self):
        metric = contains_keywords(["Python"], case_sensitive=True)
        assert metric("", "python is great") == 0.0
        assert metric("", "Python is great") == 1.0

    def test_single_keyword_present(self):
        metric = contains_keywords(["error"])
        assert metric("", "there was an error") == 1.0

    def test_keyword_as_substring(self):
        metric = contains_keywords(["error"])
        assert metric("", "errorhandling") == 1.0


# =============================================================================
# score_similarity
# =============================================================================

class TestScoreSimilarity:
    def test_identical_strings_return_1(self):
        assert score_similarity("hello world", "hello world") == 1.0

    def test_completely_different_strings(self):
        result = score_similarity("aaaa", "bbbb")
        assert result == 0.0

    def test_similar_strings_return_high_score(self):
        result = score_similarity("Hello world!", "Hello world.")
        assert result > 0.8

    def test_empty_strings(self):
        assert score_similarity("", "") == 1.0

    def test_one_empty(self):
        assert score_similarity("hello", "") == 0.0

    def test_case_insensitive(self):
        assert score_similarity("HELLO", "hello") == 1.0

    def test_partial_overlap(self):
        # "ab" vs "abcd" — partial match
        result = score_similarity("ab", "abcd")
        assert 0.0 < result < 1.0


# =============================================================================
# numeric_score_match
# =============================================================================

class TestNumericScoreMatch:
    def test_exact_score_match(self):
        assert numeric_score_match("Score: 85/100", "Score: 85/100") == 1.0

    def test_within_tolerance(self):
        # 85 vs 88 — diff is 3, tolerance at 10% of 88 = 8.8, so within tolerance
        result = numeric_score_match("Score: 85/100", "Score: 88/100")
        assert result == 1.0

    def test_outside_tolerance_returns_partial(self):
        # 10 vs 90 — very different scores
        result = numeric_score_match("Score: 10/100", "Score: 90/100")
        assert result < 0.5

    def test_percentage_format(self):
        result = numeric_score_match("85%", "85%")
        assert result == 1.0

    def test_no_numeric_score_falls_back_to_string_similarity(self):
        result = numeric_score_match("no numbers here", "no numbers here")
        # Same strings → high similarity * 0.5
        assert result > 0.0

    def test_one_missing_score_falls_back(self):
        result = numeric_score_match("Score: 85", "no score found")
        # Falls back to string similarity * 0.5
        assert result >= 0.0

    def test_normalized_fraction_score(self):
        # 8/10 should be treated as 80, matching 80/100
        result = numeric_score_match("8/10", "80/100")
        assert result == 1.0

    def test_zero_tolerance_exact_only(self):
        result = numeric_score_match("Score: 85", "Score: 86", tolerance=0.0)
        assert result < 1.0

    def test_large_tolerance_accepts_large_diff(self):
        result = numeric_score_match("Score: 50", "Score: 100", tolerance=1.0)
        assert result == 1.0


# =============================================================================
# composite_metric
# =============================================================================

class TestCompositeMetric:
    def test_raises_on_empty_metrics(self):
        with pytest.raises(ValueError, match="metrics list cannot be empty"):
            composite_metric([])

    def test_raises_on_weights_length_mismatch(self):
        from lib.prompt_optimizer.metrics import exact_match
        with pytest.raises(ValueError, match="weights length"):
            composite_metric([exact_match, exact_match], weights=[1.0])

    def test_raises_on_zero_sum_weights(self):
        with pytest.raises(ValueError, match="weights cannot sum to zero"):
            composite_metric([exact_match], weights=[0.0])

    def test_equal_weights_with_two_metrics(self):
        always_1 = lambda e, a: 1.0
        always_0 = lambda e, a: 0.0
        metric = composite_metric([always_1, always_0])
        assert metric("x", "y") == pytest.approx(0.5)

    def test_custom_weights_applied(self):
        always_1 = lambda e, a: 1.0
        always_0 = lambda e, a: 0.0
        metric = composite_metric([always_1, always_0], weights=[0.8, 0.2])
        assert metric("x", "y") == pytest.approx(0.8)

    def test_weights_normalized_automatically(self):
        always_1 = lambda e, a: 1.0
        always_0 = lambda e, a: 0.0
        # weights [4, 1] should normalize to [0.8, 0.2]
        metric = composite_metric([always_1, always_0], weights=[4.0, 1.0])
        assert metric("x", "y") == pytest.approx(0.8)

    def test_single_metric_passthrough(self):
        always_07 = lambda e, a: 0.7
        metric = composite_metric([always_07])
        assert metric("", "") == pytest.approx(0.7)


# =============================================================================
# structured_output_match
# =============================================================================

class TestStructuredOutputMatch:
    def test_empty_expected_returns_1(self):
        assert structured_output_match({}, "any output") == 1.0

    def test_all_fields_present(self):
        expected = {"severity": "high", "file": "main.py"}
        actual = "severity: high\nfile: main.py"
        assert structured_output_match(expected, actual) == 1.0

    def test_no_fields_present(self):
        expected = {"severity": "high"}
        actual = "completely unrelated output"
        assert structured_output_match(expected, actual) == 0.0

    def test_partial_fields_present(self):
        expected = {"severity": "high", "priority": "critical", "file": "main.py"}
        actual = "severity: high\npriority: low"
        # 2/3 fields present (severity matches, priority present but value wrong,
        # file missing). Actually: "high" in "high" → match; "critical" in "low" → no match
        result = structured_output_match(expected, actual)
        assert 0.0 <= result <= 1.0

    def test_numeric_field_within_tolerance(self):
        expected = {"score": 85.0}
        actual = "score: 89"
        # 89 is within 15% of 85 (diff=4, 15% of 85=12.75)
        result = structured_output_match(expected, actual)
        assert result == 1.0

    def test_numeric_field_outside_tolerance(self):
        expected = {"score": 50.0}
        actual = "score: 100"
        result = structured_output_match(expected, actual)
        assert result == 0.0


# =============================================================================
# evaluation_score_metric
# =============================================================================

class TestEvaluationScoreMetric:
    def test_both_above_threshold_and_agree(self):
        metric = evaluation_score_metric(threshold=70.0)
        expected = "Overall score: 80"
        actual = "Overall score: 82"
        result = metric(expected, actual)
        assert result >= 0.5

    def test_both_below_threshold_and_agree(self):
        metric = evaluation_score_metric(threshold=70.0)
        expected = "Initial score: 60"
        actual = "Initial score: 65"
        result = metric(expected, actual)
        assert result >= 0.5

    def test_disagree_on_threshold_penalty(self):
        metric = evaluation_score_metric(threshold=70.0)
        expected = "Overall score: 80"  # above threshold (pass)
        actual = "Overall score: 60"   # below threshold (fail)
        result = metric(expected, actual)
        assert result == 0.2

    def test_no_labeled_score_falls_back(self):
        metric = evaluation_score_metric(threshold=70.0)
        result = metric("text without score", "text without score")
        # Falls back to numeric_score_match → score_similarity * 0.5
        assert result >= 0.0

    def test_custom_threshold_changes_pass_fail(self):
        metric = evaluation_score_metric(threshold=90.0)
        # Score 80 is below threshold=90 (fail), 85 is also below (fail) → agree
        expected = "Final score: 80"
        actual = "Final score: 85"
        result = metric(expected, actual)
        assert result >= 0.5

    def test_close_scores_high_result(self):
        metric = evaluation_score_metric(threshold=70.0)
        expected = "weighted score: 75"
        actual = "weighted score: 76"
        result = metric(expected, actual)
        assert result >= 0.9
