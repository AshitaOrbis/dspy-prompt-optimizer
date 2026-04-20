"""Tests for writing-review and fact-checker demo transformers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer.demo_transformers import (
    transform_writing_review_demo,
    transform_factcheck_demo,
    get_demo_transformer,
    TRANSFORMER_MAP,
    TARGET_TRANSFORMER_MAP,
)

DATASETS = Path(__file__).parent.parent / "datasets"


def test_writing_review_transformer_registered():
    assert get_demo_transformer("writing-review") is transform_writing_review_demo
    assert get_demo_transformer("writing_review") is transform_writing_review_demo
    assert get_demo_transformer("writing_review_quality_match") is transform_writing_review_demo


def test_factcheck_transformer_registered():
    assert get_demo_transformer("fact-checker") is transform_factcheck_demo
    assert get_demo_transformer("fact_check") is transform_factcheck_demo
    assert get_demo_transformer("fact_check_quality_match") is transform_factcheck_demo


def test_writing_review_transform_condenses_real_sample():
    path = DATASETS / "writing-reviews.jsonl"
    if not path.exists():
        return  # skip if dataset not present
    with open(path) as f:
        sample = json.loads(f.readline())
    td = transform_writing_review_demo(sample["input"], sample["expected"])
    # Output should be materially smaller than original
    assert len(td.output_text) < len(sample["expected"]) // 2, (
        f"Output {len(td.output_text)} not smaller than half of "
        f"original {len(sample['expected'])}"
    )
    # Should contain at least one perspective label
    assert any(h in td.output_text for h in
               ["Editorial Critic", "Target Reader", "Tone Analyst",
                "Technical Accuracy", "Fact-Check Summary", "Overall"]), (
        f"No perspective header in output: {td.output_text[:200]!r}"
    )
    assert "[...truncated...]" in td.input_text
    assert td.transformation_applied == "writing_review"


def test_factcheck_transform_condenses_real_sample():
    path = DATASETS / "fact-checks.jsonl"
    if not path.exists():
        return  # skip if dataset not present
    with open(path) as f:
        sample = json.loads(f.readline())
    td = transform_factcheck_demo(sample["input"], sample["expected"])
    assert len(td.output_text) < len(sample["expected"]) // 2, (
        f"Output {len(td.output_text)} not smaller than half of "
        f"original {len(sample['expected'])}"
    )
    assert td.transformation_applied == "fact_check"


def test_writing_review_transform_synthetic():
    """Test with synthetic input to avoid dataset dependency."""
    expected = """## Editorial Critic Review: "Test Post"

The argument structure is strong with clear causal chain.
Evidence is well-cited.
Originality stands out.

## Target Reader Evaluation

Hook: Strong opening with concrete numbers.
Value: Clear practical takeaway.
Engagement: Maintains attention throughout.

## Tone Analyst Review: "Test Post"

**Voice Match Score: 5/7 dimensions matching**

Laconic Style: Matches.
Aphorism Subversion: Strong example in paragraph 3.
Uncomfortable Conclusions: Earned.

## Overall Assessment

Strongest element: The framing.
Weakest element: Conclusion is rushed.
Recommendation: Minor revisions."""

    td = transform_writing_review_demo("a " * 600, expected)
    assert "Editorial Critic" in td.output_text
    assert "Target Reader" in td.output_text
    assert "Tone Analyst" in td.output_text
    assert "Overall Assessment" in td.output_text
    assert "[...truncated...]" in td.input_text


def test_factcheck_transform_synthetic_table():
    """Test with synthetic Claims Extracted table."""
    expected = """# Fact-Check Report

## Claims Extracted

| # | Claim | Type | Para | Verdict | Evidence |
|---|-------|------|------|---------|----------|
| 1 | The 1948 experiment | Historical | 3 | VERIFIED | Skinner 1948 |
| 2 | 40% improvement | Statistical | 5 | INACCURATE | Actual was 30% |
| 3 | Used Python | Technical | 7 | VERIFIED | Source code |

## Summary

- Total claims: 3
- Verified: 2
- Inaccurate: 1
"""
    td = transform_factcheck_demo("a " * 600, expected)
    assert "VERIFIED" in td.output_text
    assert "INACCURATE" in td.output_text
    assert td.transformation_applied == "fact_check"
    assert "[...truncated...]" in td.input_text


def test_severity_transformer_handles_none_severity():
    """Regression test: transform_severity_demo crashed on None severity."""
    from prompt_optimizer.demo_transformers import transform_severity_demo
    # Output that makes extract_severity_details assign severity=None
    output = """
## Code Review

Found some issues that are hard to classify:
- Consider improving error handling
- Variable naming could be clearer

Overall the code is fine.
"""
    # Should NOT crash with AttributeError: 'NoneType' object has no attribute 'capitalize'
    td = transform_severity_demo("some code", output)
    assert td is not None
    assert td.transformation_applied == "severity_classification"


if __name__ == "__main__":
    test_writing_review_transformer_registered()
    print("PASS: writing-review registered")
    test_factcheck_transformer_registered()
    print("PASS: fact-checker registered")
    test_writing_review_transform_condenses_real_sample()
    print("PASS: writing-review condenses real sample")
    test_factcheck_transform_condenses_real_sample()
    print("PASS: fact-check condenses real sample")
    test_writing_review_transform_synthetic()
    print("PASS: writing-review synthetic")
    test_factcheck_transform_synthetic_table()
    print("PASS: fact-check synthetic table")
    test_severity_transformer_handles_none_severity()
    print("PASS: severity handles None")
    print("\nAll demo transformer tests passed!")
