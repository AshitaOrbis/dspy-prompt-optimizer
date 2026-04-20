# Verification Report

**Generated**: 2026-04-15T00:04:31.349875
**Agents Verified**: 16

## Summary

| Metric | Value |
|--------|-------|
| Holdout Passed | 6 |
| Holdout Failed | 9 |
| Regressions | 7 |
| Avg Holdout Score | 0.661 |

## Holdout Results

| Agent | Score | Size | Status |
|-------|-------|------|--------|
| code-reviewer | 0.402 | 3 | FAIL |
| test-writer | 0.400 | 2 | FAIL |
| security-auditor | 0.590 | 2 | FAIL |
| performance-analyzer | 0.595 | 2 | FAIL |
| capability-evaluator | 1.000 | 2 | PASS |
| debugger | 0.833 | 6 | PASS |
| pr-preparer | 0.820 | 6 | PASS |
| feature-implementer | 0.457 | 6 | FAIL |
| refactoring-advisor | 0.693 | 6 | FAIL |
| mcp-search-framework | 1.000 | 6 | PASS |
| mgrep-guide | 1.000 | 4 | PASS |
| advanced-tool-use | 0.800 | 5 | PASS |
| publication-review-gpt | 0.593 | 7 | FAIL |
| publication-review-gemini | 0.120 | 7 | FAIL |
| publication-review-opus | 0.613 | 7 | FAIL |

## Regression Check

| Agent | Baseline | Current | Change | Status |
|-------|----------|---------|--------|--------|
| code-reviewer | 0.525 | 0.402 | -0.123 | REGRESSED |
| test-writer | 0.608 | 0.400 | -0.208 | REGRESSED |
| security-auditor | 0.760 | 0.590 | -0.170 | REGRESSED |
| performance-analyzer | 0.748 | 0.595 | -0.153 | REGRESSED |
| capability-evaluator | 1.000 | 1.000 | +0.000 | OK |
| debugger | 0.933 | 0.833 | -0.100 | REGRESSED |
| pr-preparer | 0.790 | 0.820 | +0.030 | OK |
| feature-implementer | 0.565 | 0.457 | -0.108 | REGRESSED |
| refactoring-advisor | 0.733 | 0.693 | -0.040 | OK |
| mcp-search-framework | 1.000 | 1.000 | +0.000 | OK |
| mgrep-guide | 1.000 | 1.000 | +0.000 | OK |
| advanced-tool-use | 0.000 | 0.800 | +0.800 | OK |
| publication-review-gpt | 0.533 | 0.593 | +0.060 | OK |
| publication-review-gemini | 0.532 | 0.120 | -0.412 | REGRESSED |
| publication-review-opus | 0.622 | 0.613 | -0.009 | OK |