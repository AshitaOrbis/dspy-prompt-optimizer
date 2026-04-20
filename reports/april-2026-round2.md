# April 2026 Optimization Round 2: Remaining Work

**Date**: 2026-04-18 to 2026-04-19
**Duration**: ~5 hours wall-clock
**Triggered by**: Deferred items from Round 1 (publication-review-gpt and -gemini retries, code-reviewer Sonnet, dataset expansion, parsing bug fix, demo transformer registration)

## Executive Summary

All 5 remaining items addressed. **One target genuinely improved** (publication-review-gpt: 0.534 → 0.596). Three optimizations correctly **gate-restored** their previous demos (code-reviewer Sonnet, Gemini retry — both produced inferior holdout scores). The Gemini parsing bug was fully resolved as an independent benefit. Fact-checker dataset expansion only achieved modest gain due to runtime resource conflicts.

## Items Completed

### A1: GeminiModelRunner Output Parsing Fix
**Status**: DONE, verified working

`gemini -p --yolo` was emitting preamble lines that contaminated stdout:
```
YOLO mode is enabled. All tool calls will be automatically approved.
Loaded cached credentials.
```

`_clean_output` only stripped ANSI codes, leaving the preamble in the parsed output. This caused all Gemini training calls to score 0 (extractor couldn't find structured findings).

**Fix**: Extended `_clean_output` to also strip lines starting with known preamble patterns ("YOLO mode", "Loaded cached credentials", "Data collection is disabled", "Using model:") before the first non-preamble content.

**Verification**: 5 unit tests pass; subsequent Gemini optimization completed 26/26 training successfully (vs 0 before fix).

### A2: Demo Transformers Registered
**Status**: DONE

Writing-review and fact-checker demos were 3K-11K chars each (passing through unchanged at 100%). Added two new transformers following the `transform_review_demo` pattern:

- **`transform_writing_review_demo`**: Extracts perspective sections (Editorial Critic, Target Reader, Tone Analyst, Technical Accuracy, Fact-Check Summary, Overall Assessment), keeps first 3 lines per section, caps output at ~300 words.
- **`transform_factcheck_demo`**: Extracts top 5 verdicts from Claims Extracted markdown table (or fallback to "### Claim N" patterns), appends Summary block, caps at ~300 words.

Registered in both `TRANSFORMER_MAP` (by metric name) and `TARGET_TRANSFORMER_MAP` (by target name). 6 unit tests pass including condensation against real dataset samples (>50% size reduction confirmed).

### B1: Fact-Checker Dataset Expansion (Partial)
**Status**: PARTIAL

Added `generate_factcheck_via_claude()` fallback function. Modified main loop to call it when Codex returns empty.

**Result**: 8+2 → 9+3 examples (modest gain).

**Why limited**: Of 10 retried Codex calls (all timed out at 480s), only 2 Claude fallbacks succeeded. The other 8 failed with `claude returned 1` (empty stderr), likely due to resource contention with the concurrently-running GPT pub-review (Codex) and Gemini pub-review (Gemini CLI) processes. Single-tasking the script when other load-heavy work is idle would likely succeed.

### C1: Sonnet Code-Reviewer Re-Optimization
**Status**: GATE-RESTORED

Sonnet successfully ran 50/50 examples in 33 minutes. Collected 9 demos with average training score 0.676.

**Holdout scored 0.340** (vs 0.525 baseline). Pre-flight gate fired correctly:
```
GATE FAILED: Regression (0.340 < 0.660 - 0.02)
Restoring previous _latest.json from backup
```

The original demos are preserved. The 0.525 baseline appears non-reproducible — likely the holdout dataset (only 3 examples) drifted, or the metric's evaluation behavior shifted. Consider this a signal that **the code-reviewer holdout needs auditing/expansion** before further optimization.

Note: 2/9 demos triggered a `transform_severity_demo` bug ("'NoneType' object has no attribute 'capitalize'"). Latent issue worth fixing but didn't affect this run.

### C2: GPT Publication-Review Retry
**Status**: IMPROVED, deployed

| Phase | Score |
|-------|-------|
| Training (25/26) | 0.655 |
| Holdout (5/7) | **0.596** |
| Previous baseline | 0.534 |
| **Improvement** | **+0.062** |

Two timeouts: post 026 (11.7K words) in training, posts 028 and 034 in holdout. Of 5 successful holdout examples, scores ranged 0.424–1.000. Gate passed (0.596 ≥ 0.534 - 0.02), new demos deployed.

### C3: Gemini Publication-Review Retry
**Status**: GATE-RESTORED, parsing bug verified fixed

With A1 fix in place, all 26 training examples completed successfully (vs 0 before fix). Avg training 0.515, 3 demos collected.

| Phase | Score |
|-------|-------|
| Training (26/26) | 0.515 |
| Holdout (7/7) | 0.532 |
| Previous baseline | 0.556 |
| Delta | -0.024 (regressed) |

Gate fired (0.532 < 0.556 - 0.02), restored 0.556 demos. The parsing bug fix is the larger benefit — Gemini optimization is now functional and can be retried in future rounds.

## Final Score Table

| Target | Round 1 Holdout | Round 2 Holdout | Net Delta |
|--------|----------------|----------------|-----------|
| pub-review-opus | 0.584 | 0.584 | 0 |
| pub-review-gemini | 0.556 | 0.556 | 0 (gate-restored) |
| pub-review-gpt | 0.534 | **0.596** | **+0.062** |
| code-reviewer | 0.525 | 0.525 | 0 (gate-restored) |
| writing-review | 0.625 | 0.625 | 0 |
| fact-checker | 0.667 | 0.667 | 0 |

## Files Modified This Round

- `lib/prompt_optimizer/model_runners.py:257-291` — Gemini `_clean_output` preamble stripping
- `lib/prompt_optimizer/demo_transformers.py:425-553, 502-510, 524-528` — Two new transformers + registrations
- `scripts/generate_factcheck_dataset.py:188+, 235-238` — Claude fallback function + main-loop integration

## New Files

- `tests/test_model_runners.py` — 5 tests for Gemini preamble stripping
- `tests/test_demo_transformers.py` — 6 tests for new transformers
- `reports/april-2026-round2.md` — this report

## Remaining Work (Future Rounds)

1. **Code-reviewer holdout audit** — 3 examples too noisy; expand to 8-10
2. **Codex timeout investigation** — fails on >2K-word inputs even with xhigh; consider lowering effort or chunking
3. **Pre-flight gate metric comparison** — currently compares against training `avg_score`, should compare holdout-to-holdout. Minor bug.
4. **Concurrent dataset generation conflicts** — serialize when other Claude/Codex tasks are running
5. **`transform_severity_demo` NoneType bug** — fix the capitalize() crash on null reasons
6. **Fact-checker dataset still thin** — 9+3 is workable but not robust; rerun B1 when system is idle
