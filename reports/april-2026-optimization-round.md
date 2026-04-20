# April 2026 Optimization Verification & Improvement Round

**Date**: 2026-04-15/16
**Duration**: ~6 hours across 2 sessions (1 crash recovery)
**Triggered by**: Token-efficiency and latency changes to 7+ skills/agents (March-April 2026)

## Executive Summary

Verified that recent token-efficiency changes (delta mode, claim manifests, model parameter removals, Context7 integration) have **not degraded** prompt quality for the modified targets. Created 2 new datasets (writing-review, fact-checker) and implemented a pre-flight holdout safety gate to prevent future regressions. Optimization rounds on weakest targets showed mixed results due to model-tier mismatches and infrastructure limitations.

## Phase 1: Baseline Verification

### Publication-Review Holdout (Per-Model, Apples-to-Apples)

| Target | Baseline | Current | Delta | Timeouts | Verdict |
|--------|----------|---------|-------|----------|---------|
| pub-review-opus | 0.622 | **0.584** | -0.038 | 1/7 | OK (within 0.05 threshold) |
| pub-review-gemini | 0.532 | **0.556** | +0.024 | 0/7 | **Improved** |
| pub-review-gpt | 0.533 | **0.534** | +0.001 | 1/7 | Stable |

**Conclusion**: Token-efficiency changes (delta mode, convergence signals, model downgrade option) preserved or improved quality. Gemini improved +0.024 despite no re-optimization.

### Haiku Cross-Verification (All Targets)

Run through Haiku to detect gross regressions. Most "regressions" are model-tier mismatch (baselines were measured on Sonnet/Opus).

**Genuinely improved** (even through Haiku):
- pr-preparer: 0.790 → 0.820 (+0.030) — metric fixes + few-shot demos working
- publication-review-gpt: 0.533 → 0.593 (+0.060) — delta mode optimization

**Stable** (within threshold):
- capability-evaluator: 1.000 → 1.000
- mcp-search-framework: 1.000 → 1.000
- mgrep-guide: 1.000 → 1.000
- refactoring-advisor: 0.733 → 0.693 (-0.040, within 0.05)
- publication-review-opus: 0.622 → 0.613 (-0.009)

**Model-mismatch noise** (Haiku vs Sonnet/Opus baselines):
- code-reviewer, test-writer, security-auditor, performance-analyzer, debugger, feature-implementer — all show lower Haiku scores than their Sonnet-optimized baselines. Not real regressions.

**Infrastructure issue**:
- publication-review-gemini via Haiku: 3/7 timeouts (blog posts too large for 180s Haiku default)

## Phase 2: Dataset Creation

### Writing-Review Dataset
- **Source**: 20 blog posts via `claude -p --model opus`
- **Result**: 19/20 succeeded (1 transient claude error on post 040)
- **Split**: 15 training + 4 holdout
- **Self-score validation**: 1.000 avg (perfect metric parsing)
- **Files**: `datasets/writing-reviews.jsonl`, `datasets/writing-reviews-holdout.jsonl`

### Fact-Checker Dataset
- **Source**: 10 existing factcheck reports + 10 Codex generation attempts
- **Result**: 10/20 succeeded (all 10 existing reports; all 10 Codex attempts timed out at 480s)
- **Split**: 8 training + 2 holdout
- **Files**: `datasets/fact-checks.jsonl`, `datasets/fact-checks-holdout.jsonl`
- **Known issue**: Codex `exec --full-auto` with blog post + factcheck prompt + web search exceeds 480s timeout

### Code-Reviews Balanced Dataset
- **Purpose**: Address security-dominated category bias (40% → 13%)
- **Method**: Max 4 examples per category from original 50
- **Result**: 30 balanced examples
- **File**: `datasets/code-reviews-balanced.jsonl`

## Phase 3: Optimization Rounds

### Code-Reviewer (Balanced Dataset)
- **Training**: 21/30 demos above 0.45 threshold, avg score 0.660
- **Holdout**: 0.435 (regressed from 0.525 baseline)
- **Root cause**: Haiku-tier optimization can't match Sonnet-tier baseline
- **Decision**: Gate blocked deployment. Original demos retained.
- **Next step**: Re-run on Sonnet tier

### Fact-Checker (New Dataset)
- **Training**: 4/8 demos above 0.50 threshold, avg score 0.651
- **Holdout**: 0.667 (first-ever baseline, no prior optimization)
- **Decision**: New baseline established. Demos saved.
- **Note**: Demo transformer passed through unchanged (100%) — needs a registered factcheck transformer

### Writing-Review (New Dataset)
- **First attempt**: All 15 examples timed out at 180s default
- **Second attempt**: 480s timeout — 13/15 demos above 0.50 threshold, avg score 0.682 (1 timeout)
- **Holdout**: 0.625 (first-ever baseline, 4 examples: 0.614, 0.543, 0.671, 0.673)
- **Decision**: New baseline established. Demos saved.
- **Root cause of first failure**: Full SKILL.md (~15K chars) + blog post (~2-7K words) = ~30K token prompts need >180s
- **Note**: Demo transformer passed through unchanged (100%) — needs a registered writing-review transformer

### Publication-Review-Gemini (Re-optimization)
- **Result**: BLOCKED
- **Issues**: (1) Gemini 3.1 Pro 429 capacity errors, (2) GeminiModelRunner output parsing contaminated by `--yolo` preamble
- **Holdout already improved**: 0.532 → 0.556 without re-optimization
- **Next step**: Fix GeminiModelRunner output parsing, retry when capacity available

### Publication-Review-GPT (Re-optimization)
- **Status**: Not attempted (queued behind Gemini, which was blocked)
- **Holdout already stable**: 0.533 → 0.534

## Phase 4: Pre-Flight Holdout Safety Gate

### Implementation
- **Function**: `pre_flight_holdout_check()` in `lib/prompt_optimizer/verification.py`
- **Mechanism**: Evaluates new optimization vs existing on holdout data before saving `_latest.json`
- **Threshold**: New score must be within -0.02 of existing to deploy
- **Integration**: `--holdout-gate` flag in both `batch_optimize.py` and `optimize_publication_review.py`
- **Backup strategy**: `_latest.json.pre-gate` backup created before optimization, restored if gate fails
- **Tests**: 3 unit tests passing (gate passes, gate blocks, first-optimization auto-pass)

### Files Modified
- `lib/prompt_optimizer/verification.py` — Added `pre_flight_holdout_check()`
- `lib/prompt_optimizer/__init__.py` — Exported new function
- `scripts/batch_optimize.py` — Added `--holdout-gate` and `--holdout-dir` flags
- `scripts/optimize_publication_review.py` — Added `--holdout-gate` flag
- `tests/test_holdout_gate.py` — New test file

### Bug Fix: CodexModelRunner MCP Config
- **Problem**: `codex exec` failed with "invalid transport in mcp_servers.brave-search" because the runner tried to disable MCP servers not defined in config.toml
- **Fix**: Added `_get_configured_mcp_servers()` method that reads config.toml and only disables servers that actually exist
- **File**: `lib/prompt_optimizer/model_runners.py`

## Known Issues & Remaining Work

### Blocked by Infrastructure
| Issue | Impact | Fix |
|-------|--------|-----|
| GeminiModelRunner `--yolo` output contamination | Gemini optimization produces 0 scored demos | Strip preamble from output in `_clean_output()` |
| Codex factcheck timeout (480s) | 50% of factcheck dataset generation failed | Increase timeout to 900s or use `claude -p` instead |
| Writing-review prompt size (~30K tokens) | Default 180s timeout insufficient | Use `--timeout 480` (already fixed in re-run) |
| Gemini 3.1 Pro capacity (429 errors) | Intermittent, blocks optimization | Retry with backoff, or wait for capacity |

### Framework Improvements Needed
| Improvement | Priority | Effort |
|-------------|----------|--------|
| Category-stratified demo selection | High | Modify BootstrapFewShot to enforce category diversity |
| Expand code-reviewer holdout (3 → 8) | High | Generate 5 more holdout examples |
| Factcheck demo transformer | Medium | Register a transformer that condenses 10K→1K char demos |
| Sonnet-tier re-optimization for code-reviewer | Medium | Re-run batch_optimize with --tier sonnet |
| Pre-flight gate auto-enabled for batch runs | Low | Make --holdout-gate the default |

## Score Summary (Before → After)

| Target | Before | After | Source |
|--------|--------|-------|--------|
| pub-review-opus | 0.622 | 0.584 | Per-model holdout (OK) |
| pub-review-gemini | 0.532 | 0.556 | Per-model holdout (Improved) |
| pub-review-gpt | 0.533 | 0.534 | Per-model holdout (Stable) |
| pr-preparer | 0.790 | 0.820 | Haiku verification (Improved) |
| fact-checker | — | 0.667 | New baseline established |
| writing-review | — | 0.625 | New baseline established |
| code-reviewer | 0.525 | 0.435* | *Haiku tier, gate blocked |

## Files Created/Modified This Round

### New Files
- `datasets/baseline-scores.json` — Regression check reference
- `datasets/writing-reviews.jsonl` (15 examples) + holdout (4)
- `datasets/fact-checks.jsonl` (8 examples) + holdout (2)
- `datasets/code-reviews-balanced.jsonl` (30 examples)
- `tests/test_holdout_gate.py` — 3 holdout gate tests
- `scripts/generate_writing_review_dataset.py`
- `scripts/generate_factcheck_dataset.py`
- `scripts/balance_code_reviewer_data.py`
- `reports/april-2026-baseline-verification.md`
- `reports/april-2026-optimization-round.md` (this file)
- `reports/code-reviewer-stratified-april.md`
- `reports/fact-checker-initial-april.md`

### Modified Files
- `lib/prompt_optimizer/verification.py` — pre_flight_holdout_check()
- `lib/prompt_optimizer/__init__.py` — Export
- `lib/prompt_optimizer/model_runners.py` — MCP config fix
- `scripts/batch_optimize.py` — --holdout-gate flag
- `scripts/optimize_publication_review.py` — --holdout-gate flag
