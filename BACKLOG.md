# DSPy Prompt Optimizer — Backlog

Deferred work surfaced from optimization rounds. Items here are intentionally not blocking and can be picked up when Claude budget and runtime conditions allow. Source reports under `reports/`.

## Status snapshot (2026-04-30)

Last campaign round completed 2026-04-19 (`reports/april-2026-round2.md`). Two genuine improvements deployed across the campaign (`pr-preparer` 0.352 → 0.790, `refactoring-advisor` 0.308 → 0.733, `publication-review-gpt` 0.534 → 0.596). Code-reviewer retried twice in April; both retries gate-restored — the 0.525 baseline holdout is non-reproducible because the holdout set is too small (3 examples). **Holdout expansion is the unblocking item** before any further code-reviewer optimization is worth attempting.

## P1 — Unblocks further optimization

### 1. Audit and expand code-reviewer holdout dataset
- Current holdout: 3 examples → noise dominates the score
- Target: 8–10 examples minimum
- Why this is the gating item: any code-reviewer re-opt produces a holdout score that may swing ±0.18 from random sampling alone, so the gate either always fires (false regression) or never fires (false improvement). Until expanded, further code-reviewer rounds burn Claude budget without producing trustworthy signal.

## P2 — Quality fixes

### 2. Pre-flight gate metric comparison bug
- Currently compares against training `avg_score` instead of holdout-to-holdout
- Source: `reports/april-2026-round2.md` item 3
- Minor but biases the gate decision

### 3. `transform_severity_demo` NoneType crash
- 2/9 demos in the Sonnet code-reviewer rerun triggered `'NoneType' object has no attribute 'capitalize'` on a null `reason` field
- Latent — didn't affect campaign output but will eventually corrupt a run

## P3 — Capacity / scaling

### 4. Codex timeout investigation
- Codex fails on >2K-word inputs even at xhigh effort; affects fact-checker dataset generation and pub-review-gpt runs
- Options: lower effort flag, chunk inputs, or fall back to Claude (already partially done)

### 5. Concurrent dataset-generation conflicts — DONE (2026-06-22, [dspy-1])
- [x] `generate_factcheck_dataset.py` Claude fallbacks fail with `claude returned 1` when GPT/Gemini optimization runs are also active
- Resolved via detect-and-back-off: the `claude -p` fallback now retries up to 3× with exponential backoff (5s, 10s) on transient non-zero exits (timeouts/success are not retried). Chose retry-with-backoff over flock because the failure is transient contention, not a hard mutual-exclusion requirement.

### 6. Fact-checker dataset still thin
- Currently 9 training + 3 holdout (target was 12+4)
- Rerun B1 single-tasked when system is idle

## Strategic / framework-level (from Mar 2026 memory)

These are the broader changes that would lift the whole campaign rather than fixing one target:

- Category-stratified demo selection (force diversity across issue categories)
- Rebalance code-reviewer training data (reduce security-issue overrepresentation)
- Rework `test_coverage_score` metric — current ceiling is 0.47, needs redesign
- Add pre-flight holdout check that aborts save if score regresses (partially done — gate exists but compares wrong metric, see P2 #2)
- **`program.md` directive input** (from `claude-evolution/pipeline/investigations/archived/20260403-autoagent-vs-claude-evolution.md`): accept an optional human-written Markdown "objective description" alongside the numeric metric. The optimizer prepends this to the COPRO/iterative meta-prompt so the optimizer knows *what to focus on* (e.g., "improve tool selection accuracy on edge cases involving ambiguous file types") instead of only the bare metric. Pattern stolen from `kevinrgu/autoagent`; the rest of AutoAgent's architecture (Docker + Harbor benchmarks) is not worth porting.

## Reference

- `reports/april-2026-round2.md` — most recent campaign round, contains "Remaining Work" section that this backlog supersedes
- `reports/regression-investigation.md` — original code-reviewer regression analysis
- `reports/dspy-optimization-summary.md` — campaign summary across all rounds

## 2026-07-16 — deploy script corrupts agents with markdown-heading demos (found during code-reviewer promotion)

`scripts/deploy_optimized_prompts.py` `inject_demos_to_agent()` locates the end of the
old `## Few-Shot Examples` section via `rest.find("\n## ")`. When a demo's fenced
**Output:** block itself contains a `## ` heading (code-reviewer's do: `## Code Review
Summary`), the heuristic stops at the nested heading instead of the real next section —
a naive run leaves stale tail-fragments of the old examples duplicated in the file.
Fix: anchor on the actual next top-level section (track fence state, or require the
known following heading). The 2026-07-16 promotion was done manually anchored on
`## Guidelines` (see ~/.claude/agents/code-reviewer.md.bak-2026-07-16 for the pre-state);
the script itself is still buggy and will corrupt any future promotion run verbatim.

- **2026-08-02 (bq-019):** `tests/test_model_runners.py` has 2 pre-existing failures (gemini preamble-stripping expectations) on BOTH the workspace and public lines, present before and after the deployment-gate fix (verified at HEAD~1). Root cause likely the gemini→agy preamble drift; fix the stripper or the fixtures.
