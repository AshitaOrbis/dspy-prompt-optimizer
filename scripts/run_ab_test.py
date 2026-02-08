#!/usr/bin/env python3
"""
Run live A/B test for plan quality directives.

Runs each gold example through Claude CLI twice:
  A) Baseline: plan prompt without quality directives
  B) Treatment: plan prompt with quality directives prepended

Scores both outputs against expected and writes results to a JSONL file.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from prompt_optimizer.metrics import plan_quality_match

TRAINING_DATA = Path("datasets/plan-quality.jsonl")
RESULTS_FILE = Path("tmp/ab-test-results-opus.jsonl")
TIMEOUT = 300
MODEL = "opus"

# The 4 plan-mode directives we're testing (treatment condition)
PLAN_DIRECTIVES = """## Plan Mode Quality Instructions

**CRITICAL**: A plan that requires revision passes is a failed plan. Invest exploration depth upfront.

During plan mode, thoroughness OVERRIDES efficiency:
- Read ALL files relevant to the change, not just the entry point
- Trace dependency chains (imports, callers, tests, configs)
- Identify at minimum: affected files, edge cases, migration needs, test impact

### Quality Gates (verify before presenting plan)

| Gate | Requirement |
|------|-------------|
| **File coverage** | Every file that will be modified is named with its path |
| **Edge cases** | At least 2 edge cases or failure modes are identified |
| **Dependencies** | Upstream and downstream effects are traced |
| **Risks** | Non-obvious risks or gotchas are called out |
| **Verification** | Each step has a concrete way to verify correctness |

### Anti-Patterns to Avoid

- Do NOT use single-sentence plan steps. Each step specifies what changes, where, and why.
- Do NOT say "and other related changes". Name every file.
- Do NOT skip test impact analysis.

This ratio applies to CODE GENERATION only. Plan mode uses the inverse: invest 80% in thorough exploration, 20% in concise presentation."""

BASE_PROMPT_TEMPLATE = """You are creating an implementation plan. Be thorough and specific.

{task_input}

Create a detailed plan with:
- List of files to modify (with full paths)
- Numbered implementation steps (what file, what changes, why)
- Edge cases and failure modes
- Dependency analysis (upstream/downstream effects)
- Risks and gotchas"""

TREATMENT_PROMPT_TEMPLATE = """{directives}

---

You are creating an implementation plan. Be thorough and specific.

{task_input}

Create a detailed plan with:
- List of files to modify (with full paths)
- Numbered implementation steps (what file, what changes, why)
- Edge cases and failure modes
- Dependency analysis (upstream/downstream effects)
- Risks and gotchas"""


def run_claude(prompt: str, model: str = MODEL) -> tuple:
    """Run a prompt through Claude CLI."""
    cmd = [
        "claude", "--print", "--model", model,
        "--dangerously-skip-permissions",
        "--", prompt,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        if result.returncode != 0:
            return False, result.stderr or f"Exit code: {result.returncode}"
        return True, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def main():
    # Load data
    with open(TRAINING_DATA) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    gold_examples = [(i, ex) for i, ex in enumerate(examples)
                     if ex.get("metadata", {}).get("quality") == "gold"]

    print(f"A/B Test: Plan Mode Quality Directives")
    print(f"Model: {MODEL}")
    print(f"Examples: {len(gold_examples)} gold plans")
    print(f"Timeout: {TIMEOUT}s per call")
    print()

    # Clear previous results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.unlink(missing_ok=True)

    results = []

    for idx, (i, ex) in enumerate(gold_examples):
        task_type = ex["metadata"]["task_type"]
        task_input = ex["input"]
        expected = ex["expected"]

        # Baseline prompt (no directives)
        baseline_prompt = BASE_PROMPT_TEMPLATE.format(task_input=task_input)

        # Treatment prompt (with directives)
        treatment_prompt = TREATMENT_PROMPT_TEMPLATE.format(
            directives=PLAN_DIRECTIVES, task_input=task_input
        )

        # Run baseline
        print(f"[{idx+1}/{len(gold_examples)}] {task_type}: baseline...", end=" ", flush=True)
        b_ok, b_output = run_claude(baseline_prompt)
        b_score = plan_quality_match(expected, b_output) if b_ok else 0.0
        b_status = f"{b_score:.3f}" if b_ok else "FAIL"
        print(b_status, end="  ", flush=True)

        time.sleep(1)

        # Run treatment
        print(f"treatment...", end=" ", flush=True)
        t_ok, t_output = run_claude(treatment_prompt)
        t_score = plan_quality_match(expected, t_output) if t_ok else 0.0
        delta = t_score - b_score
        t_status = f"{t_score:.3f}" if t_ok else "FAIL"
        sign = "+" if delta >= 0 else ""
        print(f"{t_status}  delta: {sign}{delta:.3f}")

        result = {
            "index": i,
            "task_type": task_type,
            "baseline_score": round(b_score, 4),
            "treatment_score": round(t_score, 4),
            "baseline_success": b_ok,
            "treatment_success": t_ok,
            "delta": round(delta, 4),
        }
        results.append(result)

        # Write incrementally
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        time.sleep(1)

    # Summary
    both_ok = [r for r in results if r["baseline_success"] and r["treatment_success"]]
    b_scores_all = [r["baseline_score"] for r in results]
    t_scores_all = [r["treatment_score"] for r in results]
    b_scores = [r["baseline_score"] for r in both_ok]
    t_scores = [r["treatment_score"] for r in both_ok]

    b_mean = sum(b_scores) / len(b_scores) if b_scores else 0
    t_mean = sum(t_scores) / len(t_scores) if t_scores else 0
    delta = t_mean - b_mean
    delta_pct = (delta / max(b_mean, 0.001)) * 100

    print(f"\n{'='*60}")
    print(f"A/B TEST RESULTS")
    print(f"{'='*60}")
    print(f"Model:            {MODEL}")
    print(f"Total examples:   {len(results)}")
    print(f"Both succeeded:   {len(both_ok)}")
    print(f"B-only timeout:   {sum(1 for r in results if not r['baseline_success'] and r['treatment_success'])}")
    print(f"T-only timeout:   {sum(1 for r in results if r['baseline_success'] and not r['treatment_success'])}")
    print(f"Both timeout:     {sum(1 for r in results if not r['baseline_success'] and not r['treatment_success'])}")
    print()
    print(f"--- Head-to-head ({len(both_ok)} examples) ---")
    print(f"Baseline mean:    {b_mean:.3f}")
    print(f"Treatment mean:   {t_mean:.3f}")
    print(f"Delta:            {delta:+.3f} ({delta_pct:+.1f}%)")
    print()

    wins = sum(1 for r in both_ok if r["delta"] > 0.02)
    losses = sum(1 for r in both_ok if r["delta"] < -0.02)
    ties = sum(1 for r in both_ok if abs(r["delta"]) <= 0.02)
    print(f"Wins:             {wins}/{len(both_ok)} (treatment better)")
    print(f"Losses:           {losses}/{len(both_ok)} (baseline better)")
    print(f"Ties:             {ties}/{len(both_ok)}")
    print()

    # Per-example detail for head-to-head
    print("--- Per-example detail (both succeeded) ---")
    for r in both_ok:
        sign = "+" if r["delta"] >= 0 else ""
        print(f"  #{r['index']+1:2d} {r['task_type']:10s}  B={r['baseline_score']:.3f}  T={r['treatment_score']:.3f}  {sign}{r['delta']:.3f}")

    if delta_pct > 20:
        print("\nVERDICT: Strong improvement. Directives clearly working.")
    elif delta_pct > 10:
        print("\nVERDICT: Moderate improvement. Directives helping.")
    elif delta_pct > 0:
        print("\nVERDICT: Slight improvement.")
    elif delta_pct > -5:
        print("\nVERDICT: No meaningful difference.")
    else:
        print("\nVERDICT: Regression.")

    print(f"\nDetailed results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
