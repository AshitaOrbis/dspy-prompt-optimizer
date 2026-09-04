#!/usr/bin/env python3
"""
Dedicated optimization script for publication-review prompts.

Runs BootstrapFewShot-style optimization for each reviewer model (GPT-5.4,
Opus 4.6, Gemini 3.1 Pro) using review-audit data as training signal.

Supports checkpointing: if interrupted by quota limits, re-run the same
command to continue from where it left off.

Usage:
    python scripts/optimize_publication_review.py --model opus --threshold 0.4
    python scripts/optimize_publication_review.py --model gpt --threshold 0.4
    python scripts/optimize_publication_review.py --model gemini --threshold 0.4
    python scripts/optimize_publication_review.py --self-score
    python scripts/optimize_publication_review.py --model opus --holdout-only
    python scripts/optimize_publication_review.py --model opus --reset  # clear checkpoint
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional, Dict, Tuple

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer import DemoStorage
from prompt_optimizer.bootstrap import (
    HoldoutGateError,
    TrainingExample,
    load_holdout_jsonl,
    promote_candidate_atomic,
    promote_candidate_with_holdout,
)
from prompt_optimizer.storage import OptimizedPrompt, Demo
from prompt_optimizer.metrics import publication_review_match
from prompt_optimizer.model_runners import get_runner_for_model
from prompt_optimizer.format_instructions import get_format_instruction
from prompt_optimizer.demo_transformers import transform_review_demo

# Paths
SKILL_MD_PATH = Path.home() / ".claude" / "skills" / "publication-review" / "SKILL.md"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
STATUS_PATH = Path(__file__).parent.parent / "optimized-prompts" / "status.json"
CHECKPOINT_DIR = Path(__file__).parent.parent / "optimized-prompts" / "checkpoints"

# Model-to-prompt section mapping
MODEL_SECTIONS = {
    "gpt": "GPT-5.4 Prompt",
    "gemini": "Gemini 3.1 Pro Prompt",
    "opus": "Opus 4.6 Prompt",
}

CONSECUTIVE_FAILURE_LIMIT = 3
MAX_RETRIES_PER_EXAMPLE = 3


@dataclass
class Checkpoint:
    """Resumable optimization state."""
    model: str
    threshold: float
    max_demos: int
    completed_indices: List[int] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)  # JSON index -> score
    demos: List[Dict] = field(default_factory=list)  # serialized demo dicts
    failed_indices: List[int] = field(default_factory=list)  # terminal after retry budget
    retry_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = ""

    def save(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = CHECKPOINT_DIR / f"publication-review-{self.model}.json"
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, model: str) -> Optional['Checkpoint']:
        path = CHECKPOINT_DIR / f"publication-review-{model}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def clear(cls, model: str):
        path = CHECKPOINT_DIR / f"publication-review-{model}.json"
        if path.exists():
            path.unlink()


def extract_prompt_section(skill_text: str, model: str) -> str:
    """Extract a model's prompt template from SKILL.md."""
    section_name = MODEL_SECTIONS[model]
    pattern = rf'###\s*{re.escape(section_name)}.*?\n```\n(.*?)\n```'
    match = re.search(pattern, skill_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = rf'###\s*{re.escape(section_name)}.*?\n(.*?)(?=\n###|\Z)'
    match = re.search(pattern, skill_text, re.DOTALL)
    if match:
        code_match = re.search(r'```\n(.*?)\n```', match.group(1), re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

    raise ValueError(f"Could not find prompt section '{section_name}' in SKILL.md")


def load_training_data(model: str) -> list:
    path = DATASETS_DIR / f"publication-review-{model}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append(TrainingExample(
                    input_text=data["input"],
                    expected_output=data["expected"],
                    metadata=data.get("metadata"),
                ))
    return examples


def load_holdout_data(model: str, *, required: bool = False) -> list:
    path = DATASETS_DIR / f"publication-review-{model}-holdout.jsonl"
    if not required and not path.exists():
        return []
    return load_holdout_jsonl(path)


def update_status(model: str, result: dict):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as f:
            status = json.load(f)
    else:
        status = {}
    target_name = f"publication-review-{model}"
    status[target_name] = {
        "optimized": True,
        "algorithm": "bootstrap",
        "optimized_with_model": model,
        "training_score": result.get("training_score"),
        "holdout_score": result.get("holdout_score"),
        "demos_selected": result.get("demos_selected", 0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    payload = json.dumps(status, indent=2).encode("utf-8")
    temporary_path = STATUS_PATH.with_name(f"{STATUS_PATH.name}.tmp-{os.getpid()}")
    try:
        with open(temporary_path, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, STATUS_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_training_with_checkpoints(
    model: str,
    runner,
    base_prompt: str,
    training_data: list,
    format_instruction: str,
    threshold: float,
    max_demos: int,
    verbose: bool,
) -> Tuple[List[Dict], Dict[str, float], float]:
    """Run training phase with checkpoint support.

    Returns (demos, scores_dict, avg_score).
    Saves checkpoint after each successful example.
    Stops early after CONSECUTIVE_FAILURE_LIMIT consecutive failures. A failed
    example remains pending until its bounded retry budget is exhausted.
    """
    # Load or create checkpoint
    ckpt = Checkpoint.load(model)
    if ckpt and ckpt.threshold == threshold and ckpt.max_demos == max_demos:
        if verbose:
            done = len(ckpt.completed_indices)
            print(f"  Resuming from checkpoint: {done}/{len(training_data)} completed, "
                  f"{len(ckpt.demos)} demos collected")
    else:
        if ckpt:
            print(f"  Checkpoint exists but params changed (threshold/max_demos). Starting fresh.")
        ckpt = Checkpoint(model=model, threshold=threshold, max_demos=max_demos)

    full_prompt = base_prompt + "\n\n" + format_instruction

    # Old checkpoints recorded every failure as terminal. Migrate those entries
    # to retryable state so the first resumed run actually retries them.
    for failed_index in list(ckpt.failed_indices):
        key = str(failed_index)
        if key not in ckpt.retry_state:
            ckpt.retry_state[key] = {
                "attempts": 1,
                "error_class": "LegacyTransientFailure",
                "pending": True,
            }
            ckpt.failed_indices.remove(failed_index)

    consecutive_failures = 0
    for i, ex in enumerate(training_data):
        if i in ckpt.completed_indices:
            continue
        retry = ckpt.retry_state.get(str(i), {})
        if i in ckpt.failed_indices or int(retry.get("attempts", 0)) >= MAX_RETRIES_PER_EXAMPLE:
            if i not in ckpt.failed_indices:
                ckpt.failed_indices.append(i)
            continue

        slug = ex.metadata.get("post_slug", f"example-{i}") if ex.metadata else f"example-{i}"

        try:
            result = runner.run(full_prompt + "\n\n" + ex.input_text)
        except Exception as exc:
            result = None
            failure_error = str(exc)
            failure_class = type(exc).__name__
        else:
            failure_error = getattr(result, "error", "missing runner result")
            failure_class = "RunnerFailure"

        if result is None or not getattr(result, "success", False):
            attempts = int(retry.get("attempts", 0)) + 1
            pending = attempts < MAX_RETRIES_PER_EXAMPLE
            ckpt.retry_state[str(i)] = {
                "attempts": attempts,
                "error_class": failure_class,
                "error": failure_error,
                "pending": pending,
            }
            ckpt.scores[str(i)] = 0.0
            if not pending and i not in ckpt.failed_indices:
                ckpt.failed_indices.append(i)
            consecutive_failures += 1
            if verbose:
                retry_label = "pending retry" if pending else "retry budget exhausted"
                print(
                    f"  [{i+1}/{len(training_data)}] {slug}: FAILED "
                    f"({failure_error}; {retry_label} {attempts}/{MAX_RETRIES_PER_EXAMPLE})"
                )

            ckpt.save()

            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                print(f"\n  {CONSECUTIVE_FAILURE_LIMIT} consecutive failures — likely quota exhausted.")
                print(f"  Checkpoint saved. Re-run the same command to continue.")
                ckpt.save()
                break
            continue

        consecutive_failures = 0
        score = publication_review_match(ex.expected_output, result.output)
        ckpt.scores[str(i)] = score  # JSON keys must be strings
        if i not in ckpt.completed_indices:
            ckpt.completed_indices.append(i)
        if i in ckpt.failed_indices:
            ckpt.failed_indices.remove(i)
        ckpt.retry_state.pop(str(i), None)

        if score >= threshold and len(ckpt.demos) < max_demos:
            # Transform demo for compactness
            transformed = transform_review_demo(ex.input_text, result.output, ex.metadata)
            demo_dict = {
                "input_text": transformed.input_text,
                "output_text": transformed.output_text,
                "score": score,
                "metadata": ex.metadata,
            }
            ckpt.demos.append(demo_dict)
            if verbose:
                print(f"  [{i+1}/{len(training_data)}] {slug}: {score:.3f} [DEMO {len(ckpt.demos)}/{max_demos}]")
        else:
            if verbose:
                print(f"  [{i+1}/{len(training_data)}] {slug}: {score:.3f}")

        # Save checkpoint after each success
        ckpt.save()

    # Failed attempts remain explicit zeros until a successful retry replaces
    # them, so outages cannot silently shrink the denominator.
    scored_values = [float(v) for v in ckpt.scores.values()]
    avg_score = sum(scored_values) / len(scored_values) if scored_values else 0.0

    return ckpt.demos[:max_demos], ckpt.scores, avg_score


def build_optimized_prompt(
    base_prompt: str,
    demos: List[Dict],
    format_instruction: str,
    threshold: float,
    avg_score: float,
) -> OptimizedPrompt:
    """Construct an OptimizedPrompt from collected demos."""
    demo_objects = []
    for d in demos:
        demo_objects.append(Demo(
            input_text=d["input_text"],
            output_text=d["output_text"],
            score=d["score"],
            metadata=d.get("metadata"),
        ))

    return OptimizedPrompt(
        base_prompt=base_prompt,
        demos=demo_objects,
        optimization_date=time.strftime("%Y-%m-%dT%H:%M:%S"),
        metric_name="publication_review_match",
        threshold=threshold,
        avg_score=avg_score,
        metadata={"algorithm": "bootstrap"},
        format_instruction=format_instruction,
    )


def run_holdout(model: str, runner, storage: DemoStorage, threshold: float, verbose: bool) -> Optional[float]:
    """Run a complete holdout evaluation using the saved optimized prompt."""
    try:
        opt = storage.load_optimized_prompt(f"publication-review-{model}")
    except Exception as exc:
        print(f"  Optimized prompt is unreadable for {model}: {exc}")
        return None
    if opt is None:
        print(f"  No optimized prompt found for {model}. Run training first.")
        return None

    try:
        holdout_data = load_holdout_data(model, required=True)
    except HoldoutGateError as exc:
        print(f"  Holdout unavailable for {model}: {exc}")
        return None

    full_prompt = opt.to_prompt()
    if verbose:
        print(f"  Optimized prompt: {len(full_prompt)} chars, {len(opt.demos)} demos")
        print(f"  Holdout examples: {len(holdout_data)}")

    scores = []
    for ex in holdout_data:
        slug = ex.metadata.get("post_slug", "?") if ex.metadata else "?"
        try:
            result = runner.run(full_prompt + "\n\n" + ex.input_text)
        except Exception as exc:
            print(f"  {slug}: FAILED ({type(exc).__name__}: {exc})")
            return None
        if result is None or not result.success:
            print(f"  {slug}: FAILED ({getattr(result, 'error', 'missing result')})")
            return None
        score = publication_review_match(ex.expected_output, result.output)
        if score is None:
            print(f"  {slug}: FAILED (metric returned None)")
            return None
        scores.append(float(score))
        if verbose:
            print(f"  {slug}: {score:.3f}")

    if len(scores) != len(holdout_data):
        print(f"  Holdout coverage incomplete: {len(scores)}/{len(holdout_data)}")
        return None
    avg = sum(scores) / len(scores)
    print(f"\n  Holdout average: {avg:.3f}")
    passed = avg >= threshold
    print(f"  {'PASS' if passed else 'FAIL'} (threshold: {threshold})")
    return avg


def main():
    parser = argparse.ArgumentParser(
        description="Optimize publication-review prompts (with checkpoint support)",
    )
    parser.add_argument("--model", choices=["gpt", "opus", "gemini", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--max-demos", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Score base prompt on first 3 examples (real API calls)")
    parser.add_argument("--self-score", action="store_true",
                        help="Fast pipeline validation (no API calls)")
    parser.add_argument("--holdout-only", action="store_true",
                        help="Run holdout evaluation using saved optimized prompt")
    parser.add_argument("--reset", action="store_true",
                        help="Clear checkpoint for model(s) and start fresh")
    parser.add_argument("--holdout-gate", action="store_true",
                        help="Compare holdout score against previous before keeping _latest.json")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet and args.verbose

    skill_text = SKILL_MD_PATH.read_text()
    models = ["gpt", "opus", "gemini"] if args.model == "all" else [args.model]
    format_instruction = get_format_instruction("publication_review")
    storage = DemoStorage()
    exit_code = 0

    for model in models:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  publication-review-{model}")
            print(f"{'='*60}")

        # Reset checkpoint if requested
        if args.reset:
            Checkpoint.clear(model)
            print(f"  Checkpoint cleared for {model}")
            continue

        try:
            base_prompt = extract_prompt_section(skill_text, model)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            exit_code = 1
            continue

        try:
            training_data = load_training_data(model)
        except Exception as exc:
            print(f"Error loading training data for {model}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        if verbose:
            print(f"Base prompt: {len(base_prompt)} chars, Training: {len(training_data)} examples")

        # --- Self-score mode ---
        if args.self_score:
            scores = [publication_review_match(ex.expected_output, ex.expected_output) for ex in training_data]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  Self-score: {avg:.3f} ({len(scores)} examples)")
            holdout = load_holdout_data(model)
            if holdout:
                h = [publication_review_match(ex.expected_output, ex.expected_output) for ex in holdout]
                print(f"  Holdout self-score: {sum(h)/len(h):.3f} ({len(h)} examples)")
            continue

        runner = get_runner_for_model(model)

        # --- Dry-run mode ---
        if args.dry_run:
            print(f"\n[DRY RUN] Scoring base prompt on first 3 examples...")
            scores = []
            for ex in training_data[:3]:
                result = runner.run(base_prompt + "\n\n" + format_instruction + "\n\n" + ex.input_text)
                slug = ex.metadata.get("post_slug", "?") if ex.metadata else "?"
                if result.success:
                    score = publication_review_match(ex.expected_output, result.output)
                    scores.append(score)
                    print(f"  {slug}: {score:.3f}")
                else:
                    print(f"  {slug}: FAILED ({result.error})")
            if scores:
                print(f"\n  Average: {sum(scores)/len(scores):.3f}")
            continue

        # --- Holdout-only mode ---
        if args.holdout_only:
            print(f"\n[HOLDOUT] Evaluating saved optimized prompt...")
            holdout_score = run_holdout(model, runner, storage, args.threshold, verbose)
            if holdout_score is not None:
                update_status(model, {
                    "training_score": None,
                    "holdout_score": holdout_score,
                    "demos_selected": None,
                })
            else:
                exit_code = 1
            continue

        # --- Full optimization with checkpoints ---
        print(f"\nTraining phase (checkpoint-enabled)...")
        print(f"  Threshold: {args.threshold}, Max demos: {args.max_demos}")

        demos, scores_dict, avg_score = run_training_with_checkpoints(
            model=model,
            runner=runner,
            base_prompt=base_prompt,
            training_data=training_data,
            format_instruction=format_instruction,
            threshold=args.threshold,
            max_demos=args.max_demos,
            verbose=verbose,
        )

        checkpoint = Checkpoint.load(model)
        n_completed = (
            len(set(checkpoint.completed_indices))
            if checkpoint is not None
            else len(scores_dict)
        )
        n_total = len(training_data)
        print(f"\n  Completed: {n_completed}/{n_total}")
        print(f"  Demos collected: {len(demos)}")
        print(f"  Avg score (failed attempts included as zero): {avg_score:.3f}")

        if n_completed != n_total or (checkpoint and checkpoint.failed_indices):
            terminal = len(checkpoint.failed_indices) if checkpoint else 0
            print(
                f"  INCOMPLETE COVERAGE: {n_completed}/{n_total} completed, "
                f"{terminal} exhausted retry budgets. Re-run to retry pending examples.",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        if not demos:
            print(f"  No demos above threshold {args.threshold}. "
                  f"Consider lowering --threshold or re-running for more examples.")
            exit_code = 1
            continue

        target_name = f"publication-review-{model}"
        opt_prompt = build_optimized_prompt(
            base_prompt=base_prompt,
            demos=demos,
            format_instruction=format_instruction,
            threshold=args.threshold,
            avg_score=avg_score,
        )

        status_updated = False
        try:
            if args.holdout_gate:
                holdout_data = load_holdout_data(model, required=True)

                def commit_status(new_score: float, _existing_score: float) -> None:
                    update_status(model, {
                        "training_score": avg_score,
                        "holdout_score": new_score,
                        "demos_selected": len(demos),
                    })

                gate_result = promote_candidate_with_holdout(
                    agent_name=target_name,
                    candidate=opt_prompt,
                    holdout_data=holdout_data,
                    metric_fn=publication_review_match,
                    runner=runner,
                    storage=storage,
                    min_improvement=-0.02,
                    verbose=verbose,
                    on_promoted=commit_status,
                )
                holdout_score = gate_result.new_score
                status_updated = True
                print(
                    f"  Holdout gate passed and promoted atomically: "
                    f"{gate_result.new_score:.3f} vs {gate_result.existing_score:.3f}"
                )
            else:
                promote_candidate_atomic(target_name, opt_prompt, storage)
                print(f"  Saved optimized prompt ({len(demos)} demos)")
                print(f"\nHoldout phase (advisory; --holdout-gate not requested)...")
                holdout_score = run_holdout(model, runner, storage, args.threshold, verbose)
        except Exception as exc:
            print(
                f"  HOLDOUT GATE FAILED for {target_name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        if not status_updated:
            update_status(model, {
                "training_score": avg_score,
                "holdout_score": holdout_score,
                "demos_selected": len(demos),
            })

        # A complete training run whose candidate was promoted can be cleared.
        Checkpoint.clear(model)
        print(f"  Checkpoint cleared (optimization complete)")

    print("\nDone.")
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
