#!/usr/bin/env python3
"""
Batch optimization script for processing multiple agents and skills.

Run BootstrapFewShot optimization on multiple targets in one command.

Usage:
    # Optimize multiple agents
    python batch_optimize.py --agents code-reviewer,test-writer,security-auditor

    # Optimize multiple skills
    python batch_optimize.py --skills mcp-search-framework,mgrep-guide

    # Mixed batch
    python batch_optimize.py --agents code-reviewer --skills mgrep-guide

    # Parallel mode (faster, but may hit rate limits)
    python batch_optimize.py --agents code-reviewer,test-writer --parallel

Examples:
    # Full agent optimization suite
    python batch_optimize.py --agents code-reviewer,test-writer,security-auditor,performance-analyzer

    # Save report to file
    python batch_optimize.py --agents code-reviewer,test-writer --report reports/batch-results.md
"""

import argparse
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer.validation import validate_name, contained_path
from prompt_optimizer import (
    ClaudeRunner,
    DemoStorage,
    exact_match,
    score_similarity,
    numeric_score_match,
    evaluation_score_metric,
    # Agent-specific metrics
    issue_severity_match,
    test_coverage_score,
    complexity_classification,
    security_cwe_match,
    # Tier 1 agent metrics
    pr_quality_match,
    refactoring_match,
    implementation_completeness,
    root_cause_match,
    # Tier 2 agent metrics
    discovery_quality_match,
    fact_check_quality_match,
    research_quality_match,
    # Skill-specific metrics
    routing_accuracy,
    binary_decision_match,
    tool_tier_classification,
    # Tier 3 skill metrics
    writing_review_quality_match,
    handoff_quality_match,
    # Publication review metric
    publication_review_match,
    # Phase 3: COPRO and Iterative
    COPROOptimizer,
    IterativeOptimizer,
    BootstrapFewShot,
)
# Phase 6: Format instructions and demo transformers
from prompt_optimizer.format_instructions import (
    get_format_instruction,
    get_format_type_for_target,
)
from prompt_optimizer.demo_transformers import get_demo_transformer
from prompt_optimizer.batch import (
    BatchTarget,
    optimize_batch_sequential,
    optimize_batch_parallel,
    generate_batch_report,
)
from prompt_optimizer.verification import pre_flight_holdout_check
from prompt_optimizer.bootstrap import TrainingExample as HoldoutExample


# Default metric mappings
AGENT_METRICS = {
    "code-reviewer": issue_severity_match,
    "test-writer": test_coverage_score,
    "security-auditor": security_cwe_match,
    "performance-analyzer": complexity_classification,
    "capability-evaluator": evaluation_score_metric(threshold=70.0),
    # Tier 1
    "pr-preparer": pr_quality_match,
    "refactoring-advisor": refactoring_match,
    "feature-implementer": implementation_completeness,
    "debugger": root_cause_match,
    # Tier 2
    "capability-discoverer": discovery_quality_match,
    "fact-checker": fact_check_quality_match,
    "web-researcher": research_quality_match,
}

SKILL_METRICS = {
    "mcp-search-framework": routing_accuracy,
    "mgrep-guide": binary_decision_match,
    "advanced-tool-use": tool_tier_classification,
    "dispatching-parallel-agents": binary_decision_match,
    # Tier 3
    "writing-review": writing_review_quality_match,
    "session-handoff": handoff_quality_match,
    # Publication review (per-model targets — use dedicated script for optimization)
    "publication-review-gpt": publication_review_match,
    "publication-review-gemini": publication_review_match,
    "publication-review-opus": publication_review_match,
}

# Default training data paths (relative to data_dir)
AGENT_DATA_PATHS = {
    "code-reviewer": "code-reviews.jsonl",
    "test-writer": "test-suites.jsonl",
    "security-auditor": "security-audits.jsonl",
    "performance-analyzer": "performance-analysis.jsonl",
    "capability-evaluator": "evaluations.jsonl",
    # Tier 1
    "pr-preparer": "pr-preparations.jsonl",
    "refactoring-advisor": "refactoring-decisions.jsonl",
    "feature-implementer": "feature-implementations.jsonl",
    "debugger": "debugging-sessions.jsonl",
    # Tier 2
    "capability-discoverer": "discoveries.jsonl",
    "fact-checker": "fact-checks.jsonl",
    "web-researcher": "web-research.jsonl",
}

SKILL_DATA_PATHS = {
    "mcp-search-framework": "search-routing.jsonl",
    "mgrep-guide": "search-decisions.jsonl",
    "advanced-tool-use": "tool-selection.jsonl",
    # Tier 3
    "writing-review": "writing-reviews.jsonl",
    "session-handoff": "session-handoffs.jsonl",
    # Publication review (per-model targets)
    "publication-review-gpt": "publication-review-gpt.jsonl",
    "publication-review-gemini": "publication-review-gemini.jsonl",
    "publication-review-opus": "publication-review-opus.jsonl",
}


def find_agent_path(agent_name: str) -> Path:
    """Find agent file in ~/.claude/agents/ (name-validated, path-contained)."""
    validate_name(agent_name, kind="agent name")
    global_path = contained_path(Path.home() / ".claude" / "agents", f"{agent_name}.md")
    if global_path.exists():
        return global_path

    project_path = contained_path(Path.cwd() / ".claude" / "agents", f"{agent_name}.md")
    if project_path.exists():
        return project_path

    raise FileNotFoundError(f"Agent not found: {agent_name}")


def find_skill_path(skill_name: str) -> Path:
    """Find skill file in ~/.claude/skills/ (name-validated, path-contained)."""
    validate_name(skill_name, kind="skill name")
    global_path = contained_path(Path.home() / ".claude" / "skills", skill_name, "SKILL.md")
    if global_path.exists():
        return global_path

    project_path = contained_path(Path.cwd() / ".claude" / "skills", skill_name, "SKILL.md")
    if project_path.exists():
        return project_path

    alt_global = contained_path(Path.home() / ".claude" / "skills", f"{skill_name}.md")
    if alt_global.exists():
        return alt_global

    raise FileNotFoundError(f"Skill not found: {skill_name}")


def build_agent_target(
    agent_name: str,
    data_dir: Path,
    threshold: float,
    max_demos: int,
    use_transformers: bool = True,
) -> BatchTarget:
    """Build a BatchTarget for an agent."""
    prompt_path = find_agent_path(agent_name)

    # Find training data
    if agent_name in AGENT_DATA_PATHS:
        data_path = data_dir / AGENT_DATA_PATHS[agent_name]
    else:
        # Try default naming convention
        data_path = data_dir / f"{agent_name}.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Get metric
    metric_fn = AGENT_METRICS.get(agent_name, score_similarity)

    # Get transformer config if enabled
    demo_transformer = None
    format_instruction = ""
    if use_transformers:
        format_type = get_format_type_for_target(agent_name)
        format_instruction = get_format_instruction(format_type)
        demo_transformer = get_demo_transformer(agent_name)

    return BatchTarget(
        name=agent_name,
        prompt_path=prompt_path,
        training_data_path=data_path,
        metric_fn=metric_fn,
        threshold=threshold,
        max_demos=max_demos,
        demo_transformer=demo_transformer,
        format_instruction=format_instruction,
    )


def build_skill_target(
    skill_name: str,
    data_dir: Path,
    threshold: float,
    max_demos: int,
    use_transformers: bool = True,
) -> BatchTarget:
    """Build a BatchTarget for a skill."""
    prompt_path = find_skill_path(skill_name)

    # Find training data
    if skill_name in SKILL_DATA_PATHS:
        data_path = data_dir / SKILL_DATA_PATHS[skill_name]
    else:
        # Try default naming convention
        data_path = data_dir / f"{skill_name}.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Get metric
    metric_fn = SKILL_METRICS.get(skill_name, routing_accuracy)

    # Get transformer config if enabled
    demo_transformer = None
    format_instruction = ""
    if use_transformers:
        format_type = get_format_type_for_target(skill_name)
        format_instruction = get_format_instruction(format_type)
        demo_transformer = get_demo_transformer(skill_name)

    return BatchTarget(
        name=skill_name,
        prompt_path=prompt_path,
        training_data_path=data_path,
        metric_fn=metric_fn,
        threshold=threshold,
        max_demos=max_demos,
        demo_transformer=demo_transformer,
        format_instruction=format_instruction,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch optimize multiple agents and skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--agents",
        type=str,
        help="Comma-separated list of agent names to optimize",
    )
    parser.add_argument(
        "--skills",
        type=str,
        help="Comma-separated list of skill names to optimize",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Directory containing training data files (default: datasets)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Score threshold for demo selection (default: 0.7)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=3,
        help="Maximum demos per target (default: 3)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run optimizations in parallel (faster, may hit rate limits)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Maximum parallel workers (default: 3, only with --parallel)",
    )
    parser.add_argument(
        "--model",
        default="haiku",
        choices=["sonnet", "opus", "haiku"],
        help="Model to use for optimization runs (default: haiku)",
    )
    parser.add_argument(
        "--tier",
        default=None,
        choices=["haiku", "sonnet", "opus", "tiered"],
        help="Model tier (haiku, sonnet, opus) or 'tiered' for progressive Haiku->Sonnet->Opus",
    )
    parser.add_argument(
        "--algorithm",
        default="bootstrap",
        choices=["bootstrap", "copro", "iterative"],
        help="Optimization algorithm (default: bootstrap)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Rounds for iterative optimization (default: 3)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=5,
        help="Variants for COPRO optimization (default: 5)",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save markdown report",
    )
    parser.add_argument(
        "-o", "--output",
        default="optimized-prompts",
        help="Output directory for optimized prompts",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print progress (default: True)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout per API call in seconds (default: 180, or PROMPT_OPTIMIZER_TIMEOUT env var)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available agents and skills with training data",
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        default=True,
        help="Apply demo transformers and format instructions (default: True)",
    )
    parser.add_argument(
        "--no-transformers",
        action="store_true",
        help="Disable demo transformers and format instructions",
    )
    parser.add_argument(
        "--holdout-gate",
        action="store_true",
        help="Run pre-flight holdout check before saving _latest.json. "
             "Prevents regression by comparing new optimization against existing.",
    )
    parser.add_argument(
        "--holdout-dir",
        type=str,
        default="datasets",
        help="Directory containing holdout files (default: datasets, used with --holdout-gate)",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available Agents (with training data):")
        for agent, path in AGENT_DATA_PATHS.items():
            data_exists = Path(args.data_dir, path).exists()
            status = "[OK]" if data_exists else "[MISSING]"
            print(f"  {status} {agent}")

        print("\nAvailable Skills (with training data):")
        for skill, path in SKILL_DATA_PATHS.items():
            data_exists = Path(args.data_dir, path).exists()
            status = "[OK]" if data_exists else "[MISSING]"
            print(f"  {status} {skill}")
        return

    # Validate inputs
    if not args.agents and not args.skills:
        parser.error("At least one of --agents or --skills is required")

    verbose = not args.quiet and args.verbose
    data_dir = Path(args.data_dir)

    # Determine transformer usage
    use_transformers = args.use_transformers and not args.no_transformers

    if verbose and use_transformers:
        print("Transformers: ENABLED (format instructions + demo transformers)")
    elif verbose:
        print("Transformers: DISABLED")

    # Build targets
    targets = []

    if args.agents:
        agent_names = [a.strip() for a in args.agents.split(",")]
        for name in agent_names:
            try:
                target = build_agent_target(
                    name, data_dir, args.threshold, args.max_demos, use_transformers
                )
                targets.append(target)
                if verbose:
                    print(f"Added agent target: {name}")
            except FileNotFoundError as e:
                print(f"Warning: {e}", file=sys.stderr)

    if args.skills:
        skill_names = [s.strip() for s in args.skills.split(",")]
        for name in skill_names:
            try:
                target = build_skill_target(
                    name, data_dir, args.threshold, args.max_demos, use_transformers
                )
                targets.append(target)
                if verbose:
                    print(f"Added skill target: {name}")
            except FileNotFoundError as e:
                print(f"Warning: {e}", file=sys.stderr)

    if not targets:
        print("Error: No valid targets found", file=sys.stderr)
        sys.exit(1)

    # Determine model(s) to use
    if args.tier:
        model = args.tier if args.tier != "tiered" else "haiku"
    else:
        model = args.model

    if verbose:
        print(f"\nTotal targets: {len(targets)}")
        print(f"Model: {model}{' (tiered progression)' if args.tier == 'tiered' else ''}")
        print(f"Algorithm: {args.algorithm}")
        print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
        print()

    # Setup
    runner = ClaudeRunner(model=model, timeout=args.timeout)
    storage = DemoStorage()

    # Backup existing _latest.json files if holdout gate is enabled
    if args.holdout_gate:
        import shutil as gate_shutil
        prompts_dir = storage.prompts_dir
        for target in targets:
            latest_path = prompts_dir / f"{target.name}_latest.json"
            backup_path = prompts_dir / f"{target.name}_latest.json.pre-gate"
            if latest_path.exists():
                gate_shutil.copy2(latest_path, backup_path)
                if verbose:
                    print(f"  Backed up {target.name}_latest.json for holdout gate")

    # Tiered optimization: Haiku -> Sonnet -> Opus
    if args.tier == "tiered":
        if verbose:
            print("=" * 50)
            print("TIERED OPTIMIZATION: Haiku -> Sonnet -> Opus")
            print("=" * 50)

        # Phase 1: Haiku (quick baseline)
        if verbose:
            print("\n[Phase 1] Haiku baseline...")
        haiku_runner = ClaudeRunner(model="haiku", timeout=args.timeout)
        summary = optimize_batch_sequential(
            targets=targets,
            runner=haiku_runner,
            storage=storage,
            verbose=verbose,
        )

        # Phase 2: Sonnet (refinement) on successful targets
        successful_targets = [t for t in targets if any(
            r.success for r in summary.results if r.target.name == t.name
        )]
        if successful_targets and verbose:
            print(f"\n[Phase 2] Sonnet refinement on {len(successful_targets)} targets...")
        if successful_targets:
            sonnet_runner = ClaudeRunner(model="sonnet", timeout=args.timeout)
            summary = optimize_batch_sequential(
                targets=successful_targets,
                runner=sonnet_runner,
                storage=storage,
                verbose=verbose,
            )

        # Phase 3: Opus (polish) on high-scoring targets
        if verbose:
            print(f"\n[Phase 3] Opus polish...")
        opus_runner = ClaudeRunner(model="opus", timeout=args.timeout)
        summary = optimize_batch_sequential(
            targets=successful_targets[:3],  # Opus is expensive, limit targets
            runner=opus_runner,
            storage=storage,
            verbose=verbose,
        )

    # Standard optimization with selected algorithm
    elif args.algorithm == "bootstrap":
        if args.parallel:
            summary = optimize_batch_parallel(
                targets=targets,
                runner=runner,
                storage=storage,
                max_workers=args.workers,
                verbose=verbose,
            )
        else:
            summary = optimize_batch_sequential(
                targets=targets,
                runner=runner,
                storage=storage,
                verbose=verbose,
            )

    elif args.algorithm in ("copro", "iterative"):
        # For COPRO/Iterative, process sequentially with enhanced algorithms
        from prompt_optimizer.bootstrap import TrainingExample
        import json

        results = []
        for target in targets:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Processing: {target.name}")
                print(f"{'='*50}")

            # Load training data
            training_data = []
            with open(target.training_data_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        training_data.append(TrainingExample(
                            input_text=data["input"],
                            expected_output=data["expected"],
                            metadata=data.get("metadata"),
                        ))

            # Load base prompt
            with open(target.prompt_path) as f:
                base_prompt = f.read()

            if args.algorithm == "copro":
                copro = COPROOptimizer(runner, n_variants=args.variants, storage=storage)
                copro_result, bootstrap_result = copro.optimize_with_bootstrap(
                    base_prompt=base_prompt,
                    training_data=training_data,
                    metric_fn=target.metric_fn,
                    threshold=target.threshold,
                    max_demos=target.max_demos,
                    agent_name=target.name,
                    verbose=verbose,
                )
                if verbose:
                    print(f"COPRO improvement: {copro_result.improvement:+.3f}")

            else:  # iterative
                iterative = IterativeOptimizer(runner, max_rounds=args.rounds, storage=storage)
                iter_result = iterative.optimize(
                    base_prompt=base_prompt,
                    training_data=training_data,
                    metric_fn=target.metric_fn,
                    threshold=target.threshold,
                    max_demos=target.max_demos,
                    agent_name=target.name,
                    verbose=verbose,
                )
                if verbose:
                    if iter_result.rounds:
                        initial_score = iter_result.rounds[0].score
                        improvement = iter_result.final_score - initial_score
                    else:
                        improvement = 0.0
                    print(f"Iterative improvement: {improvement:+.3f}")

        # Use bootstrap summary for now (algorithms track their own results)
        summary = optimize_batch_sequential(
            targets=targets,
            runner=runner,
            storage=storage,
            verbose=False,
        )

    else:
        # Default bootstrap
        if args.parallel:
            summary = optimize_batch_parallel(
                targets=targets,
                runner=runner,
                storage=storage,
                max_workers=args.workers,
                verbose=verbose,
            )
        else:
            summary = optimize_batch_sequential(
                targets=targets,
                runner=runner,
                storage=storage,
                verbose=verbose,
            )

    # Pre-flight holdout gate (if enabled)
    # Strategy: backup _latest.json before optimization, restore if gate fails.
    # Since optimization already ran and overwrote _latest.json, we compare the
    # new _latest.json against the backed-up version on holdout data.
    if args.holdout_gate:
        import json as gate_json
        import shutil
        holdout_dir = Path(args.holdout_dir)
        all_metrics = {**AGENT_METRICS, **SKILL_METRICS}
        prompts_dir = storage.prompts_dir

        if verbose:
            print("\n" + "=" * 50)
            print("PRE-FLIGHT HOLDOUT GATE")
            print("=" * 50)

        for target in targets:
            # Find holdout file
            data_basename = AGENT_DATA_PATHS.get(target.name) or SKILL_DATA_PATHS.get(target.name) or target.name
            data_basename = data_basename.replace(".jsonl", "")
            holdout_path = holdout_dir / f"{data_basename}-holdout.jsonl"

            if not holdout_path.exists():
                if verbose:
                    print(f"\n  {target.name}: No holdout file, skipping gate")
                continue

            metric_fn = all_metrics.get(target.name)
            if not metric_fn:
                if verbose:
                    print(f"\n  {target.name}: No metric function, skipping gate")
                continue

            # Load holdout data
            holdout_data = []
            with open(holdout_path) as f:
                for line in f:
                    if line.strip():
                        data = gate_json.loads(line)
                        holdout_data.append(HoldoutExample(
                            input_text=data["input"],
                            expected_output=data["expected"],
                            metadata=data.get("metadata"),
                        ))

            # Check if there's a backup from before this run
            latest_path = prompts_dir / f"{target.name}_latest.json"
            backup_path = prompts_dir / f"{target.name}_latest.json.pre-gate"

            # Load the current (newly optimized) prompt
            new_optimized = storage.load_optimized_prompt(target.name)
            if not new_optimized:
                if verbose:
                    print(f"\n  {target.name}: No optimization found, skipping gate")
                continue

            # Evaluate new optimization on holdout
            new_prompt = new_optimized.to_prompt()
            new_scores = []
            for i, ex in enumerate(holdout_data):
                full_prompt = f"{new_prompt}\n\n## New Input\n\n{ex.input_text}"
                result = runner.run(full_prompt)
                if result.success:
                    score = metric_fn(ex.expected_output, result.output)
                    new_scores.append(score)
                    if verbose:
                        print(f"  [{target.name} holdout {i+1}/{len(holdout_data)}] {score:.3f}")

            new_avg = sum(new_scores) / len(new_scores) if new_scores else 0.0

            if verbose:
                print(f"\n  {target.name}: New holdout avg = {new_avg:.3f}")

            if backup_path.exists():
                # Evaluate the BACKUP (previous _latest.json) on the SAME
                # holdout data to get an apples-to-apples holdout comparison.
                # (Previous versions compared new holdout vs old training
                # avg_score — wrong scale, wrong metric.)
                from prompt_optimizer.storage import DemoStorage, OptimizedPrompt, Demo
                with open(backup_path) as bf:
                    backup_data = gate_json.load(bf)
                old_demos = [Demo(**d) for d in backup_data.get("demos", [])]
                old_optimized = OptimizedPrompt(
                    base_prompt=backup_data["base_prompt"],
                    demos=old_demos,
                    optimization_date=backup_data["optimization_date"],
                    metric_name=backup_data["metric_name"],
                    threshold=backup_data["threshold"],
                    avg_score=backup_data["avg_score"],
                    metadata=backup_data.get("metadata"),
                    format_instruction=backup_data.get("format_instruction", ""),
                )
                old_prompt = old_optimized.to_prompt()
                old_scores = []
                for i, ex in enumerate(holdout_data):
                    full_prompt = f"{old_prompt}\n\n## New Input\n\n{ex.input_text}"
                    result = runner.run(full_prompt)
                    if result.success:
                        score = metric_fn(ex.expected_output, result.output)
                        old_scores.append(score)
                        if verbose:
                            print(f"  [{target.name} backup holdout {i+1}/{len(holdout_data)}] {score:.3f}")

                old_avg = sum(old_scores) / len(old_scores) if old_scores else 0.0

                if verbose:
                    print(f"  {target.name}: Previous holdout avg = {old_avg:.3f}")

                if new_avg < old_avg - 0.02:
                    if verbose:
                        print(f"  GATE FAILED: Regression ({new_avg:.3f} < {old_avg:.3f} - 0.02)")
                        print(f"  Restoring previous _latest.json from backup")
                    shutil.copy2(backup_path, latest_path)
                    # Keep backup for auditing; caller can clean up if desired
                else:
                    if verbose:
                        print(f"  GATE PASSED: {new_avg:.3f} >= {old_avg:.3f} - 0.02")
                    # Clean up backup on pass
                    backup_path.unlink(missing_ok=True)
            else:
                if verbose:
                    print(f"  No previous optimization (first run). Gate: auto-pass.")

    # Generate report
    if args.report:
        report_path = Path(args.report)
        report = generate_batch_report(summary, report_path)
        print(f"\nReport saved to: {report_path}")
    else:
        # Print summary report to stdout
        print("\n" + generate_batch_report(summary))

    # Exit with error code if any failed
    if summary.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
