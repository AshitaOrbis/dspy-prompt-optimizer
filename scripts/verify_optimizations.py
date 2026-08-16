#!/usr/bin/env python3
"""
Verification CLI for validating prompt optimizations.

Run holdout evaluations, cross-validation, and regression checks
on optimized agents and skills.

Usage:
    # Verify a single agent
    python verify_optimizations.py --agent code-reviewer \
        --holdout datasets/code-reviews-holdout.jsonl

    # Verify all agents with holdout data
    python verify_optimizations.py --all --report reports/verification.md

    # Run cross-validation
    python verify_optimizations.py --agent code-reviewer \
        --cross-validate --k 5 \
        --training-data datasets/code-reviews.jsonl

    # Check for regressions
    python verify_optimizations.py --agent code-reviewer \
        --check-regression --threshold 0.02

Examples:
    # Full verification suite
    python verify_optimizations.py --all \
        --holdout-dir datasets/ \
        --report reports/phase3-verification.md

    # Update baseline after successful optimization
    python verify_optimizations.py --agent code-reviewer \
        --update-baseline --score 0.92
"""

import argparse
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer import (
    ClaudeRunner,
    DemoStorage,
    # Base metrics
    score_similarity,
    evaluation_score_metric,
    # Agent-specific metrics
    issue_severity_match,
    test_coverage_score,
    complexity_classification,
    security_cwe_match,
    # Phase 3 agent metrics
    root_cause_match,
    pr_quality_match,
    implementation_completeness,
    refactoring_match,
    # Plan quality metric
    plan_quality_match,
    # Tier 2 agent metrics
    discovery_quality_match,
    fact_check_quality_match,
    research_quality_match,
    # Tier 3 skill metrics
    writing_review_quality_match,
    handoff_quality_match,
    # Publication review metric
    publication_review_match,
    # Skill-specific metrics
    routing_accuracy,
    binary_decision_match,
    tool_tier_classification,
)
from prompt_optimizer.verification import (
    VerificationSuite,
    generate_verification_report,
)


# Metric mappings
AGENT_METRICS = {
    "code-reviewer": issue_severity_match,
    "test-writer": test_coverage_score,
    "security-auditor": security_cwe_match,
    "performance-analyzer": complexity_classification,
    "capability-evaluator": evaluation_score_metric(threshold=70.0),
    # Phase 3 agents
    "debugger": root_cause_match,
    "pr-preparer": pr_quality_match,
    "feature-implementer": implementation_completeness,
    "refactoring-advisor": refactoring_match,
    # Tier 2 agents
    "capability-discoverer": discovery_quality_match,
    "fact-checker": fact_check_quality_match,
    "web-researcher": research_quality_match,
}

SKILL_METRICS = {
    "mcp-search-framework": routing_accuracy,
    "mgrep-guide": binary_decision_match,
    "advanced-tool-use": tool_tier_classification,
    "plan-mode-quality": plan_quality_match,
    # Tier 3 skills
    "writing-review": writing_review_quality_match,
    "session-handoff": handoff_quality_match,
    # Publication review (per-model targets)
    "publication-review-gpt": publication_review_match,
    "publication-review-gemini": publication_review_match,
    "publication-review-opus": publication_review_match,
}

# Data file mappings
AGENT_DATA_PATHS = {
    "code-reviewer": "code-reviews",
    "test-writer": "test-suites",
    "security-auditor": "security-audits",
    "performance-analyzer": "performance-analysis",
    "capability-evaluator": "evaluations",
    # Phase 3 agents
    "debugger": "debugging-sessions",
    "pr-preparer": "pr-preparations",
    "feature-implementer": "feature-implementations",
    "refactoring-advisor": "refactoring-decisions",
    # Tier 2 agents
    "capability-discoverer": "discoveries",
    "fact-checker": "fact-checks",
    "web-researcher": "web-research",
}

SKILL_DATA_PATHS = {
    "mcp-search-framework": "search-routing",
    "mgrep-guide": "search-decisions",
    "advanced-tool-use": "tool-selection",
    "plan-mode-quality": "plan-quality",
    # Tier 3 skills
    "writing-review": "writing-reviews",
    "session-handoff": "session-handoffs",
    # Publication review (per-model targets)
    "publication-review-gpt": "publication-review-gpt",
    "publication-review-gemini": "publication-review-gemini",
    "publication-review-opus": "publication-review-opus",
}


def get_metric_fn(name: str):
    """Get metric function for agent or skill."""
    if name in AGENT_METRICS:
        return AGENT_METRICS[name]
    if name in SKILL_METRICS:
        return SKILL_METRICS[name]
    return score_similarity


def get_data_basename(name: str) -> str:
    """Get data file basename for agent or skill."""
    if name in AGENT_DATA_PATHS:
        return AGENT_DATA_PATHS[name]
    if name in SKILL_DATA_PATHS:
        return SKILL_DATA_PATHS[name]
    return name


def find_holdout_file(name: str, holdout_dir: Path) -> Path:
    """Find holdout file for agent or skill."""
    basename = get_data_basename(name)

    # Try various naming patterns
    patterns = [
        f"{basename}-holdout.jsonl",
        f"{name}-holdout.jsonl",
        f"{basename}_holdout.jsonl",
        f"holdout/{basename}.jsonl",
    ]

    for pattern in patterns:
        path = holdout_dir / pattern
        if path.exists():
            return path

    return holdout_dir / f"{basename}-holdout.jsonl"


def find_training_file(name: str, data_dir: Path) -> Path:
    """Find training file for agent or skill."""
    basename = get_data_basename(name)

    patterns = [
        f"{basename}.jsonl",
        f"{name}.jsonl",
    ]

    for pattern in patterns:
        path = data_dir / pattern
        if path.exists():
            return path

    return data_dir / f"{basename}.jsonl"


def list_available_targets(data_dir: Path):
    """List agents and skills with available data."""
    print("Available Agents:")
    for agent, basename in AGENT_DATA_PATHS.items():
        training_path = data_dir / f"{basename}.jsonl"
        holdout_path = data_dir / f"{basename}-holdout.jsonl"
        t_status = "[OK]" if training_path.exists() else "[MISSING]"
        h_status = "[OK]" if holdout_path.exists() else "[MISSING]"
        print(f"  {agent}")
        print(f"    Training: {t_status} {training_path.name}")
        print(f"    Holdout:  {h_status} {holdout_path.name}")

    print("\nAvailable Skills:")
    for skill, basename in SKILL_DATA_PATHS.items():
        training_path = data_dir / f"{basename}.jsonl"
        holdout_path = data_dir / f"{basename}-holdout.jsonl"
        t_status = "[OK]" if training_path.exists() else "[MISSING]"
        h_status = "[OK]" if holdout_path.exists() else "[MISSING]"
        print(f"  {skill}")
        print(f"    Training: {t_status} {training_path.name}")
        print(f"    Holdout:  {h_status} {holdout_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify prompt optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target selection
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--agent",
        type=str,
        help="Agent name to verify",
    )
    target_group.add_argument(
        "--skill",
        type=str,
        help="Skill name to verify",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Verify all agents and skills with available holdout data",
    )

    # Data paths
    parser.add_argument(
        "--holdout",
        type=str,
        help="Path to holdout JSONL file (for single agent/skill)",
    )
    parser.add_argument(
        "--holdout-dir",
        type=str,
        default="datasets",
        help="Directory containing holdout files (default: datasets)",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        help="Path to training data JSONL (for cross-validation)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Directory containing training data (default: datasets)",
    )
    parser.add_argument(
        "--baseline-scores",
        type=str,
        default="datasets/baseline-scores.json",
        help="Path to baseline scores JSON file",
    )

    # Verification options
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run k-fold cross-validation",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Check for regression against baseline",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.7,
        help="Score threshold to pass holdout evaluation (default: 0.7)",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.02,
        help="Maximum acceptable score drop for regression (default: 0.02)",
    )

    # Baseline management
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline score for agent/skill",
    )
    parser.add_argument(
        "--score",
        type=float,
        help="Score to use when updating baseline",
    )

    # Output
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save markdown report",
    )

    # Model and execution
    parser.add_argument(
        "--model",
        default="haiku",
        choices=["sonnet", "opus", "haiku", "fable"],
        help="Model to use for evaluation (default: haiku)",
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
        "--list",
        action="store_true",
        help="List available agents and skills with data status",
    )

    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_available_targets(Path(args.data_dir))
        return

    # Handle baseline update
    if args.update_baseline:
        if not (args.agent or args.skill):
            parser.error("--update-baseline requires --agent or --skill")
        if args.score is None:
            parser.error("--update-baseline requires --score")

        name = args.agent or args.skill
        baseline_path = Path(args.baseline_scores)
        runner = ClaudeRunner(model=args.model)
        suite = VerificationSuite(runner, baseline_scores_path=baseline_path)
        suite.update_baseline(name, args.score, verbose=not args.quiet)
        return

    # Validate inputs
    if not (args.agent or args.skill or args.all):
        parser.error("One of --agent, --skill, or --all is required")

    verbose = not args.quiet and args.verbose
    holdout_dir = Path(args.holdout_dir)
    data_dir = Path(args.data_dir)
    baseline_path = Path(args.baseline_scores)

    # Setup
    runner = ClaudeRunner(model=args.model)
    storage = DemoStorage()
    suite = VerificationSuite(runner, storage, baseline_path)

    # Determine targets
    if args.all:
        # Find all agents/skills with holdout data
        targets = []
        metric_fns = {}

        for agent in AGENT_DATA_PATHS:
            holdout_path = find_holdout_file(agent, holdout_dir)
            if holdout_path.exists():
                targets.append(agent)
                metric_fns[agent] = get_metric_fn(agent)

        for skill in SKILL_DATA_PATHS:
            holdout_path = find_holdout_file(skill, holdout_dir)
            if holdout_path.exists():
                targets.append(skill)
                metric_fns[skill] = get_metric_fn(skill)

        if not targets:
            print("No targets found with holdout data", file=sys.stderr)
            sys.exit(1)

        if verbose:
            print(f"Found {len(targets)} targets with holdout data")

        # Build data basename mapping (agent/skill name -> dataset basename)
        all_data_basenames = {}
        all_data_basenames.update(AGENT_DATA_PATHS)
        all_data_basenames.update(SKILL_DATA_PATHS)

        # Run full verification
        report = suite.run_full_verification(
            agent_names=targets,
            holdout_data_dir=holdout_dir,
            training_data_dir=data_dir,
            metric_fns=metric_fns,
            pass_threshold=args.pass_threshold,
            regression_threshold=args.regression_threshold,
            run_cross_validation=args.cross_validate,
            k_folds=args.k,
            verbose=verbose,
            data_basenames=all_data_basenames,
        )

        # Generate and save report
        if args.report:
            report_content = generate_verification_report(report, Path(args.report))
            print(f"\nReport saved to: {args.report}")
        else:
            print("\n" + generate_verification_report(report))

        # Exit with error if any failures
        if report.summary.get('holdout_failed', 0) > 0 or report.summary.get('regressions', 0) > 0:
            sys.exit(1)

    else:
        # Single target verification
        name = args.agent or args.skill
        metric_fn = get_metric_fn(name)

        # Holdout evaluation
        if args.holdout:
            holdout_path = Path(args.holdout)
        else:
            holdout_path = find_holdout_file(name, holdout_dir)

        if holdout_path.exists():
            holdout_data = suite.load_holdout_data(holdout_path)
            result = suite.run_holdout_evaluation(
                agent_name=name,
                holdout_data=holdout_data,
                metric_fn=metric_fn,
                pass_threshold=args.pass_threshold,
                verbose=verbose,
            )

            # Regression check
            if args.check_regression:
                suite.check_regression(
                    agent_name=name,
                    new_score=result.holdout_score,
                    regression_threshold=args.regression_threshold,
                    verbose=verbose,
                )

            if not result.passed:
                sys.exit(1)
        else:
            if verbose:
                print(f"No holdout data found at {holdout_path}")

        # Cross-validation
        if args.cross_validate:
            if args.training_data:
                training_path = Path(args.training_data)
            else:
                training_path = find_training_file(name, data_dir)

            if not training_path.exists():
                print(f"Training data not found: {training_path}", file=sys.stderr)
                sys.exit(1)

            training_data = suite.load_holdout_data(training_path)
            cv_result = suite.run_cross_validation(
                agent_name=name,
                training_data=training_data,
                metric_fn=metric_fn,
                k=args.k,
                verbose=verbose,
            )


if __name__ == "__main__":
    main()
