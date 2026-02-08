#!/usr/bin/env python3
"""
CLI tool for optimizing agent prompts using BootstrapFewShot.

Usage:
    python optimize_agent.py --agent capability-evaluator \
        --training-data datasets/evaluations.jsonl \
        --metric score_similarity \
        --output optimized-prompts/

Examples:
    # Basic optimization
    python optimize_agent.py -a capability-evaluator -t datasets/evaluations.jsonl

    # With holdout validation
    python optimize_agent.py -a capability-evaluator -t datasets/evaluations.jsonl --holdout 0.2

    # Compare optimized vs baseline
    python optimize_agent.py -a capability-evaluator --compare
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer import (
    BootstrapFewShot,
    ClaudeRunner,
    DemoStorage,
    exact_match,
    contains_keywords,
    score_similarity,
    composite_metric,
    # Agent-specific metrics
    issue_severity_match,
    test_coverage_score,
    complexity_classification,
    security_cwe_match,
    # Phase 3: High-priority agent metrics
    root_cause_match,
    pr_quality_match,
    implementation_completeness,
    refactoring_match,
    # Skill-specific metrics
    routing_accuracy,
    binary_decision_match,
    tool_tier_classification,
    # Phase 3: COPRO and Iterative
    COPROOptimizer,
    IterativeOptimizer,
)
from prompt_optimizer.bootstrap import TrainingExample
from prompt_optimizer.metrics import evaluation_score_metric, numeric_score_match
from prompt_optimizer.utils import find_agent_path, parse_markdown_prompt


# Available metrics
METRICS = {
    # Base metrics
    "exact_match": exact_match,
    "score_similarity": score_similarity,
    "numeric_score": numeric_score_match,
    "evaluation_score": evaluation_score_metric(threshold=70.0),
    # Agent-specific metrics
    "issue_severity": issue_severity_match,
    "test_coverage": test_coverage_score,
    "complexity": complexity_classification,
    "security_cwe": security_cwe_match,
    # Phase 3: High-priority agent metrics
    "root_cause": root_cause_match,
    "pr_quality": pr_quality_match,
    "implementation": implementation_completeness,
    "refactoring": refactoring_match,
    # Skill-specific metrics
    "routing": routing_accuracy,
    "binary_decision": binary_decision_match,
    "tool_tier": tool_tier_classification,
}


def load_training_data(path: str) -> list[TrainingExample]:
    """Load training data from JSONL file."""
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


def load_agent_prompt(agent_name: str) -> str:
    """Load the base prompt for an agent using shared path resolution."""
    agent_path = find_agent_path(agent_name)
    with open(agent_path) as f:
        content = f.read()
    return parse_markdown_prompt(content)


def save_optimized_agent(agent_name: str, optimized_content: str, output_dir: str):
    """Save optimized agent to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{agent_name}-optimized.md"
    with open(file_path, "w") as f:
        f.write(optimized_content)

    print(f"Saved optimized agent to: {file_path}")
    return file_path


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Optimize agent prompts using BootstrapFewShot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    return parser


def parse_args_only(args=None):
    """Parse arguments without running main. For testing."""
    parser = create_parser()
    _add_arguments(parser)
    return parser.parse_args(args)


def _add_arguments(parser):
    """Add all arguments to the parser."""
    parser.add_argument(
        "-a", "--agent",
        required=True,
        help="Name of the agent to optimize (e.g., capability-evaluator)",
    )
    parser.add_argument(
        "-t", "--training-data",
        help="Path to JSONL training data file",
    )
    parser.add_argument(
        "-m", "--metric",
        default="evaluation_score",
        choices=list(METRICS.keys()),
        help="Metric function to use (default: evaluation_score)",
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
        help="Maximum number of demos to include (default: 3)",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.0,
        help="Holdout ratio for validation (default: 0, no holdout)",
    )
    parser.add_argument(
        "-o", "--output",
        default="optimized-prompts",
        help="Output directory for optimized prompts",
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        choices=["sonnet", "opus", "haiku"],
        help="Model to use for optimization runs (default: sonnet)",
    )
    parser.add_argument(
        "--algorithm",
        default="bootstrap",
        choices=["bootstrap", "copro", "iterative"],
        help="Optimization algorithm to use (default: bootstrap)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds for iterative optimization (default: 3)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=5,
        help="Number of variants for COPRO optimization (default: 5)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare existing optimized prompt against baseline",
    )
    # Phase 5: Regularization parameters
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of cross-validation folds (0 = disabled, default: 0)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Example dropout rate for regularization (0.0-0.5, default: 0)",
    )
    parser.add_argument(
        "--probe-holdout",
        action="store_true",
        help="Enable probe holdout for overfitting detection",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Base timeout for model calls in seconds (default: 180)",
    )
    parser.add_argument(
        "--max-overfitting-gap",
        type=float,
        default=0.3,
        help="Maximum allowed train-probe gap before flagging overfitting (default: 0.3)",
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


def main():
    parser = create_parser()
    _add_arguments(parser)
    args = parser.parse_args()

    # Setup
    verbose = not args.quiet and args.verbose
    runner = ClaudeRunner(model=args.model, timeout=args.timeout)
    storage = DemoStorage()

    # Prepare probe holdout if enabled
    probe_holdout = None
    if args.probe_holdout and args.training_data:
        # Use 10% of training data as probe holdout
        all_data = load_training_data(args.training_data)
        probe_size = max(1, len(all_data) // 10)
        probe_holdout = all_data[:probe_size]
        if verbose:
            print(f"Using {probe_size} examples for probe holdout")

    # Create optimizer with Phase 5 regularization parameters
    optimizer = BootstrapFewShot(
        runner=runner,
        max_demos=args.max_demos,
        storage=storage,
        probe_holdout=probe_holdout,
        max_overfitting_gap=args.max_overfitting_gap,
        dropout_rate=args.dropout,
    )

    # Load agent prompt
    try:
        base_prompt = load_agent_prompt(args.agent)
        if verbose:
            print(f"Loaded agent prompt: {args.agent}")
            print(f"Prompt length: {len(base_prompt)} chars")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    metric_fn = METRICS[args.metric]

    # Compare mode
    if args.compare:
        optimized = storage.load_optimized_prompt(args.agent)
        if not optimized:
            print(f"No optimized prompt found for {args.agent}", file=sys.stderr)
            sys.exit(1)

        if not args.training_data:
            print("--training-data required for comparison", file=sys.stderr)
            sys.exit(1)

        test_data = load_training_data(args.training_data)
        results = optimizer.compare_baseline(
            base_prompt=base_prompt,
            test_data=test_data,
            optimized=optimized,
            metric_fn=metric_fn,
            verbose=verbose,
        )

        # Summary
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        print(f"Agent: {args.agent}")
        print(f"Test examples: {len(test_data)}")
        print(f"Baseline score: {results['baseline_score']:.3f}")
        print(f"Optimized score: {results['optimized_score']:.3f}")
        print(f"Improvement: {results['improvement']:+.3f} ({results['improvement_pct']:+.1f}%)")

        if results['improvement'] > 0:
            print("\n[SUCCESS] Optimization improved performance!")
        else:
            print("\n[WARNING] Optimization did not improve performance")

        return

    # Optimization mode
    if not args.training_data:
        print("--training-data required for optimization", file=sys.stderr)
        sys.exit(1)

    training_data = load_training_data(args.training_data)
    if verbose:
        print(f"Loaded {len(training_data)} training examples")
        print(f"Algorithm: {args.algorithm}")
        if args.cv_folds > 0:
            print(f"Cross-validation: {args.cv_folds} folds")
        if args.dropout > 0:
            print(f"Dropout rate: {args.dropout}")
        if args.probe_holdout:
            print(f"Probe holdout: enabled")

    # Phase 5: Cross-validation optimization
    if args.cv_folds > 0:
        if verbose:
            print(f"\nRunning {args.cv_folds}-fold cross-validation optimization...")
        cv_result = optimizer.optimize_with_cv(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            k=args.cv_folds,
            verbose=verbose,
        )
        print(f"\n{'='*50}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"Fold scores: {[f'{s:.3f}' for s in cv_result['fold_scores']]}")
        print(f"Mean score: {cv_result['mean_score']:.3f}")
        print(f"Std deviation: {cv_result.get('std_score', 0):.3f}")
        if cv_result.get('low_fold_warning'):
            print(f"[WARNING] Low fold detected - possible overfitting")

        # Continue with standard optimization using CV-validated approach
        result = optimizer.optimize(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            threshold=args.threshold,
            agent_name=args.agent,
            verbose=verbose,
        )

    # Select algorithm
    elif args.algorithm == "copro":
        if verbose:
            print(f"Using COPRO with {args.variants} variants")
        copro = COPROOptimizer(
            runner=runner,
            n_variants=args.variants,
            storage=storage,
        )
        copro_result = copro.optimize(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            agent_name=args.agent,
            verbose=verbose,
        )
        # Run bootstrap on best variant
        result = optimizer.optimize(
            base_prompt=copro_result.best_variant,
            training_data=training_data,
            metric_fn=metric_fn,
            threshold=args.threshold,
            agent_name=args.agent,
            verbose=verbose,
        )
        print(f"\nCOPRO improvement: {copro_result.improvement:+.3f}")

    elif args.algorithm == "iterative":
        if verbose:
            print(f"Using Iterative optimization with {args.rounds} rounds")
        iterative = IterativeOptimizer(
            runner=runner,
            max_rounds=args.rounds,
            storage=storage,
        )
        iter_result = iterative.optimize(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            threshold=args.threshold,
            max_demos=args.max_demos,
            agent_name=args.agent,
            verbose=verbose,
        )
        result = iter_result.final_bootstrap
        print(f"\nIterative improvement: {iter_result.total_improvement:+.3f}")

    elif args.holdout > 0:
        result, holdout_score = optimizer.optimize_with_holdout(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            holdout_ratio=args.holdout,
            threshold=args.threshold,
            agent_name=args.agent,
            verbose=verbose,
        )
        print(f"\nHoldout validation score: {holdout_score:.3f}")
    else:
        result = optimizer.optimize(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            threshold=args.threshold,
            agent_name=args.agent,
            verbose=verbose,
        )

    # Summary
    print("\n" + "=" * 50)
    print("OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"Agent: {args.agent}")
    print(f"Total examples: {result.total_examples}")
    print(f"Successful demos: {result.successful_examples}")
    print(f"Failed runs: {result.failed_examples}")
    print(f"Average demo score: {result.avg_score:.3f}")
    print(f"Demos selected: {len(result.optimized_prompt.demos)}")

    if result.successful_examples > 0:
        # Save the optimized prompt
        save_optimized_agent(
            args.agent,
            result.optimized_prompt.to_markdown(),
            args.output,
        )
        print(f"\n[SUCCESS] Optimization complete!")
        print(f"Use --compare to validate against baseline")
    else:
        print(f"\n[WARNING] No successful demos collected")
        print(f"Consider lowering --threshold or checking training data quality")


if __name__ == "__main__":
    main()
