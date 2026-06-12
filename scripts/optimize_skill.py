#!/usr/bin/env python3
"""
CLI tool for optimizing skill prompts using BootstrapFewShot.

Skills differ from agents in that they typically define decision trees
or routing logic. This optimizer focuses on improving decision accuracy.

Usage:
    python optimize_skill.py --skill mcp-search-framework \
        --training-data datasets/search-routing.jsonl \
        --metric routing \
        --output optimized-prompts/

Examples:
    # Optimize search routing skill
    python optimize_skill.py -s mcp-search-framework -t datasets/search-routing.jsonl

    # Optimize mgrep decision skill
    python optimize_skill.py -s mgrep-guide -t datasets/search-decisions.jsonl --metric binary_decision

    # Compare optimized vs baseline
    python optimize_skill.py -s mcp-search-framework --compare
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
    score_similarity,
    # Skill-specific metrics
    routing_accuracy,
    binary_decision_match,
    tool_tier_classification,
    # Plan quality metric
    plan_quality_match,
)
from prompt_optimizer.bootstrap import TrainingExample
from prompt_optimizer.storage import OptimizedPrompt
from prompt_optimizer.utils import find_skill_path
from prompt_optimizer.validation import validate_name, contained_path, ValidationError


# Available metrics for skills
SKILL_METRICS = {
    "routing": routing_accuracy,
    "binary_decision": binary_decision_match,
    "tool_tier": tool_tier_classification,
    "plan_quality": plan_quality_match,
    "exact_match": exact_match,
    "score_similarity": score_similarity,
}

# Skill name to metric mapping (defaults)
SKILL_DEFAULT_METRICS = {
    "mcp-search-framework": "routing",
    "mgrep-guide": "binary_decision",
    "advanced-tool-use": "tool_tier",
    "dispatching-parallel-agents": "binary_decision",
    "plan-mode-quality": "plan_quality",
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


def load_skill_prompt(skill_name: str) -> str:
    """Load the base prompt for a skill."""
    # Use the shared, name-validated + path-contained resolver (utils.find_skill_path)
    # rather than a local copy, so '--skill ../../etc/foo' cannot traverse out of
    # the skills directory.
    skill_path = find_skill_path(skill_name)

    with open(skill_path) as f:
        content = f.read()

    # Extract content after YAML frontmatter if present
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()

    return content


def save_optimized_skill(skill_name: str, optimized_content: str, output_dir: str):
    """Save optimized skill to output directory."""
    validate_name(skill_name, kind="skill name")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Containment-check the derived path so the (validated) name still can't
    # escape the output directory via any edge case.
    file_path = contained_path(output_path, f"{skill_name}-optimized.md")
    with open(file_path, "w") as f:
        f.write(optimized_content)

    print(f"Saved optimized skill to: {file_path}")
    return file_path


def inject_examples_into_skill(skill_content: str, demos: list) -> str:
    """Inject few-shot examples into a skill's decision tree or routing section."""
    # Find a good insertion point - after decision tree or at end
    insertion_markers = [
        "## Examples",
        "## Quick Reference",
        "## Decision Tree",
    ]

    for marker in insertion_markers:
        if marker in skill_content:
            # Insert examples section before this marker
            parts = skill_content.split(marker, 1)
            examples_section = format_skill_examples(demos)
            return f"{parts[0]}\n{examples_section}\n\n{marker}{parts[1]}"

    # Append to end if no marker found
    examples_section = format_skill_examples(demos)
    return f"{skill_content}\n\n{examples_section}"


def format_skill_examples(demos: list) -> str:
    """Format demos as a skill examples section."""
    lines = [
        "## Optimized Examples",
        "",
        "These examples were automatically generated through prompt optimization.",
        "",
    ]

    for i, demo in enumerate(demos, 1):
        lines.extend([
            f"### Example {i} (Score: {demo.score:.2f})",
            "",
            f"**Query:** {demo.input_text[:200]}{'...' if len(demo.input_text) > 200 else ''}",
            "",
            f"**Decision:** {demo.output_text[:300]}{'...' if len(demo.output_text) > 300 else ''}",
            "",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Optimize skill prompts using BootstrapFewShot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-s", "--skill",
        required=True,
        help="Name of the skill to optimize (e.g., mcp-search-framework)",
    )
    parser.add_argument(
        "-t", "--training-data",
        help="Path to JSONL training data file",
    )
    parser.add_argument(
        "-m", "--metric",
        choices=list(SKILL_METRICS.keys()),
        help="Metric function to use (default: auto-detect from skill)",
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
        default=5,
        help="Maximum number of demos to include (default: 5, skills benefit from more examples)",
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
        default="haiku",
        choices=["sonnet", "opus", "haiku"],
        help="Model to use for optimization runs (default: haiku for fast iteration)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare existing optimized prompt against baseline",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test evaluation on training data without optimization",
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

    args = parser.parse_args()

    # Validate the skill name up front, before it is used to build any path or
    # passed to a claude run. Rejects traversal/injection on the primary entry.
    try:
        validate_name(args.skill, kind="skill name")
    except ValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Setup
    verbose = not args.quiet and args.verbose
    runner = ClaudeRunner(model=args.model)
    storage = DemoStorage()
    optimizer = BootstrapFewShot(runner, max_demos=args.max_demos, storage=storage)

    # Load skill prompt
    try:
        base_prompt = load_skill_prompt(args.skill)
        if verbose:
            print(f"Loaded skill prompt: {args.skill}")
            print(f"Prompt length: {len(base_prompt)} chars")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine metric
    if args.metric:
        metric_fn = SKILL_METRICS[args.metric]
    elif args.skill in SKILL_DEFAULT_METRICS:
        metric_name = SKILL_DEFAULT_METRICS[args.skill]
        metric_fn = SKILL_METRICS[metric_name]
        if verbose:
            print(f"Auto-selected metric: {metric_name}")
    else:
        metric_fn = SKILL_METRICS["routing"]
        if verbose:
            print("Using default metric: routing")

    # Test mode - evaluate baseline
    if args.test:
        if not args.training_data:
            print("--training-data required for test mode", file=sys.stderr)
            sys.exit(1)

        training_data = load_training_data(args.training_data)
        print(f"\nTesting baseline on {len(training_data)} examples...")

        scores = []
        for i, example in enumerate(training_data):
            full_prompt = f"{base_prompt}\n\n## Query\n\n{example.input_text}"
            result = runner.run(full_prompt)

            if result.success:
                score = metric_fn(example.expected_output, result.output)
                scores.append(score)
                if verbose:
                    status = "PASS" if score >= 0.7 else "FAIL"
                    print(f"  [{i+1}/{len(training_data)}] Score: {score:.2f} [{status}]")

        avg_score = sum(scores) / len(scores) if scores else 0
        passing = sum(1 for s in scores if s >= 0.7)

        print(f"\n{'='*50}")
        print(f"TEST RESULTS: {args.skill}")
        print(f"{'='*50}")
        print(f"Average score: {avg_score:.3f}")
        print(f"Passing (>=0.7): {passing}/{len(scores)} ({100*passing/len(scores):.1f}%)")
        return

    # Compare mode
    if args.compare:
        optimized = storage.load_optimized_prompt(f"skill-{args.skill}")
        if not optimized:
            print(f"No optimized prompt found for skill-{args.skill}", file=sys.stderr)
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

        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")
        print(f"Skill: {args.skill}")
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

    if args.holdout > 0:
        result, holdout_score = optimizer.optimize_with_holdout(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            holdout_ratio=args.holdout,
            threshold=args.threshold,
            agent_name=f"skill-{args.skill}",
            verbose=verbose,
        )
        print(f"\nHoldout validation score: {holdout_score:.3f}")
    else:
        result = optimizer.optimize(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            threshold=args.threshold,
            agent_name=f"skill-{args.skill}",
            verbose=verbose,
        )

    # Summary
    print(f"\n{'='*50}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*50}")
    print(f"Skill: {args.skill}")
    print(f"Total examples: {result.total_examples}")
    print(f"Successful demos: {result.successful_examples}")
    print(f"Failed runs: {result.failed_examples}")
    print(f"Average demo score: {result.avg_score:.3f}")
    print(f"Demos selected: {len(result.optimized_prompt.demos)}")

    if result.successful_examples > 0:
        # Save the optimized prompt
        save_optimized_skill(
            args.skill,
            result.optimized_prompt.to_markdown(),
            args.output,
        )

        # Also create an integrated version with examples injected
        integrated_content = inject_examples_into_skill(
            base_prompt,
            result.optimized_prompt.demos
        )
        integrated_path = contained_path(Path(args.output), f"{args.skill}-integrated.md")
        with open(integrated_path, "w") as f:
            f.write(integrated_content)
        print(f"Saved integrated skill to: {integrated_path}")

        print(f"\n[SUCCESS] Optimization complete!")
        print(f"Use --compare to validate against baseline")
        print(f"Use --test to evaluate current performance")
    else:
        print(f"\n[WARNING] No successful demos collected")
        print(f"Consider lowering --threshold or checking training data quality")


if __name__ == "__main__":
    main()
