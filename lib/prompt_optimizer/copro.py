"""
COPRO (Coordinate-based Prompt Optimization) implementation.

Generates instruction variants using mutation strategies and selects
the best-performing one through evaluation on training data.

Based on the DSPy COPRO algorithm pattern.
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from .claude_runner import ClaudeRunner, RunResult
from .bootstrap import TrainingExample, BootstrapResult
from .storage import Demo, OptimizedPrompt, DemoStorage

logger = logging.getLogger(__name__)


@dataclass
class MutationResult:
    """Result of applying a mutation to an instruction."""
    original: str
    mutated: str
    strategy: str


@dataclass
class VariantScore:
    """Score for a prompt variant."""
    variant: str
    strategy: str
    score: float
    eval_count: int


@dataclass
class COPROResult:
    """Result from a COPRO optimization run."""
    best_variant: str
    best_score: float
    best_strategy: str
    all_variants: List[VariantScore]
    original_score: float
    improvement: float

    @property
    def baseline_score(self) -> float:
        """Alias for original_score for backward compatibility."""
        return self.original_score


class InstructionMutator:
    """
    Generate instruction variants using mutation strategies.

    Uses the Claude model to rephrase instructions in different ways
    while preserving their meaning.
    """

    def __init__(self, runner: ClaudeRunner):
        """
        Initialize the mutator.

        Args:
            runner: ClaudeRunner instance for generating mutations
        """
        self.runner = runner

    def mutate_paraphrase(self, instruction: str) -> str:
        """
        Generate a paraphrased version of the instruction.

        Rewrites using different words while preserving exact meaning.

        Args:
            instruction: Original instruction text

        Returns:
            Paraphrased instruction
        """
        prompt = f"""Paraphrase the following instruction using different words.
Keep the exact same meaning but change the wording.
Do NOT add or remove any requirements.
Output ONLY the paraphrased instruction, nothing else.

Original instruction:
{instruction}

Paraphrased instruction:"""

        result = self.runner.run(prompt)
        if result.success:
            return result.output.strip()
        return instruction

    def mutate_elaborate(self, instruction: str) -> str:
        """
        Generate a more detailed version of the instruction.

        Adds clarifying details and explicit requirements.

        Args:
            instruction: Original instruction text

        Returns:
            Elaborated instruction
        """
        prompt = f"""Make the following instruction more explicit and detailed.
Add clarifying details about what is expected.
Be specific about format, steps, and requirements.
Do NOT change the core task.
Output ONLY the elaborated instruction, nothing else.

Original instruction:
{instruction}

Elaborated instruction:"""

        result = self.runner.run(prompt)
        if result.success:
            return result.output.strip()
        return instruction

    def mutate_simplify(self, instruction: str) -> str:
        """
        Generate a simpler, more concise version of the instruction.

        Removes redundancy and simplifies language.

        Args:
            instruction: Original instruction text

        Returns:
            Simplified instruction
        """
        prompt = f"""Simplify the following instruction.
Make it more concise while keeping all essential requirements.
Remove redundant phrases and use clearer language.
Output ONLY the simplified instruction, nothing else.

Original instruction:
{instruction}

Simplified instruction:"""

        result = self.runner.run(prompt)
        if result.success:
            return result.output.strip()
        return instruction

    def mutate_reorder(self, instruction: str) -> str:
        """
        Reorder the instruction to emphasize different aspects.

        Changes the order of requirements to test if sequence matters.

        Args:
            instruction: Original instruction text

        Returns:
            Reordered instruction
        """
        prompt = f"""Reorder the following instruction to emphasize different aspects.
Keep all the same requirements but present them in a different order.
Try putting the most important requirements first.
Output ONLY the reordered instruction, nothing else.

Original instruction:
{instruction}

Reordered instruction:"""

        result = self.runner.run(prompt)
        if result.success:
            return result.output.strip()
        return instruction

    def mutate_examples_focus(self, instruction: str) -> str:
        """
        Rewrite instruction with focus on concrete examples.

        Adds example patterns or specific instances.

        Args:
            instruction: Original instruction text

        Returns:
            Example-focused instruction
        """
        prompt = f"""Rewrite the following instruction with more concrete examples.
Add specific patterns or instances that illustrate what is expected.
Keep the core requirements but make them more tangible.
Output ONLY the rewritten instruction, nothing else.

Original instruction:
{instruction}

Example-focused instruction:"""

        result = self.runner.run(prompt)
        if result.success:
            return result.output.strip()
        return instruction

    def generate_variants(
        self,
        instruction: str,
        n: int = 5,
        strategies: Optional[List[str]] = None,
    ) -> List[MutationResult]:
        """
        Generate multiple instruction variants using different strategies.

        Args:
            instruction: Original instruction text
            n: Number of variants to generate
            strategies: List of strategy names to use. If None, uses all.
                       Options: paraphrase, elaborate, simplify, reorder, examples

        Returns:
            List of MutationResult objects
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        all_strategies = {
            'paraphrase': self.mutate_paraphrase,
            'elaborate': self.mutate_elaborate,
            'simplify': self.mutate_simplify,
            'reorder': self.mutate_reorder,
            'examples': self.mutate_examples_focus,
        }

        if strategies is None:
            strategies = list(all_strategies.keys())

        # If requesting more variants than strategies, repeat strategies
        if n > len(strategies):
            strategies_to_use = (strategies * ((n // len(strategies)) + 1))[:n]
        else:
            strategies_to_use = random.sample(strategies, n)

        results = []
        for strategy in strategies_to_use:
            if strategy not in all_strategies:
                logger.warning(f"Unknown strategy: {strategy}")
                continue

            mutate_fn = all_strategies[strategy]
            mutated = mutate_fn(instruction)

            results.append(MutationResult(
                original=instruction,
                mutated=mutated,
                strategy=strategy,
            ))

        return results


class COPROOptimizer:
    """
    Coordinate-based Prompt Optimization.

    Optimizes prompts by generating instruction variants and evaluating
    each on a subset of training data to find the best-performing version.
    """

    def __init__(
        self,
        runner: ClaudeRunner,
        n_variants: int = 5,
        eval_subset_size: int = 10,
        storage: Optional[DemoStorage] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            runner: ClaudeRunner instance for model calls
            n_variants: Number of instruction variants to generate
            eval_subset_size: Number of examples to use for evaluation
            storage: Optional DemoStorage for persistence
        """
        if eval_subset_size < 1:
            raise ValueError("eval_subset_size must be >= 1")
        self.runner = runner
        self.n_variants = n_variants
        self.eval_subset_size = eval_subset_size
        self.storage = storage or DemoStorage()
        self.mutator = InstructionMutator(runner)

    def _evaluate_variant(
        self,
        variant: str,
        data_subset: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        verbose: bool = False,
    ) -> Tuple[float, int]:
        """
        Evaluate a single variant on the data subset.

        Args:
            variant: The instruction variant to evaluate
            data_subset: Training examples to evaluate on
            metric_fn: Function(expected, actual) -> score in [0,1]
            verbose: Print progress

        Returns:
            Tuple of (average_score, count_evaluated)
        """
        scores = []

        for example in data_subset:
            full_prompt = f"{variant}\n\n## Input\n\n{example.input_text}"
            result = self.runner.run(full_prompt)

            if result.success:
                score = metric_fn(example.expected_output, result.output)
                scores.append(score)

        if not scores:
            return 0.0, 0

        avg_score = sum(scores) / len(scores)
        return avg_score, len(scores)

    def _select_best(
        self,
        variants: List[VariantScore],
    ) -> VariantScore:
        """
        Select the best variant based on scores.

        Args:
            variants: List of scored variants

        Returns:
            Best performing variant
        """
        if not variants:
            raise ValueError("No variants to select from")

        # Sort by score descending, then by eval count descending
        sorted_variants = sorted(
            variants,
            key=lambda v: (v.score, v.eval_count),
            reverse=True,
        )

        return sorted_variants[0]

    def optimize(
        self,
        base_prompt: str,
        training_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        agent_name: str = "default",
        verbose: bool = True,
    ) -> COPROResult:
        """
        Optimize a prompt using COPRO.

        Algorithm:
        1. Generate N instruction variants using mutation strategies
        2. Evaluate each variant on a subset of training data
        3. Return the best-performing variant

        Args:
            base_prompt: The base prompt/instruction to optimize
            training_data: List of training examples
            metric_fn: Function(expected, actual) -> score in [0,1]
            agent_name: Name for tracking/storage
            verbose: Whether to print progress

        Returns:
            COPROResult with best variant and comparison data
        """
        if verbose:
            print(f"Running COPRO optimization...")
            print(f"Generating {self.n_variants} variants...")

        # Select evaluation subset
        eval_subset = training_data[:self.eval_subset_size]
        if len(training_data) > self.eval_subset_size:
            eval_subset = random.sample(training_data, self.eval_subset_size)

        if verbose:
            print(f"Evaluation subset: {len(eval_subset)} examples")

        # Evaluate original first
        if verbose:
            print(f"\nEvaluating original instruction...")
        original_score, original_count = self._evaluate_variant(
            base_prompt, eval_subset, metric_fn, verbose
        )
        if verbose:
            print(f"Original score: {original_score:.3f} ({original_count} examples)")

        # Generate variants
        mutations = self.mutator.generate_variants(base_prompt, n=self.n_variants)

        if verbose:
            print(f"\nGenerated {len(mutations)} variants")
            for i, m in enumerate(mutations, 1):
                preview = m.mutated[:80] + "..." if len(m.mutated) > 80 else m.mutated
                print(f"  {i}. [{m.strategy}] {preview}")

        # Evaluate each variant
        variant_scores: List[VariantScore] = []

        for i, mutation in enumerate(mutations, 1):
            if verbose:
                print(f"\nEvaluating variant {i}/{len(mutations)} [{mutation.strategy}]...")

            score, count = self._evaluate_variant(
                mutation.mutated, eval_subset, metric_fn, verbose
            )

            variant_scores.append(VariantScore(
                variant=mutation.mutated,
                strategy=mutation.strategy,
                score=score,
                eval_count=count,
            ))

            if verbose:
                print(f"  Score: {score:.3f} ({count} examples)")

        # Include original in selection
        variant_scores.append(VariantScore(
            variant=base_prompt,
            strategy="original",
            score=original_score,
            eval_count=original_count,
        ))

        # Select best
        best = self._select_best(variant_scores)

        improvement = best.score - original_score

        if verbose:
            print(f"\n{'='*50}")
            print(f"COPRO OPTIMIZATION COMPLETE")
            print(f"{'='*50}")
            print(f"Original score: {original_score:.3f}")
            print(f"Best score: {best.score:.3f} [{best.strategy}]")
            print(f"Improvement: {improvement:+.3f}")

            if best.strategy != "original":
                print(f"\nBest variant preview:")
                preview = best.variant[:200] + "..." if len(best.variant) > 200 else best.variant
                print(f"  {preview}")

        return COPROResult(
            best_variant=best.variant,
            best_score=best.score,
            best_strategy=best.strategy,
            all_variants=variant_scores,
            original_score=original_score,
            improvement=improvement,
        )

    def optimize_with_bootstrap(
        self,
        base_prompt: str,
        training_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        threshold: float = 0.7,
        max_demos: int = 3,
        agent_name: str = "default",
        verbose: bool = True,
    ) -> Tuple[COPROResult, BootstrapResult]:
        """
        Combined COPRO + BootstrapFewShot optimization.

        First optimizes the instruction using COPRO, then applies
        BootstrapFewShot to generate few-shot examples.

        Args:
            base_prompt: Base prompt to optimize
            training_data: Training examples
            metric_fn: Metric function
            threshold: Bootstrap threshold
            max_demos: Maximum demos for Bootstrap
            agent_name: Name for storage
            verbose: Print progress

        Returns:
            Tuple of (COPROResult, BootstrapResult)
        """
        from .bootstrap import BootstrapFewShot

        # Step 1: COPRO optimization
        copro_result = self.optimize(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            agent_name=agent_name,
            verbose=verbose,
        )

        # Step 2: Bootstrap with best variant
        if verbose:
            print(f"\n{'='*50}")
            print(f"Continuing with BootstrapFewShot...")
            print(f"{'='*50}")

        bootstrap = BootstrapFewShot(
            runner=self.runner,
            max_demos=max_demos,
            storage=self.storage,
        )

        bootstrap_result = bootstrap.optimize(
            base_prompt=copro_result.best_variant,
            training_data=training_data,
            metric_fn=metric_fn,
            threshold=threshold,
            agent_name=agent_name,
            verbose=verbose,
        )

        return copro_result, bootstrap_result
