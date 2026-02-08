"""
Iterative Refinement optimization with synthetic data generation.

Implements multi-round optimization where each round's output improves
the next through data augmentation and progressive refinement.

Based on DSPy iterative optimization patterns.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Dict, Any

from .claude_runner import ClaudeRunner, RunResult
from .bootstrap import BootstrapFewShot, TrainingExample, BootstrapResult
from .storage import Demo, OptimizedPrompt, DemoStorage
from .copro import COPROOptimizer, COPROResult

logger = logging.getLogger(__name__)


@dataclass
class GeneratedExample:
    """A synthetically generated training example."""
    input_text: str
    expected_output: str
    metadata: Optional[Dict[str, Any]] = None
    source: str = "synthetic"  # synthetic, edge_case, adversarial


@dataclass
class RoundResult:
    """Result from a single optimization round."""
    round_num: int
    prompt: str
    score: float
    demos: List[Demo]
    training_size: int
    new_examples: int


@dataclass
class IterativeResult:
    """Result from iterative optimization."""
    final_prompt: OptimizedPrompt
    final_score: float
    rounds: List[RoundResult]
    total_rounds: int
    converged: bool
    convergence_round: Optional[int]


class SyntheticDataGenerator:
    """
    Generate synthetic training data from existing demonstrations.

    Uses the model to create variations, edge cases, and adversarial
    examples to expand the training dataset.
    """

    def __init__(self, runner: ClaudeRunner):
        """
        Initialize the generator.

        Args:
            runner: ClaudeRunner for generating synthetic data
        """
        self.runner = runner

    def generate_similar(
        self,
        demos: List[Demo],
        n: int = 20,
        verbose: bool = False,
    ) -> List[GeneratedExample]:
        """
        Generate examples similar to existing demonstrations.

        Creates variations that test the same patterns.

        Args:
            demos: Existing successful demos
            n: Number of examples to generate
            verbose: Print progress

        Returns:
            List of generated training examples
        """
        if n <= 0:
            return []
        if not demos:
            return []

        examples = []
        per_demo = max(1, n // len(demos))

        for demo in demos:
            prompt = f"""Based on this input/output example, generate {per_demo} similar but different examples.
Each should test the same pattern but with different content.

Original Input:
{demo.input_text[:500]}

Original Output:
{demo.output_text[:500]}

Generate {per_demo} new examples in this exact format:
---
INPUT:
[new input here]
OUTPUT:
[expected output here]
---

Generate only the examples, no explanations."""

            result = self.runner.run(prompt)
            if result.success:
                parsed = self._parse_examples(result.output)
                for inp, out in parsed:
                    examples.append(GeneratedExample(
                        input_text=inp,
                        expected_output=out,
                        metadata={"source_demo_score": demo.score},
                        source="synthetic_similar",
                    ))

            if len(examples) >= n:
                break

        if verbose:
            print(f"Generated {len(examples)} similar examples")

        return examples[:n]

    def generate_edge_cases(
        self,
        demos: List[Demo],
        n: int = 10,
        verbose: bool = False,
    ) -> List[GeneratedExample]:
        """
        Generate edge case examples that test boundaries.

        Creates challenging scenarios like empty inputs, extreme values, etc.

        Args:
            demos: Existing demos for context
            n: Number of edge cases to generate
            verbose: Print progress

        Returns:
            List of edge case training examples
        """
        if not demos:
            return []

        # Use first demo as template
        template_demo = demos[0]

        prompt = f"""Based on this example of the expected task:

Input example:
{template_demo.input_text[:500]}

Output example:
{template_demo.output_text[:500]}

Generate {n} EDGE CASE examples that test boundary conditions:
- Empty or minimal input
- Very long input
- Special characters
- Unusual but valid inputs
- Corner cases

Format each as:
---
INPUT:
[edge case input]
OUTPUT:
[expected output for this edge case]
---

Generate only the examples, no explanations."""

        result = self.runner.run(prompt)
        examples = []

        if result.success:
            parsed = self._parse_examples(result.output)
            for inp, out in parsed:
                examples.append(GeneratedExample(
                    input_text=inp,
                    expected_output=out,
                    metadata={"edge_case": True},
                    source="edge_case",
                ))

        if verbose:
            print(f"Generated {len(examples)} edge case examples")

        return examples[:n]

    def generate_adversarial(
        self,
        demos: List[Demo],
        n: int = 5,
        verbose: bool = False,
    ) -> List[GeneratedExample]:
        """
        Generate adversarial examples designed to be challenging.

        Creates inputs that might confuse or trip up the model.

        Args:
            demos: Existing demos for context
            n: Number of adversarial examples to generate
            verbose: Print progress

        Returns:
            List of adversarial training examples
        """
        if not demos:
            return []

        template_demo = demos[0]

        prompt = f"""Based on this example of the expected task:

Input example:
{template_demo.input_text[:500]}

Output example:
{template_demo.output_text[:500]}

Generate {n} ADVERSARIAL examples that might confuse a model:
- Misleading inputs that look like one thing but are another
- Inputs with subtle errors
- Ambiguous cases
- Inputs that require careful reading

The output should still be the CORRECT response.

Format each as:
---
INPUT:
[adversarial input]
OUTPUT:
[correct expected output]
---

Generate only the examples, no explanations."""

        result = self.runner.run(prompt)
        examples = []

        if result.success:
            parsed = self._parse_examples(result.output)
            for inp, out in parsed:
                examples.append(GeneratedExample(
                    input_text=inp,
                    expected_output=out,
                    metadata={"adversarial": True},
                    source="adversarial",
                ))

        if verbose:
            print(f"Generated {len(examples)} adversarial examples")

        return examples[:n]

    def _parse_examples(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse generated examples from model output.

        Args:
            text: Raw model output containing examples

        Returns:
            List of (input, output) tuples
        """
        examples = []

        # Split by delimiter
        parts = text.split("---")

        current_input = None
        current_output = None

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Look for INPUT: and OUTPUT: sections
            if "INPUT:" in part and "OUTPUT:" in part:
                lines = part.split("\n")
                input_lines = []
                output_lines = []
                current_section = None

                for line in lines:
                    if line.strip().startswith("INPUT:"):
                        current_section = "input"
                        rest = line.replace("INPUT:", "").strip()
                        if rest:
                            input_lines.append(rest)
                    elif line.strip().startswith("OUTPUT:"):
                        current_section = "output"
                        rest = line.replace("OUTPUT:", "").strip()
                        if rest:
                            output_lines.append(rest)
                    elif current_section == "input":
                        input_lines.append(line)
                    elif current_section == "output":
                        output_lines.append(line)

                if input_lines and output_lines:
                    examples.append((
                        "\n".join(input_lines).strip(),
                        "\n".join(output_lines).strip(),
                    ))

        return examples

    def validate_generated(
        self,
        examples: List[GeneratedExample],
        metric_fn: Callable[[str, str], float],
        base_prompt: str,
        min_score: float = 0.5,
        verbose: bool = False,
    ) -> List[TrainingExample]:
        """
        Validate generated examples by running them through the model.

        Only keeps examples where the model produces output similar to expected.

        Args:
            examples: Generated examples to validate
            metric_fn: Metric function for scoring
            base_prompt: The prompt to use for validation
            min_score: Minimum score to accept example
            verbose: Print progress

        Returns:
            List of validated TrainingExample objects
        """
        validated = []

        for i, example in enumerate(examples):
            full_prompt = f"{base_prompt}\n\n## Input\n\n{example.input_text}"
            result = self.runner.run(full_prompt)

            if result.success:
                score = metric_fn(example.expected_output, result.output)
                if score >= min_score:
                    validated.append(TrainingExample(
                        input_text=example.input_text,
                        expected_output=example.expected_output,
                        metadata={
                            **(example.metadata or {}),
                            "source": example.source,
                            "validation_score": score,
                        },
                    ))

                if verbose:
                    status = "PASS" if score >= min_score else "FAIL"
                    print(f"  [{i+1}/{len(examples)}] {status} (score: {score:.2f})")

        if verbose:
            print(f"Validated {len(validated)}/{len(examples)} examples")

        return validated


class IterativeOptimizer:
    """
    Multi-round optimization with data augmentation.

    Runs multiple optimization rounds where each round:
    1. Optimizes using current training data
    2. Uses successful demos to generate more synthetic data
    3. Expands training set for next round

    Continues until convergence or max rounds reached.
    """

    def __init__(
        self,
        runner: ClaudeRunner,
        max_rounds: int = 3,
        convergence_threshold: float = 0.01,
        storage: Optional[DemoStorage] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            runner: ClaudeRunner instance
            max_rounds: Maximum optimization rounds
            convergence_threshold: Score improvement threshold for convergence
            storage: Optional DemoStorage for persistence
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.runner = runner
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.storage = storage or DemoStorage()
        self.generator = SyntheticDataGenerator(runner)

    def _run_round(
        self,
        prompt: str,
        data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        threshold: float,
        max_demos: int,
        round_num: int,
        agent_name: str,
        verbose: bool,
    ) -> Tuple[BootstrapResult, float]:
        """
        Run a single optimization round.

        Args:
            prompt: Current prompt
            data: Training data
            metric_fn: Metric function
            threshold: Demo threshold
            max_demos: Max demos
            round_num: Current round number
            agent_name: Agent name for storage (avoids overwriting)
            verbose: Print progress

        Returns:
            Tuple of (BootstrapResult, score)
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"ROUND {round_num}")
            print(f"{'='*50}")
            print(f"Training examples: {len(data)}")

        bootstrap = BootstrapFewShot(
            runner=self.runner,
            max_demos=max_demos,
            storage=self.storage,
        )

        result = bootstrap.optimize(
            base_prompt=prompt,
            training_data=data,
            metric_fn=metric_fn,
            threshold=threshold,
            agent_name=f"{agent_name}_round_{round_num}",
            verbose=verbose,
        )

        return result, result.avg_score

    def _expand_training_data(
        self,
        demos: List[Demo],
        original_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        prompt: str,
        verbose: bool,
    ) -> List[TrainingExample]:
        """
        Expand training data using synthetic generation.

        Args:
            demos: Successful demos from current round
            original_data: Original training examples
            metric_fn: Metric function for validation
            prompt: Current prompt for validation
            verbose: Print progress

        Returns:
            Expanded training data list
        """
        if verbose:
            print(f"\nExpanding training data...")

        # Generate synthetic examples
        similar = self.generator.generate_similar(demos, n=10, verbose=verbose)
        edge_cases = self.generator.generate_edge_cases(demos, n=5, verbose=verbose)

        all_generated = similar + edge_cases

        # Validate generated examples
        if verbose:
            print(f"\nValidating {len(all_generated)} generated examples...")

        validated = self.generator.validate_generated(
            examples=all_generated,
            metric_fn=metric_fn,
            base_prompt=prompt,
            min_score=0.5,
            verbose=verbose,
        )

        # Combine with original data
        expanded = list(original_data) + validated

        if verbose:
            print(f"Expanded dataset: {len(original_data)} -> {len(expanded)} examples")
            if len(expanded) == len(original_data):
                print("Warning: No synthetic examples passed validation")

        return expanded

    def _check_convergence(self, scores: List[float]) -> bool:
        """
        Check if optimization has converged.

        Convergence requires:
        1. Non-negative improvement (not a regression)
        2. Improvement below threshold (plateaued)

        Args:
            scores: List of scores from each round

        Returns:
            True if converged (plateaued without regression)
        """
        if len(scores) < 2:
            return False

        # Check if improvement is non-negative but below threshold
        # Regressions (negative improvement) should NOT trigger convergence
        improvement = scores[-1] - scores[-2]
        return improvement >= 0 and improvement < self.convergence_threshold

    def optimize(
        self,
        base_prompt: str,
        training_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        threshold: float = 0.7,
        max_demos: int = 3,
        agent_name: str = "default",
        verbose: bool = True,
    ) -> IterativeResult:
        """
        Run iterative optimization.

        Algorithm:
        Round 1: BootstrapFewShot on original data -> demos_1
        Round 2: Use demos_1 to generate synthetic data -> data_2, optimize -> demos_2
        Round 3: Add edge cases -> data_3, optimize -> final_demos
        Continue until convergence or max_rounds

        Args:
            base_prompt: Base prompt to optimize
            training_data: Initial training examples
            metric_fn: Metric function
            threshold: Demo threshold
            max_demos: Maximum demos
            agent_name: Name for storage
            verbose: Print progress

        Returns:
            IterativeResult with final prompt and round history
        """
        if verbose:
            print(f"Starting iterative optimization...")
            print(f"Max rounds: {self.max_rounds}")
            print(f"Convergence threshold: {self.convergence_threshold}")
            print(f"Initial training examples: {len(training_data)}")

        current_prompt = base_prompt
        current_data = list(training_data)
        rounds: List[RoundResult] = []
        scores: List[float] = []
        converged = False
        convergence_round = None

        for round_num in range(1, self.max_rounds + 1):
            # Run optimization round
            result, score = self._run_round(
                prompt=current_prompt,
                data=current_data,
                metric_fn=metric_fn,
                threshold=threshold,
                max_demos=max_demos,
                round_num=round_num,
                agent_name=agent_name,
                verbose=verbose,
            )

            scores.append(score)

            # Record round result
            new_examples = len(current_data) - len(training_data)
            rounds.append(RoundResult(
                round_num=round_num,
                prompt=result.optimized_prompt.to_prompt(),
                score=score,
                demos=result.optimized_prompt.demos,
                training_size=len(current_data),
                new_examples=new_examples,
            ))

            # Check convergence
            if self._check_convergence(scores):
                converged = True
                convergence_round = round_num
                if verbose:
                    print(f"\nConverged at round {round_num}!")
                break

            # Prepare for next round (unless this is the last)
            if round_num < self.max_rounds:
                # Update prompt with best demos
                current_prompt = result.optimized_prompt.to_prompt()

                # Expand training data (accumulate across rounds)
                if result.optimized_prompt.demos:
                    current_data = self._expand_training_data(
                        demos=result.optimized_prompt.demos,
                        original_data=current_data,  # Accumulate from current, not original
                        metric_fn=metric_fn,
                        prompt=current_prompt,
                        verbose=verbose,
                    )

        # Use the best round's result
        best_round = max(rounds, key=lambda r: r.score)

        # Create final optimized prompt
        final_prompt = OptimizedPrompt(
            base_prompt=base_prompt,
            demos=best_round.demos,
            optimization_date=datetime.now().isoformat(),
            metric_name=metric_fn.__name__ if hasattr(metric_fn, '__name__') else "custom",
            threshold=threshold,
            avg_score=best_round.score,
            metadata={
                "optimization_type": "iterative",
                "total_rounds": len(rounds),
                "converged": converged,
                "best_round": best_round.round_num,
            },
        )

        # Save to storage
        if best_round.demos:
            self.storage.save_demos(agent_name, best_round.demos)
            self.storage.save_optimized_prompt(agent_name, final_prompt)

        if verbose:
            print(f"\n{'='*50}")
            print(f"ITERATIVE OPTIMIZATION COMPLETE")
            print(f"{'='*50}")
            print(f"Total rounds: {len(rounds)}")
            print(f"Converged: {converged}")
            print(f"Best round: {best_round.round_num}")
            print(f"Final score: {best_round.score:.3f}")
            print(f"Round scores: {[f'{s:.3f}' for s in scores]}")

        return IterativeResult(
            final_prompt=final_prompt,
            final_score=best_round.score,
            rounds=rounds,
            total_rounds=len(rounds),
            converged=converged,
            convergence_round=convergence_round,
        )

    def optimize_with_copro(
        self,
        base_prompt: str,
        training_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        threshold: float = 0.7,
        max_demos: int = 3,
        n_variants: int = 5,
        agent_name: str = "default",
        verbose: bool = True,
    ) -> Tuple[COPROResult, IterativeResult]:
        """
        Combined COPRO + Iterative optimization.

        First optimizes the instruction using COPRO, then applies
        iterative refinement with data augmentation.

        Args:
            base_prompt: Base prompt to optimize
            training_data: Training examples
            metric_fn: Metric function
            threshold: Demo threshold
            max_demos: Maximum demos
            n_variants: Number of COPRO variants
            agent_name: Name for storage
            verbose: Print progress

        Returns:
            Tuple of (COPROResult, IterativeResult)
        """
        if verbose:
            print(f"{'='*50}")
            print(f"COMBINED COPRO + ITERATIVE OPTIMIZATION")
            print(f"{'='*50}")

        # Step 1: COPRO optimization
        copro = COPROOptimizer(
            runner=self.runner,
            n_variants=n_variants,
            storage=self.storage,
        )

        copro_result = copro.optimize(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            agent_name=f"{agent_name}_copro",
            verbose=verbose,
        )

        # Step 2: Iterative refinement with best COPRO variant
        if verbose:
            print(f"\n{'='*50}")
            print(f"Starting iterative refinement with best COPRO variant...")
            print(f"{'='*50}")

        iterative_result = self.optimize(
            base_prompt=copro_result.best_variant,
            training_data=training_data,
            metric_fn=metric_fn,
            threshold=threshold,
            max_demos=max_demos,
            agent_name=agent_name,
            verbose=verbose,
        )

        return copro_result, iterative_result
