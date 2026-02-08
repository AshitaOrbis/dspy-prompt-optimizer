"""
BootstrapFewShot implementation - the core DSPy optimization algorithm.

This implements automated few-shot example generation by:
1. Running the model on training examples
2. Collecting successful traces where metric >= threshold
3. Using successful traces as few-shot examples for future runs

Phase 5 additions:
- Probe holdout for early overfitting detection
- K-fold cross-validation during optimization
- Example dropout regularization
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Dict, Any

from .claude_runner import ClaudeRunner, RunResult
from .storage import Demo, OptimizedPrompt, DemoStorage

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example for optimization."""
    input_text: str
    expected_output: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BootstrapResult:
    """Result from a bootstrap optimization run."""
    optimized_prompt: OptimizedPrompt
    total_examples: int
    successful_examples: int
    failed_examples: int
    avg_score: float
    traces: List[Tuple[TrainingExample, str, float]]  # (example, output, score)


class BootstrapFewShot:
    """
    BootstrapFewShot optimizer.

    Automatically generates few-shot examples by running the model on training data
    and collecting successful outputs to use as demonstrations.

    Phase 5 features:
    - Probe holdout for early overfitting detection
    - K-fold cross-validation during optimization
    - Example dropout regularization for robustness

    Phase 6 features (metric mismatch fix):
    - Demo transformer for post-processing verbose outputs
    - Format instruction for guiding structured output
    """

    def __init__(
        self,
        runner: ClaudeRunner,
        max_demos: int = 3,
        storage: Optional[DemoStorage] = None,
        probe_holdout: Optional[List['TrainingExample']] = None,
        probe_check_interval: int = 5,
        max_overfitting_gap: float = 0.3,
        dropout_rate: float = 0.0,
        demo_transformer: Optional[Callable] = None,
        format_instruction: str = "",
    ):
        """
        Initialize the optimizer.

        Args:
            runner: ClaudeRunner instance for model calls
            max_demos: Maximum number of demos to include in optimized prompt
            storage: Optional DemoStorage for persistence
            probe_holdout: Optional probe examples for overfitting detection
            probe_check_interval: Check probe every N examples
            max_overfitting_gap: Max train-probe score gap before warning
            dropout_rate: Probability of dropping each demo (0.0-0.5)
            demo_transformer: Optional function to transform verbose demos into
                            structured format. Signature: (input, output, metadata) -> TransformedDemo
            format_instruction: Optional instruction string to append to prompts
                              to guide Claude toward structured output format
        """
        self.runner = runner
        self.max_demos = max_demos
        self.storage = storage or DemoStorage()
        self.probe_holdout = probe_holdout
        self.probe_check_interval = probe_check_interval
        self.max_overfitting_gap = max_overfitting_gap
        self.dropout_rate = min(0.5, max(0.0, dropout_rate))  # Cap at 50%
        self.demo_transformer = demo_transformer
        self.format_instruction = format_instruction
        self._overfitting_warnings: List[str] = []

    def optimize(
        self,
        base_prompt: str,
        training_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        threshold: float = 0.7,
        agent_name: str = "default",
        verbose: bool = True,
    ) -> BootstrapResult:
        """
        Optimize a prompt using BootstrapFewShot.

        Args:
            base_prompt: The base prompt/instruction to optimize
            training_data: List of training examples
            metric_fn: Function(expected, actual) -> score in [0,1]
            threshold: Minimum score to include as demo
            agent_name: Name for storage/tracking
            verbose: Whether to print progress

        Returns:
            BootstrapResult with optimized prompt and metrics
        """
        successful_demos: List[Tuple[TrainingExample, str, float]] = []
        all_traces: List[Tuple[TrainingExample, str, float]] = []
        failed_count = 0
        self._overfitting_warnings = []  # Reset warnings

        if verbose:
            print(f"Running BootstrapFewShot on {len(training_data)} examples...")
            print(f"Threshold: {threshold}, Max demos: {self.max_demos}")
            if self.probe_holdout:
                print(f"Probe holdout: {len(self.probe_holdout)} examples (check every {self.probe_check_interval})")
            if self.dropout_rate > 0:
                print(f"Dropout rate: {self.dropout_rate:.1%}")
            if self.format_instruction:
                print(f"Format instruction: enabled ({len(self.format_instruction)} chars)")
            if self.demo_transformer:
                print(f"Demo transformer: enabled")

        # Apply format instruction to base prompt if provided
        effective_prompt = base_prompt
        if self.format_instruction:
            if "## Output Format" not in base_prompt:
                effective_prompt = f"{base_prompt}\n\n{self.format_instruction}"

        for i, example in enumerate(training_data):
            if verbose:
                print(f"  [{i+1}/{len(training_data)}] Processing example...", end=" ")

            # Run the model with effective prompt + example input
            full_prompt = f"{effective_prompt}\n\n## Input\n\n{example.input_text}"
            result = self.runner.run(full_prompt)

            if not result.success:
                failed_count += 1
                if verbose:
                    print(f"FAILED: {result.error}")
                continue

            # Score the output
            score = metric_fn(example.expected_output, result.output)
            all_traces.append((example, result.output, score))

            if verbose:
                print(f"Score: {score:.2f}", end="")

            if score >= threshold:
                successful_demos.append((example, result.output, score))
                if verbose:
                    print(" [DEMO]")

                # Probe check for overfitting detection
                if self.probe_holdout and (i + 1) % self.probe_check_interval == 0:
                    # Build temporary demo list for probe evaluation
                    temp_demos = [
                        Demo(
                            input_text=ex.input_text,
                            output_text=output,
                            score=s,
                            metadata=ex.metadata,
                        )
                        for ex, output, s in successful_demos[-self.max_demos:]
                    ]
                    train_avg = sum(t[2] for t in successful_demos) / len(successful_demos)
                    probe_score = self._evaluate_on_probe(base_prompt, temp_demos, metric_fn)

                    if self._check_overfitting(train_avg, probe_score, i + 1):
                        if verbose:
                            print(f"    [WARNING] Overfitting: train={train_avg:.3f}, probe={probe_score:.3f}")
            else:
                if verbose:
                    print("")

        if verbose:
            print(f"\nCollected {len(successful_demos)} demos from {len(training_data)} examples")
            if self._overfitting_warnings:
                print(f"Overfitting warnings: {len(self._overfitting_warnings)}")

        # Select best demos
        best_demos = self._select_best_demos(successful_demos)

        # Apply demo transformer if provided
        if self.demo_transformer and best_demos:
            if verbose:
                print("Applying demo transformer to condense outputs...")
            transformed_demos = []
            for ex, output, score in best_demos:
                try:
                    transformed = self.demo_transformer(
                        ex.input_text,
                        output,
                        ex.metadata,
                    )
                    # Use transformed output
                    transformed_demos.append((ex, transformed.output_text, score))
                    if verbose:
                        orig_len = len(output)
                        new_len = len(transformed.output_text)
                        print(f"  Transformed: {orig_len} -> {new_len} chars ({100*new_len/orig_len:.0f}%)")
                except Exception as e:
                    # Keep original if transformation fails
                    transformed_demos.append((ex, output, score))
                    if verbose:
                        print(f"  Transform failed: {e}, keeping original")
            best_demos = transformed_demos

        # Create optimized prompt
        demo_objects = [
            Demo(
                input_text=ex.input_text,
                output_text=output,
                score=score,
                metadata=ex.metadata,
            )
            for ex, output, score in best_demos
        ]

        avg_score = (
            sum(d.score for d in demo_objects) / len(demo_objects)
            if demo_objects else 0.0
        )

        optimized = OptimizedPrompt(
            base_prompt=base_prompt,
            demos=demo_objects,
            optimization_date=datetime.now().isoformat(),
            metric_name=metric_fn.__name__ if hasattr(metric_fn, '__name__') else "custom",
            threshold=threshold,
            avg_score=avg_score,
            format_instruction=self.format_instruction,  # Phase 6: Save for evaluation
        )

        # Save to storage
        if demo_objects:
            self.storage.save_demos(agent_name, demo_objects)
            self.storage.save_optimized_prompt(agent_name, optimized)
            if verbose:
                print(f"Saved optimized prompt for '{agent_name}'")

        return BootstrapResult(
            optimized_prompt=optimized,
            total_examples=len(training_data),
            successful_examples=len(successful_demos),
            failed_examples=failed_count,
            avg_score=avg_score,
            traces=all_traces,
        )

    def _select_best_demos(
        self,
        demos: List[Tuple[TrainingExample, str, float]],
        use_diversity: bool = True,
    ) -> List[Tuple[TrainingExample, str, float]]:
        """
        Select the best demos for inclusion in the optimized prompt.

        Uses a combination of:
        1. Highest scores
        2. Diversity of examples (if metadata available)

        Args:
            demos: List of (TrainingExample, output, score) tuples
            use_diversity: Whether to apply diversity-based selection

        Returns:
            Selected demos list
        """
        if len(demos) <= self.max_demos:
            return sorted(demos, key=lambda x: x[2], reverse=True)

        if use_diversity:
            # Use diversity-aware selection
            from .diversity import select_diverse_demos
            return select_diverse_demos(demos, max_demos=self.max_demos)

        # Fallback: simple top-N selection
        sorted_demos = sorted(demos, key=lambda x: x[2], reverse=True)
        return sorted_demos[:self.max_demos]

    def _apply_dropout(
        self,
        demos: List[Demo],
    ) -> List[Demo]:
        """
        Apply random dropout to demos for regularization.

        Randomly drops examples during training to improve robustness
        and prevent overfitting to specific examples.

        Args:
            demos: List of Demo objects

        Returns:
            Demos with some randomly dropped
        """
        if self.dropout_rate <= 0 or len(demos) <= 1:
            return demos

        # Ensure we keep at least one demo
        keep_count = max(1, int(len(demos) * (1 - self.dropout_rate)))
        return random.sample(demos, keep_count)

    def _evaluate_on_probe(
        self,
        base_prompt: str,
        demos: List[Demo],
        metric_fn: Callable[[str, str], float],
    ) -> float:
        """
        Evaluate current prompt configuration on probe holdout set.

        Args:
            base_prompt: The base instruction
            demos: Current demo set
            metric_fn: Metric function

        Returns:
            Average score on probe set
        """
        if not self.probe_holdout:
            return 0.0

        # Build the test prompt with demos
        optimized = OptimizedPrompt(
            base_prompt=base_prompt,
            demos=demos,
            optimization_date=datetime.now().isoformat(),
            metric_name=metric_fn.__name__ if hasattr(metric_fn, '__name__') else "custom",
            threshold=0.0,
            avg_score=0.0,
        )
        test_prompt = optimized.to_prompt()

        scores = []
        for example in self.probe_holdout:
            full_prompt = f"{test_prompt}\n\n## New Input\n\n{example.input_text}"
            result = self.runner.run(full_prompt)

            if result.success:
                score = metric_fn(example.expected_output, result.output)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _check_overfitting(
        self,
        train_score: float,
        probe_score: float,
        example_idx: int,
    ) -> bool:
        """
        Check for overfitting and log warning if detected.

        Args:
            train_score: Score on training data
            probe_score: Score on probe holdout
            example_idx: Current example index

        Returns:
            True if overfitting detected
        """
        gap = train_score - probe_score

        if gap > self.max_overfitting_gap:
            warning = f"Overfitting detected at example {example_idx}: train={train_score:.3f}, probe={probe_score:.3f}, gap={gap:.3f}"
            self._overfitting_warnings.append(warning)
            logger.warning(warning)
            return True

        return False

    def optimize_with_cv(
        self,
        base_prompt: str,
        training_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        k: int = 3,
        min_fold_score: float = 0.5,
        threshold: float = 0.7,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimize with K-fold cross-validation to prevent overfitting.

        Evaluates the optimization approach on multiple train/test splits
        to ensure consistent performance and detect overfitting.

        Args:
            base_prompt: Base prompt to optimize
            training_data: All training examples
            metric_fn: Metric function
            k: Number of folds (default: 3)
            min_fold_score: Minimum acceptable score per fold
            threshold: Score threshold for demo selection
            verbose: Print progress

        Returns:
            Dict with fold_scores, mean_score, std_score, low_fold_warning
        """
        if k < 2:
            raise ValueError("k must be >= 2 for cross-validation")
        if k > len(training_data):
            raise ValueError(f"k ({k}) cannot exceed training data size ({len(training_data)})")

        if verbose:
            print(f"Running {k}-fold cross-validation...")

        # Shuffle data
        shuffled = training_data.copy()
        random.shuffle(shuffled)

        # Split into k folds
        fold_size = len(shuffled) // k
        folds = []
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(shuffled)
            folds.append(shuffled[start:end])

        fold_scores = []
        low_fold_warning = False

        for fold_idx in range(k):
            if verbose:
                print(f"\nFold {fold_idx + 1}/{k}:")

            # Create train/test split
            test_fold = folds[fold_idx]
            train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
            train_data = [ex for fold in train_folds for ex in fold]

            # Train on training folds
            result = self.optimize(
                base_prompt=base_prompt,
                training_data=train_data,
                metric_fn=metric_fn,
                threshold=threshold,
                agent_name=f"cv_fold_{fold_idx}",
                verbose=False,
            )

            # Evaluate on test fold
            if result.optimized_prompt.demos:
                test_prompt = result.optimized_prompt.to_prompt()
            else:
                test_prompt = base_prompt

            scores = []
            for example in test_fold:
                full_prompt = f"{test_prompt}\n\n## Input\n\n{example.input_text}"
                run_result = self.runner.run(full_prompt)

                if run_result.success:
                    score = metric_fn(example.expected_output, run_result.output)
                    scores.append(score)

            fold_score = sum(scores) / len(scores) if scores else 0.0
            fold_scores.append(fold_score)

            if fold_score < min_fold_score:
                low_fold_warning = True

            if verbose:
                status = "OK" if fold_score >= min_fold_score else "LOW"
                print(f"  Fold {fold_idx + 1} score: {fold_score:.3f} [{status}]")

        # Calculate statistics
        mean_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
        variance = sum((s - mean_score) ** 2 for s in fold_scores) / len(fold_scores) if fold_scores else 0.0
        std_score = variance ** 0.5

        if verbose:
            print(f"\nCV Results:")
            print(f"  Mean: {mean_score:.3f}")
            print(f"  Std:  {std_score:.3f}")
            if low_fold_warning:
                print(f"  WARNING: Some folds below min_fold_score ({min_fold_score})")

        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'low_fold_warning': low_fold_warning,
            'k': k,
            'min_fold_score': min_fold_score,
        }

    def optimize_with_holdout(
        self,
        base_prompt: str,
        training_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        holdout_ratio: float = 0.2,
        threshold: float = 0.7,
        agent_name: str = "default",
        verbose: bool = True,
    ) -> Tuple[BootstrapResult, float]:
        """
        Optimize with holdout set for validation.

        Args:
            base_prompt: Base prompt to optimize
            training_data: All training examples
            metric_fn: Metric function
            holdout_ratio: Fraction to use for validation
            threshold: Score threshold for demos
            agent_name: Name for storage
            verbose: Print progress

        Returns:
            Tuple of (BootstrapResult, holdout_score)
        """
        import random

        # Split data
        shuffled = training_data.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - holdout_ratio))
        train_set = shuffled[:split_idx]
        holdout_set = shuffled[split_idx:]

        if verbose:
            print(f"Split: {len(train_set)} train, {len(holdout_set)} holdout")

        # Optimize on training set
        result = self.optimize(
            base_prompt=base_prompt,
            training_data=train_set,
            metric_fn=metric_fn,
            threshold=threshold,
            agent_name=agent_name,
            verbose=verbose,
        )

        # Evaluate on holdout
        if verbose:
            print(f"\nEvaluating on holdout set...")

        holdout_scores = []
        optimized_prompt = result.optimized_prompt.to_prompt()

        for example in holdout_set:
            full_prompt = f"{optimized_prompt}\n\n## New Input\n\n{example.input_text}"
            run_result = self.runner.run(full_prompt)

            if run_result.success:
                score = metric_fn(example.expected_output, run_result.output)
                holdout_scores.append(score)

        holdout_avg = sum(holdout_scores) / len(holdout_scores) if holdout_scores else 0.0

        if verbose:
            print(f"Holdout score: {holdout_avg:.2f} ({len(holdout_scores)} evaluated)")

        return result, holdout_avg

    def compare_baseline(
        self,
        base_prompt: str,
        test_data: List[TrainingExample],
        optimized: OptimizedPrompt,
        metric_fn: Callable[[str, str], float],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Compare optimized prompt against baseline.

        Args:
            base_prompt: Original prompt without demos
            test_data: Test examples
            optimized: Optimized prompt with demos
            metric_fn: Metric function
            verbose: Print progress

        Returns:
            Dict with baseline_score, optimized_score, improvement
        """
        baseline_scores = []
        optimized_scores = []

        if verbose:
            print(f"Comparing on {len(test_data)} test examples...")

        for i, example in enumerate(test_data):
            # Baseline run
            baseline_prompt = f"{base_prompt}\n\n## Input\n\n{example.input_text}"
            baseline_result = self.runner.run(baseline_prompt)

            if baseline_result.success:
                score = metric_fn(example.expected_output, baseline_result.output)
                baseline_scores.append(score)

            # Optimized run
            optimized_full = f"{optimized.to_prompt()}\n\n## New Input\n\n{example.input_text}"
            optimized_result = self.runner.run(optimized_full)

            if optimized_result.success:
                score = metric_fn(example.expected_output, optimized_result.output)
                optimized_scores.append(score)

            if verbose and (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(test_data)}")

        baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        optimized_avg = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0
        improvement = optimized_avg - baseline_avg

        results = {
            "baseline_score": baseline_avg,
            "optimized_score": optimized_avg,
            "improvement": improvement,
            "improvement_pct": (improvement / baseline_avg * 100) if baseline_avg > 0 else 0.0,
            "baseline_n": len(baseline_scores),
            "optimized_n": len(optimized_scores),
        }

        if verbose:
            print(f"\nResults:")
            print(f"  Baseline:  {baseline_avg:.3f} (n={len(baseline_scores)})")
            print(f"  Optimized: {optimized_avg:.3f} (n={len(optimized_scores)})")
            print(f"  Improvement: {improvement:+.3f} ({results['improvement_pct']:+.1f}%)")

        return results
