"""
Verification framework for validating prompt optimizations.

Provides holdout evaluation, cross-validation, and regression testing
to ensure optimized prompts perform reliably.
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

from .claude_runner import ClaudeRunner
from .bootstrap import TrainingExample, BootstrapFewShot
from .storage import DemoStorage, OptimizedPrompt, Demo
from .utils import parse_markdown_prompt, find_agent_path

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Explicit outcome for every requested verification operation."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


@dataclass
class EvaluationResult:
    """Outcome for one requested example; failed calls retain a zero score."""

    index: int
    status: VerificationStatus
    score: float
    error: Optional[str] = None


@dataclass
class HoldoutResult:
    """Result from holdout evaluation."""
    agent_name: str
    status: VerificationStatus
    holdout_score: float
    holdout_size: int
    successful_evals: int
    failed_evals: int
    scores: List[float]
    evaluations: List[EvaluationResult] = field(default_factory=list)
    error: Optional[str] = None
    error_category: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Compatibility view; callers should consume ``status``."""
        return self.status == VerificationStatus.PASS


@dataclass
class CrossValidationResult:
    """Result from k-fold cross-validation."""
    agent_name: str
    status: VerificationStatus
    mean_score: float
    std_score: float
    fold_scores: List[float]
    k: int
    total_examples: int
    evaluations: List[EvaluationResult] = field(default_factory=list)
    error: Optional[str] = None
    error_category: Optional[str] = None


@dataclass
class RegressionResult:
    """Result from regression check."""
    agent_name: str
    status: VerificationStatus
    current_score: Optional[float]
    baseline_score: Optional[float]
    improvement: Optional[float]
    regressed: bool
    threshold: float
    error: Optional[str] = None


@dataclass
class PreFlightResult:
    """Typed holdout-gate result with compatibility tuple unpacking."""

    status: VerificationStatus
    should_deploy: bool
    new_score: float
    existing_score: Optional[float]
    new_evaluations: List[EvaluationResult] = field(default_factory=list)
    existing_evaluations: List[EvaluationResult] = field(default_factory=list)
    error: Optional[str] = None

    def __iter__(self):
        yield self.should_deploy
        yield self.new_score
        yield self.existing_score


@dataclass
class VerificationReport:
    """Complete verification report."""
    timestamp: str
    agents_verified: int
    holdout_results: List[HoldoutResult]
    regression_results: List[RegressionResult]
    cross_validation_results: List[CrossValidationResult]
    summary: Dict[str, Any]

    def has_blocking_failures(self, allow_missing: bool = False) -> bool:
        """Return whether automation must receive a non-zero outcome."""
        for result in self.holdout_results:
            if result.status == VerificationStatus.FAIL:
                return True
            if result.status in (VerificationStatus.NOT_RUN, VerificationStatus.ERROR):
                if not (allow_missing and result.error_category == "evidence"):
                    return True
        for result in self.regression_results:
            if result.status == VerificationStatus.FAIL or result.regressed:
                return True
            if result.status == VerificationStatus.NOT_ASSESSABLE and allow_missing:
                continue
            if result.status != VerificationStatus.PASS:
                return True
        for result in self.cross_validation_results:
            if result.status != VerificationStatus.PASS:
                return True
        return False


class VerificationSuite:
    """
    Comprehensive verification suite for prompt optimizations.

    Provides multiple validation strategies:
    - Holdout evaluation: Test on held-out data
    - Cross-validation: K-fold validation for robustness
    - Regression testing: Compare against baseline scores
    """

    def __init__(
        self,
        runner: ClaudeRunner,
        storage: Optional[DemoStorage] = None,
        baseline_scores_path: Optional[Path] = None,
    ):
        """
        Initialize the verification suite.

        Args:
            runner: ClaudeRunner for evaluation
            storage: DemoStorage for loading optimized prompts
            baseline_scores_path: Path to JSON file with baseline scores
        """
        self.runner = runner
        self.storage = storage or DemoStorage()
        self.baseline_scores_path = baseline_scores_path
        self._baseline_scores: Optional[Dict[str, float]] = None
        self._baseline_load_error: Optional[str] = None
        self._baseline_entry_errors: Dict[str, str] = {}

    def _load_baseline_scores(self) -> Dict[str, float]:
        """Load only finite scores in [0, 1], preserving unavailable state."""
        if self._baseline_scores is not None:
            return self._baseline_scores

        self._baseline_scores = {}
        self._baseline_entry_errors = {}
        if self.baseline_scores_path is None:
            self._baseline_load_error = "baseline scores path is not configured"
            return self._baseline_scores
        if not self.baseline_scores_path.exists():
            self._baseline_load_error = (
                f"baseline scores file does not exist: {self.baseline_scores_path}"
            )
            return self._baseline_scores

        try:
            with open(self.baseline_scores_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("baseline scores must be a JSON object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self._baseline_load_error = f"could not load baseline scores: {exc}"
            return self._baseline_scores

        self._baseline_load_error = None
        for key, value in data.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self._baseline_entry_errors[key] = "baseline must be numeric"
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                self._baseline_entry_errors[key] = "baseline must be finite"
                continue
            if not 0.0 <= numeric <= 1.0:
                self._baseline_entry_errors[key] = "baseline must be in [0, 1]"
                continue
            self._baseline_scores[key] = numeric

        return self._baseline_scores

    def _save_baseline_score(self, agent_name: str, score: float):
        """Explicitly create or update a validated baseline score."""
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("baseline score must be numeric")
        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError("baseline score must be finite and in [0, 1]")
        if self.baseline_scores_path is None:
            raise ValueError("baseline scores path is required for an update")
        scores = self._load_baseline_scores()
        scores[agent_name] = float(score)

        self.baseline_scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_scores_path, 'w') as f:
            json.dump(scores, f, indent=2)
        self._baseline_load_error = None
        self._baseline_entry_errors.pop(agent_name, None)

    def load_holdout_data(self, path: Path) -> List[TrainingExample]:
        """
        Load holdout data from JSONL file.

        Args:
            path: Path to holdout JSONL file

        Returns:
            List of TrainingExample objects
        """
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

    def run_holdout_evaluation(
        self,
        agent_name: str,
        holdout_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        pass_threshold: float = 0.7,
        verbose: bool = True,
    ) -> HoldoutResult:
        """
        Evaluate an optimized agent on holdout data.

        Args:
            agent_name: Name of the agent to evaluate
            holdout_data: Holdout examples
            metric_fn: Metric function
            pass_threshold: Minimum score to pass
            verbose: Print progress

        Returns:
            HoldoutResult with scores and pass/fail status
        """
        if verbose:
            print(f"\nEvaluating {agent_name} on holdout set...")

        if not holdout_data:
            return HoldoutResult(
                agent_name=agent_name,
                status=VerificationStatus.NOT_RUN,
                holdout_score=0.0,
                holdout_size=0,
                successful_evals=0,
                failed_evals=0,
                scores=[],
                error="holdout data is empty",
                error_category="evidence",
            )

        try:
            optimized = self.storage.load_optimized_prompt(agent_name)
        except Exception as exc:
            return HoldoutResult(
                agent_name=agent_name,
                status=VerificationStatus.ERROR,
                holdout_score=0.0,
                holdout_size=len(holdout_data),
                successful_evals=0,
                failed_evals=len(holdout_data),
                scores=[0.0] * len(holdout_data),
                evaluations=[
                    EvaluationResult(i, VerificationStatus.NOT_RUN, 0.0, str(exc))
                    for i in range(len(holdout_data))
                ],
                error=f"could not load optimized prompt: {exc}",
                error_category="configuration",
            )
        if not optimized:
            error = f"no optimized prompt found for {agent_name}"
            return HoldoutResult(
                agent_name=agent_name,
                status=VerificationStatus.NOT_RUN,
                holdout_score=0.0,
                holdout_size=len(holdout_data),
                successful_evals=0,
                failed_evals=len(holdout_data),
                scores=[0.0] * len(holdout_data),
                evaluations=[
                    EvaluationResult(i, VerificationStatus.NOT_RUN, 0.0, error)
                    for i in range(len(holdout_data))
                ],
                error=error,
                error_category="configuration",
            )

        if not callable(metric_fn):
            error = f"metric_fn must be callable, got {type(metric_fn).__name__}"
            return HoldoutResult(
                agent_name=agent_name,
                status=VerificationStatus.NOT_RUN,
                holdout_score=0.0,
                holdout_size=len(holdout_data),
                successful_evals=0,
                failed_evals=len(holdout_data),
                scores=[0.0] * len(holdout_data),
                evaluations=[
                    EvaluationResult(i, VerificationStatus.NOT_RUN, 0.0, error)
                    for i in range(len(holdout_data))
                ],
                error=error,
                error_category="configuration",
            )

        if (
            isinstance(pass_threshold, bool)
            or not isinstance(pass_threshold, (int, float))
            or not math.isfinite(float(pass_threshold))
            or not 0.0 <= float(pass_threshold) <= 1.0
        ):
            error = "pass threshold must be finite and in [0, 1]"
            return HoldoutResult(
                agent_name=agent_name,
                status=VerificationStatus.ERROR,
                holdout_score=0.0,
                holdout_size=len(holdout_data),
                successful_evals=0,
                failed_evals=len(holdout_data),
                scores=[0.0] * len(holdout_data),
                evaluations=[
                    EvaluationResult(i, VerificationStatus.ERROR, 0.0, error)
                    for i in range(len(holdout_data))
                ],
                error=error,
                error_category="configuration",
            )

        try:
            prompt = optimized.to_prompt()
        except Exception as exc:
            error = f"could not render optimized prompt: {exc}"
            return HoldoutResult(
                agent_name=agent_name,
                status=VerificationStatus.ERROR,
                holdout_score=0.0,
                holdout_size=len(holdout_data),
                successful_evals=0,
                failed_evals=len(holdout_data),
                scores=[0.0] * len(holdout_data),
                evaluations=[
                    EvaluationResult(i, VerificationStatus.ERROR, 0.0, error)
                    for i in range(len(holdout_data))
                ],
                error=error,
                error_category="configuration",
            )
        scores: List[float] = []
        evaluations: List[EvaluationResult] = []

        for i, example in enumerate(holdout_data):
            full_prompt = f"{prompt}\n\n## New Input\n\n{example.input_text}"
            try:
                run_result = self.runner.run(full_prompt)
                if not run_result.success:
                    raise RuntimeError(run_result.error or "model call failed")
                score = metric_fn(example.expected_output, run_result.output)
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError("metric returned a non-numeric score")
                score = float(score)
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("metric score must be finite and in [0, 1]")
                evaluations.append(
                    EvaluationResult(i, VerificationStatus.PASS, score)
                )
                scores.append(score)
                if verbose:
                    print(f"  [{i+1}/{len(holdout_data)}] Score: {score:.3f}")
            except Exception as exc:
                # Coverage rule: retain the example and a zero in the denominator.
                evaluations.append(
                    EvaluationResult(i, VerificationStatus.ERROR, 0.0, str(exc))
                )
                scores.append(0.0)
                if verbose:
                    print(f"  [{i+1}/{len(holdout_data)}] ERROR: {exc}")

        successful_count = sum(
            item.status == VerificationStatus.PASS for item in evaluations
        )
        failed_count = len(evaluations) - successful_count
        avg_score = sum(scores) / len(holdout_data)
        if failed_count:
            status = VerificationStatus.ERROR
            error = f"{failed_count} of {len(holdout_data)} evaluations failed"
        elif avg_score >= pass_threshold:
            status = VerificationStatus.PASS
            error = None
        else:
            status = VerificationStatus.FAIL
            error = f"holdout score {avg_score:.3f} is below {pass_threshold:.3f}"

        if verbose:
            print("\nHoldout Results:")
            print(f"  Average score: {avg_score:.3f}")
            print(f"  Evaluated: {successful_count}/{len(holdout_data)}")
            print(f"  Status: {status.value}")

        return HoldoutResult(
            agent_name=agent_name,
            status=status,
            holdout_score=avg_score,
            holdout_size=len(holdout_data),
            successful_evals=successful_count,
            failed_evals=failed_count,
            scores=scores,
            evaluations=evaluations,
            error=error,
            error_category="evaluation" if failed_count else None,
        )

    def run_cross_validation(
        self,
        agent_name: str,
        training_data: List[TrainingExample],
        metric_fn: Callable[[str, str], float],
        k: int = 5,
        threshold: float = 0.7,
        max_demos: int = 3,
        verbose: bool = True,
    ) -> CrossValidationResult:
        """
        Run k-fold cross-validation for robustness testing.

        Args:
            agent_name: Name of the agent
            training_data: All training data
            metric_fn: Metric function
            k: Number of folds
            threshold: Demo threshold
            max_demos: Maximum demos per fold
            verbose: Print progress

        Returns:
            CrossValidationResult with mean and std scores
        """
        def configuration_error(message: str) -> CrossValidationResult:
            return CrossValidationResult(
                agent_name=agent_name,
                status=VerificationStatus.ERROR,
                mean_score=0.0,
                std_score=0.0,
                fold_scores=[],
                k=k,
                total_examples=len(training_data),
                evaluations=[
                    EvaluationResult(i, VerificationStatus.ERROR, 0.0, message)
                    for i in range(len(training_data))
                ],
                error=message,
                error_category="configuration",
            )

        if k < 2:
            return configuration_error("k must be >= 2 for cross-validation")
        if k > len(training_data):
            return configuration_error(
                f"k ({k}) cannot exceed training data size ({len(training_data)})"
            )
        if not callable(metric_fn):
            return configuration_error(
                f"metric_fn must be callable, got {type(metric_fn).__name__}"
            )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            return configuration_error("threshold must be finite and in [0, 1]")

        if verbose:
            print(f"\nRunning {k}-fold cross-validation for {agent_name}...")

        # Load base prompt
        try:
            optimized = self.storage.load_optimized_prompt(agent_name)
            if optimized:
                base_prompt = optimized.base_prompt
                if not isinstance(base_prompt, str):
                    raise ValueError("optimized base prompt is not text")
            else:
                # Try to load from agent file using shared path resolution
                agent_path = find_agent_path(agent_name)
                with open(agent_path) as f:
                    content = f.read()
                base_prompt = parse_markdown_prompt(content)
        except Exception as exc:
            return configuration_error(f"cannot load base prompt for {agent_name}: {exc}")

        # Shuffle data
        shuffled = list(training_data)
        random.shuffle(shuffled)

        # Split into k folds
        fold_size = len(shuffled) // k
        folds = []
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(shuffled)
            folds.append(shuffled[start:end])

        fold_scores = []
        evaluations: List[EvaluationResult] = []
        evaluation_index = 0

        for fold_idx in range(k):
            if verbose:
                print(f"\nFold {fold_idx + 1}/{k}:")

            # Create train/test split
            test_fold = folds[fold_idx]
            train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
            train_data = [ex for fold in train_folds for ex in fold]

            # Train on training folds
            optimizer = BootstrapFewShot(
                runner=self.runner,
                max_demos=max_demos,
                storage=self.storage,
            )

            try:
                result = optimizer.optimize(
                    base_prompt=base_prompt,
                    training_data=train_data,
                    metric_fn=metric_fn,
                    threshold=threshold,
                    agent_name=f"{agent_name}_cv_fold{fold_idx}",
                    verbose=False,
                )
            except Exception as exc:
                for _example in test_fold:
                    evaluations.append(
                        EvaluationResult(
                            evaluation_index,
                            VerificationStatus.ERROR,
                            0.0,
                            f"fold optimization failed: {exc}",
                        )
                    )
                    evaluation_index += 1
                fold_scores.append(0.0)
                if verbose:
                    print(f"  Fold {fold_idx + 1} ERROR: {exc}")
                continue

            # Evaluate on test fold
            try:
                if result.optimized_prompt.demos:
                    test_prompt = result.optimized_prompt.to_prompt()
                else:
                    test_prompt = base_prompt
            except Exception as exc:
                for _example in test_fold:
                    evaluations.append(
                        EvaluationResult(
                            evaluation_index,
                            VerificationStatus.ERROR,
                            0.0,
                            f"fold prompt rendering failed: {exc}",
                        )
                    )
                    evaluation_index += 1
                fold_scores.append(0.0)
                if verbose:
                    print(f"  Fold {fold_idx + 1} ERROR: {exc}")
                continue

            scores = []
            for example in test_fold:
                full_prompt = f"{test_prompt}\n\n## Input\n\n{example.input_text}"
                try:
                    run_result = self.runner.run(full_prompt)
                    if not run_result.success:
                        raise RuntimeError(run_result.error or "model call failed")
                    score = metric_fn(example.expected_output, run_result.output)
                    if isinstance(score, bool) or not isinstance(score, (int, float)):
                        raise ValueError("metric returned a non-numeric score")
                    score = float(score)
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError("metric score must be finite and in [0, 1]")
                    scores.append(score)
                    evaluations.append(
                        EvaluationResult(
                            evaluation_index, VerificationStatus.PASS, score
                        )
                    )
                except Exception as exc:
                    scores.append(0.0)
                    evaluations.append(
                        EvaluationResult(
                            evaluation_index,
                            VerificationStatus.ERROR,
                            0.0,
                            str(exc),
                        )
                    )
                evaluation_index += 1

            fold_score = sum(scores) / len(scores) if scores else 0.0
            fold_scores.append(fold_score)

            if verbose:
                print(f"  Fold {fold_idx + 1} score: {fold_score:.3f}")

        # Calculate statistics
        mean_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
        variance = sum((s - mean_score) ** 2 for s in fold_scores) / len(fold_scores) if fold_scores else 0.0
        std_score = variance ** 0.5

        if verbose:
            print(f"\nCross-Validation Results:")
            print(f"  Mean score: {mean_score:.3f}")
            print(f"  Std dev: {std_score:.3f}")
            print(f"  Fold scores: {[f'{s:.3f}' for s in fold_scores]}")

        failed_count = sum(
            item.status == VerificationStatus.ERROR for item in evaluations
        )
        if failed_count:
            status = VerificationStatus.ERROR
            error = f"{failed_count} cross-validation evaluations failed"
        elif mean_score >= threshold:
            status = VerificationStatus.PASS
            error = None
        else:
            status = VerificationStatus.FAIL
            error = f"cross-validation score {mean_score:.3f} is below {threshold:.3f}"

        return CrossValidationResult(
            agent_name=agent_name,
            status=status,
            mean_score=mean_score,
            std_score=std_score,
            fold_scores=fold_scores,
            k=k,
            total_examples=len(training_data),
            evaluations=evaluations,
            error=error,
            error_category="evaluation" if failed_count else None,
        )

    def check_regression(
        self,
        agent_name: str,
        new_score: float,
        regression_threshold: float = 0.02,
        verbose: bool = True,
    ) -> RegressionResult:
        """
        Check if new score represents a regression from baseline.

        Args:
            agent_name: Name of the agent
            new_score: Score to check
            regression_threshold: Maximum acceptable score drop
            verbose: Print progress

        Returns:
            RegressionResult with comparison data
        """
        if (
            isinstance(regression_threshold, bool)
            or not isinstance(regression_threshold, (int, float))
            or not math.isfinite(float(regression_threshold))
            or not 0.0 <= float(regression_threshold) <= 1.0
        ):
            return RegressionResult(
                agent_name=agent_name,
                status=VerificationStatus.ERROR,
                current_score=float(new_score)
                if isinstance(new_score, (int, float)) and not isinstance(new_score, bool)
                else None,
                baseline_score=None,
                improvement=None,
                regressed=False,
                threshold=regression_threshold,
                error="regression threshold must be finite and in [0, 1]",
            )
        if (
            isinstance(new_score, bool)
            or not isinstance(new_score, (int, float))
            or not math.isfinite(float(new_score))
            or not 0.0 <= float(new_score) <= 1.0
        ):
            return RegressionResult(
                agent_name=agent_name,
                status=VerificationStatus.ERROR,
                current_score=None,
                baseline_score=None,
                improvement=None,
                regressed=False,
                threshold=regression_threshold,
                error="current score must be finite and in [0, 1]",
            )

        baseline_scores = self._load_baseline_scores()
        baseline_score = baseline_scores.get(agent_name)
        if baseline_score is None:
            error = (
                self._baseline_load_error
                or self._baseline_entry_errors.get(agent_name)
                or f"baseline has no valid entry for {agent_name}"
            )
            result = RegressionResult(
                agent_name=agent_name,
                status=VerificationStatus.NOT_ASSESSABLE,
                current_score=float(new_score),
                baseline_score=None,
                improvement=None,
                regressed=False,
                threshold=regression_threshold,
                error=error,
            )
            if verbose:
                print(f"\nRegression Check for {agent_name}:")
                print("  Baseline score: NOT_ASSESSABLE")
                print(f"  New score: {float(new_score):.3f}")
                print(f"  Status: {result.status.value} ({error})")
            return result

        improvement = float(new_score) - baseline_score
        regressed = improvement < -regression_threshold
        status = VerificationStatus.FAIL if regressed else VerificationStatus.PASS

        if verbose:
            print(f"\nRegression Check for {agent_name}:")
            print(f"  Baseline score: {baseline_score:.3f}")
            print(f"  New score: {float(new_score):.3f}")
            print(f"  Change: {improvement:+.3f}")
            print(f"  Threshold: -{regression_threshold:.3f}")
            print(f"  Status: {'REGRESSED' if regressed else status.value}")

        return RegressionResult(
            agent_name=agent_name,
            status=status,
            current_score=float(new_score),
            baseline_score=baseline_score,
            improvement=improvement,
            regressed=regressed,
            threshold=regression_threshold,
        )

    def update_baseline(
        self,
        agent_name: str,
        score: float,
        verbose: bool = True,
    ):
        """
        Update baseline score for an agent.

        Args:
            agent_name: Agent name
            score: New baseline score
            verbose: Print confirmation
        """
        self._save_baseline_score(agent_name, score)
        if verbose:
            print(f"Updated baseline for {agent_name}: {score:.3f}")

    def run_full_verification(
        self,
        agent_names: List[str],
        holdout_data_dir: Path,
        training_data_dir: Path,
        metric_fns: Dict[str, Callable[[str, str], float]],
        pass_threshold: float = 0.7,
        regression_threshold: float = 0.02,
        run_cross_validation: bool = False,
        check_regression: bool = True,
        k_folds: int = 5,
        verbose: bool = True,
        data_basenames: Optional[Dict[str, str]] = None,
    ) -> VerificationReport:
        """
        Run complete verification suite on multiple agents.

        Args:
            agent_names: List of agents to verify
            holdout_data_dir: Directory with holdout JSONL files
            training_data_dir: Directory with training JSONL files (for CV)
            metric_fns: Mapping of agent name to metric function
            pass_threshold: Holdout pass threshold
            regression_threshold: Regression threshold
            run_cross_validation: Whether to run CV
            k_folds: Number of CV folds
            verbose: Print progress

        Returns:
            VerificationReport with all results
        """
        if verbose:
            print("=" * 60)
            print("VERIFICATION SUITE")
            print("=" * 60)
            print(f"Agents: {agent_names}")
            print(f"Pass threshold: {pass_threshold}")
            print(f"Regression threshold: {regression_threshold}")
            print(f"Cross-validation: {run_cross_validation} ({k_folds} folds)")

        holdout_results: List[HoldoutResult] = []
        regression_results: List[RegressionResult] = []
        cv_results: List[CrossValidationResult] = []

        def unavailable_holdout(
            name: str,
            status: VerificationStatus,
            error: str,
            error_category: str = "evidence",
        ) -> HoldoutResult:
            return HoldoutResult(
                agent_name=name,
                status=status,
                holdout_score=0.0,
                holdout_size=0,
                successful_evals=0,
                failed_evals=0,
                scores=[],
                error=error,
                error_category=error_category,
            )

        def unavailable_regression(
            name: str,
            error: str,
            current_score: Optional[float] = None,
        ) -> RegressionResult:
            return RegressionResult(
                agent_name=name,
                status=VerificationStatus.NOT_ASSESSABLE,
                current_score=current_score,
                baseline_score=None,
                improvement=None,
                regressed=False,
                threshold=regression_threshold,
                error=error,
            )

        def unavailable_cv(
            name: str,
            status: VerificationStatus,
            error: str,
        ) -> CrossValidationResult:
            return CrossValidationResult(
                agent_name=name,
                status=status,
                mean_score=0.0,
                std_score=0.0,
                fold_scores=[],
                k=k_folds,
                total_examples=0,
                error=error,
                error_category="evidence",
            )

        for agent_name in agent_names:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Verifying: {agent_name}")
                print(f"{'='*60}")

            metric_fn = metric_fns.get(agent_name)
            if not metric_fn:
                error = f"no metric function for {agent_name}"
                if verbose:
                    print(f"  NOT_RUN: {error}")
                holdout_results.append(
                    unavailable_holdout(
                        agent_name,
                        VerificationStatus.NOT_RUN,
                        error,
                        error_category="configuration",
                    )
                )
                if check_regression:
                    regression_results.append(
                        unavailable_regression(agent_name, error)
                    )
                if run_cross_validation:
                    cv_results.append(
                        unavailable_cv(agent_name, VerificationStatus.NOT_RUN, error)
                    )
                continue

            # Holdout evaluation — resolve via data_basenames mapping or direct name
            data_basename = data_basenames.get(agent_name, agent_name) if data_basenames else agent_name
            holdout_path = holdout_data_dir / f"{data_basename}-holdout.jsonl"
            if not holdout_path.exists():
                # Fallback: try agent name directly
                holdout_path = holdout_data_dir / f"{agent_name}-holdout.jsonl"
            if holdout_path.exists():
                try:
                    holdout_data = self.load_holdout_data(holdout_path)
                    holdout_result = self.run_holdout_evaluation(
                        agent_name=agent_name,
                        holdout_data=holdout_data,
                        metric_fn=metric_fn,
                        pass_threshold=pass_threshold,
                        verbose=verbose,
                    )
                    holdout_results.append(holdout_result)

                    if check_regression:
                        if holdout_result.status in (
                            VerificationStatus.PASS,
                            VerificationStatus.FAIL,
                        ):
                            regression_results.append(
                                self.check_regression(
                                    agent_name=agent_name,
                                    new_score=holdout_result.holdout_score,
                                    regression_threshold=regression_threshold,
                                    verbose=verbose,
                                )
                            )
                        else:
                            regression_results.append(
                                unavailable_regression(
                                    agent_name,
                                    holdout_result.error
                                    or "holdout evaluation was incomplete",
                                    holdout_result.holdout_score,
                                )
                            )
                except Exception as e:
                    error = f"could not read holdout evidence: {e}"
                    holdout_results.append(
                        unavailable_holdout(
                            agent_name, VerificationStatus.ERROR, error
                        )
                    )
                    if check_regression:
                        regression_results.append(
                            unavailable_regression(agent_name, error)
                        )
                    if verbose:
                        print(f"  ERROR in holdout evaluation: {error}")
            else:
                error = f"holdout data not found at {holdout_path}"
                holdout_results.append(
                    unavailable_holdout(
                        agent_name, VerificationStatus.NOT_RUN, error
                    )
                )
                if check_regression:
                    regression_results.append(
                        unavailable_regression(agent_name, error)
                    )
                if verbose:
                    print(f"  NOT_RUN: {error}")

            # Cross-validation (optional)
            if run_cross_validation:
                training_path = training_data_dir / f"{data_basename}.jsonl"
                if not training_path.exists():
                    training_path = training_data_dir / f"{agent_name}.jsonl"
                if training_path.exists():
                    try:
                        training_data = self.load_holdout_data(training_path)
                        cv_result = self.run_cross_validation(
                            agent_name=agent_name,
                            training_data=training_data,
                            metric_fn=metric_fn,
                            k=k_folds,
                            verbose=verbose,
                        )
                        cv_results.append(cv_result)
                    except Exception as e:
                        error = f"cross-validation error: {e}"
                        cv_results.append(
                            unavailable_cv(
                                agent_name, VerificationStatus.ERROR, error
                            )
                        )
                        if verbose:
                            print(f"  ERROR in cross-validation: {error}")
                else:
                    error = f"training data not found at {training_path}"
                    cv_results.append(
                        unavailable_cv(
                            agent_name, VerificationStatus.NOT_RUN, error
                        )
                    )
                    if verbose:
                        print(f"  NOT_RUN: {error}")

        # Generate summary
        completed_results = [
            result
            for result in holdout_results
            if result.status in (VerificationStatus.PASS, VerificationStatus.FAIL)
        ]
        passed_count = sum(
            result.status == VerificationStatus.PASS for result in holdout_results
        )
        failed_count = sum(
            result.status == VerificationStatus.FAIL for result in holdout_results
        )
        not_run_count = sum(
            result.status == VerificationStatus.NOT_RUN for result in holdout_results
        )
        error_count = sum(
            result.status == VerificationStatus.ERROR for result in holdout_results
        )
        regressed_count = sum(1 for r in regression_results if r.regressed)
        regression_incomplete = sum(
            result.status
            in (
                VerificationStatus.NOT_ASSESSABLE,
                VerificationStatus.NOT_RUN,
                VerificationStatus.ERROR,
            )
            for result in regression_results
        )
        cv_incomplete = sum(
            result.status in (VerificationStatus.NOT_RUN, VerificationStatus.ERROR)
            for result in cv_results
        )

        summary = {
            "total_agents": len(agent_names),
            "holdout_passed": passed_count,
            "holdout_failed": failed_count,
            "holdout_not_run": not_run_count,
            "holdout_errors": error_count,
            "incomplete": not_run_count + error_count,
            "regressions": regressed_count,
            "regression_not_assessable": regression_incomplete,
            "cross_validation_incomplete": cv_incomplete,
            "average_holdout_score": (
                sum(r.holdout_score for r in holdout_results)
                / len(holdout_results)
                if holdout_results else 0.0
            ),
            "average_cv_score": (
                sum(r.mean_score for r in cv_results) / len(cv_results)
                if cv_results else 0.0
            ),
        }

        if verbose:
            print("\n" + "=" * 60)
            print("VERIFICATION SUMMARY")
            print("=" * 60)
            print(f"Agents verified: {len(completed_results)}/{len(agent_names)}")
            print(f"Holdout passed: {passed_count}/{len(agent_names)}")
            print(f"Holdout incomplete: {not_run_count + error_count}")
            print(f"Regressions detected: {regressed_count}")
            print(f"Regressions not assessable: {regression_incomplete}")
            print(f"Average holdout score: {summary['average_holdout_score']:.3f}")
            if cv_results:
                print(f"Average CV score: {summary['average_cv_score']:.3f}")

        return VerificationReport(
            timestamp=datetime.now().isoformat(),
            agents_verified=len(completed_results),
            holdout_results=holdout_results,
            regression_results=regression_results,
            cross_validation_results=cv_results,
            summary=summary,
        )


def pre_flight_holdout_check(
    agent_name: str,
    new_optimized: OptimizedPrompt,
    holdout_data: List[TrainingExample],
    metric_fn: Callable[[str, str], float],
    runner: ClaudeRunner,
    storage: DemoStorage,
    min_improvement: float = 0.0,
    verbose: bool = True,
) -> PreFlightResult:
    """
    Check if a new optimization beats the existing one on holdout data
    before saving to _latest.json.

    Prevents the regression pattern where save_optimized_prompt overwrites
    better prompts with worse ones (cf. code-reviewer 0.525 -> 0.314 incident).

    Args:
        agent_name: Name of the agent/skill
        new_optimized: The newly optimized prompt to evaluate
        holdout_data: Holdout examples for evaluation
        metric_fn: Metric function for scoring
        runner: ClaudeRunner for running evaluation
        storage: DemoStorage to load existing _latest.json
        min_improvement: New must beat existing by at least this much (default: tie OK)
        verbose: Print progress

    Returns:
        Typed result. It remains unpackable as
        ``(should_deploy, new_score, existing_score)`` for compatibility.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"PRE-FLIGHT HOLDOUT CHECK: {agent_name}")
        print(f"{'='*50}")

    if not holdout_data:
        return PreFlightResult(
            status=VerificationStatus.NOT_RUN,
            should_deploy=False,
            new_score=0.0,
            existing_score=None,
            error="holdout data is empty",
        )
    if not callable(metric_fn):
        return PreFlightResult(
            status=VerificationStatus.NOT_RUN,
            should_deploy=False,
            new_score=0.0,
            existing_score=None,
            error="metric function is unavailable",
        )
    if (
        isinstance(min_improvement, bool)
        or not isinstance(min_improvement, (int, float))
        or not math.isfinite(float(min_improvement))
    ):
        return PreFlightResult(
            status=VerificationStatus.ERROR,
            should_deploy=False,
            new_score=0.0,
            existing_score=None,
            error="minimum improvement must be finite",
        )

    def render_error(message: str) -> List[EvaluationResult]:
        return [
            EvaluationResult(i, VerificationStatus.ERROR, 0.0, message)
            for i in range(len(holdout_data))
        ]

    def evaluate(prompt: str, label: str) -> Tuple[float, List[EvaluationResult]]:
        evaluations: List[EvaluationResult] = []
        for i, example in enumerate(holdout_data):
            full_prompt = f"{prompt}\n\n## New Input\n\n{example.input_text}"
            try:
                run_result = runner.run(full_prompt)
                if run_result is None or not run_result.success:
                    error = getattr(run_result, "error", "missing runner result")
                    raise RuntimeError(error or "model call failed")
                score = metric_fn(example.expected_output, run_result.output)
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError("metric returned a non-numeric score")
                score = float(score)
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("metric score must be finite and in [0, 1]")
                evaluations.append(
                    EvaluationResult(i, VerificationStatus.PASS, score)
                )
                if verbose:
                    print(f"  [{label} {i+1}/{len(holdout_data)}] Score: {score:.3f}")
            except Exception as exc:
                evaluations.append(
                    EvaluationResult(i, VerificationStatus.ERROR, 0.0, str(exc))
                )
                if verbose:
                    print(
                        f"  [{label} {i+1}/{len(holdout_data)}] "
                        f"ERROR (scored 0.0): {exc}"
                    )
        return (
            sum(item.score for item in evaluations) / len(holdout_data),
            evaluations,
        )

    try:
        new_prompt = new_optimized.to_prompt()
    except Exception as exc:
        error = f"could not render candidate prompt: {exc}"
        return PreFlightResult(
            status=VerificationStatus.ERROR,
            should_deploy=False,
            new_score=0.0,
            existing_score=None,
            new_evaluations=render_error(error),
            error=error,
        )
    new_score, new_evaluations = evaluate(new_prompt, "new")

    try:
        existing = storage.load_optimized_prompt(agent_name)
    except Exception as exc:
        error = f"could not load incumbent prompt: {exc}"
        return PreFlightResult(
            status=VerificationStatus.ERROR,
            should_deploy=False,
            new_score=new_score,
            existing_score=None,
            new_evaluations=new_evaluations,
            existing_evaluations=render_error(error),
            error=error,
        )

    existing_score: Optional[float] = None
    existing_evaluations: List[EvaluationResult] = []
    if existing and existing.demos:
        try:
            existing_prompt = existing.to_prompt()
        except Exception as exc:
            error = f"could not render incumbent prompt: {exc}"
            return PreFlightResult(
                status=VerificationStatus.ERROR,
                should_deploy=False,
                new_score=new_score,
                existing_score=None,
                new_evaluations=new_evaluations,
                existing_evaluations=render_error(error),
                error=error,
            )
        existing_score, existing_evaluations = evaluate(existing_prompt, "existing")

    incomplete = [
        item
        for item in [*new_evaluations, *existing_evaluations]
        if item.status != VerificationStatus.PASS
    ]
    if incomplete:
        error = f"{len(incomplete)} holdout evaluations failed"
        if verbose:
            print(f"\n  Decision: ERROR ({error})")
        return PreFlightResult(
            status=VerificationStatus.ERROR,
            should_deploy=False,
            new_score=new_score,
            existing_score=existing_score,
            new_evaluations=new_evaluations,
            existing_evaluations=existing_evaluations,
            error=error,
        )

    if existing_score is None:
        should_deploy = True
        status = VerificationStatus.PASS
        if verbose:
            print(f"\n  No existing optimization. New score: {new_score:.3f}")
            print("  Decision: DEPLOY (first optimization)")
    else:
        improvement = new_score - existing_score
        should_deploy = improvement >= min_improvement
        status = VerificationStatus.PASS if should_deploy else VerificationStatus.FAIL
        if verbose:
            print(f"\n  Existing score: {existing_score:.3f}")
            print(f"  New score:      {new_score:.3f}")
            print(f"  Improvement:    {improvement:+.3f}")
            print(f"  Min required:   {min_improvement:+.3f}")
            print(
                "  Decision: DEPLOY (improvement >= threshold)"
                if should_deploy
                else "  Decision: SKIP (improvement below threshold, keeping existing)"
            )

    return PreFlightResult(
        status=status,
        should_deploy=should_deploy,
        new_score=new_score,
        existing_score=existing_score,
        new_evaluations=new_evaluations,
        existing_evaluations=existing_evaluations,
    )


def generate_verification_report(
    report: VerificationReport,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a markdown report from verification results.

    Args:
        report: VerificationReport to format
        output_path: Optional path to save report

    Returns:
        Markdown report content
    """
    lines = [
        "# Verification Report",
        "",
        f"**Generated**: {report.timestamp}",
        f"**Agents Verified**: {report.agents_verified}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Holdout Passed | {report.summary.get('holdout_passed', 0)} |",
        f"| Holdout Failed | {report.summary.get('holdout_failed', 0)} |",
        f"| Holdout Not Run | {report.summary.get('holdout_not_run', 0)} |",
        f"| Holdout Errors | {report.summary.get('holdout_errors', 0)} |",
        f"| Regressions | {report.summary.get('regressions', 0)} |",
        f"| Regression Not Assessable | {report.summary.get('regression_not_assessable', 0)} |",
        f"| Avg Holdout Score | {report.summary.get('average_holdout_score', 0):.3f} |",
    ]

    if report.summary.get('average_cv_score', 0) > 0:
        lines.append(f"| Avg CV Score | {report.summary['average_cv_score']:.3f} |")

    lines.extend(["", "## Holdout Results", ""])

    if report.holdout_results:
        lines.extend([
            "| Agent | Score | Size | Status |",
            "|-------|-------|------|--------|",
        ])
        for r in report.holdout_results:
            lines.append(
                f"| {r.agent_name} | {r.holdout_score:.3f} | "
                f"{r.holdout_size} | {r.status.value} |"
            )
    else:
        lines.append("No holdout results.")

    lines.extend(["", "## Regression Check", ""])

    if report.regression_results:
        lines.extend([
            "| Agent | Baseline | Current | Change | Status |",
            "|-------|----------|---------|--------|--------|",
        ])
        for r in report.regression_results:
            baseline = f"{r.baseline_score:.3f}" if r.baseline_score is not None else "N/A"
            current = f"{r.current_score:.3f}" if r.current_score is not None else "N/A"
            improvement = f"{r.improvement:+.3f}" if r.improvement is not None else "N/A"
            status = "REGRESSED" if r.regressed else r.status.value
            lines.append(
                f"| {r.agent_name} | {baseline} | {current} | "
                f"{improvement} | {status} |"
            )
    else:
        lines.append("No regression checks performed.")

    if report.cross_validation_results:
        lines.extend(["", "## Cross-Validation Results", ""])
        lines.extend([
            "| Agent | Mean | Std | K | Status |",
            "|-------|------|-----|---|--------|",
        ])
        for r in report.cross_validation_results:
            lines.append(
                f"| {r.agent_name} | {r.mean_score:.3f} | "
                f"{r.std_score:.3f} | {r.k} | {r.status.value} |"
            )

    content = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)

    return content
