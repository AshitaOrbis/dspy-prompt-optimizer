"""
Verification framework for validating prompt optimizations.

Provides holdout evaluation, cross-validation, and regression testing
to ensure optimized prompts perform reliably.
"""

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

from .claude_runner import ClaudeRunner
from .bootstrap import TrainingExample, BootstrapFewShot
from .storage import DemoStorage, OptimizedPrompt, Demo
from .utils import parse_markdown_prompt, find_agent_path

logger = logging.getLogger(__name__)


@dataclass
class HoldoutResult:
    """Result from holdout evaluation."""
    agent_name: str
    holdout_score: float
    holdout_size: int
    successful_evals: int
    failed_evals: int
    scores: List[float]
    passed: bool


@dataclass
class CrossValidationResult:
    """Result from k-fold cross-validation."""
    agent_name: str
    mean_score: float
    std_score: float
    fold_scores: List[float]
    k: int
    total_examples: int


@dataclass
class RegressionResult:
    """Result from regression check."""
    agent_name: str
    current_score: float
    baseline_score: float
    improvement: float
    regressed: bool
    threshold: float


@dataclass
class VerificationReport:
    """Complete verification report."""
    timestamp: str
    agents_verified: int
    holdout_results: List[HoldoutResult]
    regression_results: List[RegressionResult]
    cross_validation_results: List[CrossValidationResult]
    summary: Dict[str, Any]


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

    def _load_baseline_scores(self) -> Dict[str, float]:
        """Load baseline scores from file."""
        if self._baseline_scores is not None:
            return self._baseline_scores

        if self.baseline_scores_path and self.baseline_scores_path.exists():
            try:
                with open(self.baseline_scores_path) as f:
                    data = json.load(f)
                    # Validate structure
                    if not isinstance(data, dict):
                        raise ValueError("Baseline scores must be a dict")
                    # Keep only numeric entries. Non-numeric values (e.g. a
                    # documentation "<target>_note" string) are metadata, not
                    # scores — skip them with a warning instead of rejecting the
                    # whole file. Rejecting made _baseline_scores = {}, which
                    # silently gave every regression check a 0.000 baseline
                    # (fail-open: any score "passed"). Bug found 2026-07-06 by
                    # the code-reviewer re-baseline run.
                    scores = {}
                    for k, val in data.items():
                        if isinstance(val, bool) or not isinstance(val, (int, float)):
                            print(f"Note: skipping non-numeric baseline entry '{k}'")
                            continue
                        scores[k] = val
                    self._baseline_scores = scores
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not load baseline scores: {e}")
                self._baseline_scores = {}
        else:
            self._baseline_scores = {}

        return self._baseline_scores

    def _save_baseline_score(self, agent_name: str, score: float):
        """Save a baseline score."""
        scores = self._load_baseline_scores()
        scores[agent_name] = score

        if self.baseline_scores_path:
            self.baseline_scores_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_scores_path, 'w') as f:
                json.dump(scores, f, indent=2)

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

        # Load optimized prompt
        optimized = self.storage.load_optimized_prompt(agent_name)
        if not optimized:
            raise ValueError(f"No optimized prompt found for {agent_name}")

        if not callable(metric_fn):
            raise TypeError(f"metric_fn must be callable, got {type(metric_fn).__name__}")

        prompt = optimized.to_prompt()
        scores = []
        failed_count = 0

        for i, example in enumerate(holdout_data):
            full_prompt = f"{prompt}\n\n## New Input\n\n{example.input_text}"
            result = self.runner.run(full_prompt)

            if result.success:
                score = metric_fn(example.expected_output, result.output)
                scores.append(score)
                if verbose:
                    print(f"  [{i+1}/{len(holdout_data)}] Score: {score:.3f}")
            else:
                failed_count += 1
                if verbose:
                    print(f"  [{i+1}/{len(holdout_data)}] FAILED: {result.error}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        passed = avg_score >= pass_threshold

        if verbose:
            print(f"\nHoldout Results:")
            print(f"  Average score: {avg_score:.3f}")
            print(f"  Evaluated: {len(scores)}/{len(holdout_data)}")
            print(f"  Status: {'PASSED' if passed else 'FAILED'}")

        return HoldoutResult(
            agent_name=agent_name,
            holdout_score=avg_score,
            holdout_size=len(holdout_data),
            successful_evals=len(scores),
            failed_evals=failed_count,
            scores=scores,
            passed=passed,
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
        if k < 2:
            raise ValueError("k must be >= 2 for cross-validation")
        if k > len(training_data):
            raise ValueError(f"k ({k}) cannot exceed training data size ({len(training_data)})")
        if not callable(metric_fn):
            raise TypeError(f"metric_fn must be callable, got {type(metric_fn).__name__}")

        if verbose:
            print(f"\nRunning {k}-fold cross-validation for {agent_name}...")

        # Load base prompt
        optimized = self.storage.load_optimized_prompt(agent_name)
        if optimized:
            base_prompt = optimized.base_prompt
        else:
            # Try to load from agent file using shared path resolution
            try:
                agent_path = find_agent_path(agent_name)
                with open(agent_path) as f:
                    content = f.read()
                base_prompt = parse_markdown_prompt(content)
            except FileNotFoundError:
                raise ValueError(f"Cannot find base prompt for {agent_name}")

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

            result = optimizer.optimize(
                base_prompt=base_prompt,
                training_data=train_data,
                metric_fn=metric_fn,
                threshold=threshold,
                agent_name=f"{agent_name}_cv_fold{fold_idx}",
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

        return CrossValidationResult(
            agent_name=agent_name,
            mean_score=mean_score,
            std_score=std_score,
            fold_scores=fold_scores,
            k=k,
            total_examples=len(training_data),
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
        baseline_scores = self._load_baseline_scores()
        baseline_score = baseline_scores.get(agent_name, 0.0)

        improvement = new_score - baseline_score
        regressed = improvement < -regression_threshold

        if verbose:
            print(f"\nRegression Check for {agent_name}:")
            print(f"  Baseline score: {baseline_score:.3f}")
            print(f"  New score: {new_score:.3f}")
            print(f"  Change: {improvement:+.3f}")
            print(f"  Threshold: -{regression_threshold:.3f}")
            print(f"  Status: {'REGRESSED' if regressed else 'OK'}")

        return RegressionResult(
            agent_name=agent_name,
            current_score=new_score,
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

        holdout_results = []
        regression_results = []
        cv_results = []

        for agent_name in agent_names:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Verifying: {agent_name}")
                print(f"{'='*60}")

            metric_fn = metric_fns.get(agent_name)
            if not metric_fn:
                if verbose:
                    print(f"  WARNING: No metric function for {agent_name}, skipping")
                continue

            # Holdout evaluation — resolve via data_basenames mapping or direct name
            data_basename = data_basenames.get(agent_name, agent_name) if data_basenames else agent_name
            holdout_path = holdout_data_dir / f"{data_basename}-holdout.jsonl"
            if not holdout_path.exists():
                # Fallback: try agent name directly
                holdout_path = holdout_data_dir / f"{agent_name}-holdout.jsonl"
            if holdout_path.exists():
                holdout_data = self.load_holdout_data(holdout_path)
                try:
                    holdout_result = self.run_holdout_evaluation(
                        agent_name=agent_name,
                        holdout_data=holdout_data,
                        metric_fn=metric_fn,
                        pass_threshold=pass_threshold,
                        verbose=verbose,
                    )
                    holdout_results.append(holdout_result)

                    # Regression check
                    regression_result = self.check_regression(
                        agent_name=agent_name,
                        new_score=holdout_result.holdout_score,
                        regression_threshold=regression_threshold,
                        verbose=verbose,
                    )
                    regression_results.append(regression_result)
                except Exception as e:
                    if verbose:
                        print(f"  ERROR in holdout evaluation: {e}")
            else:
                if verbose:
                    print(f"  No holdout data found at {holdout_path}")

            # Cross-validation (optional)
            if run_cross_validation:
                training_path = training_data_dir / f"{agent_name}.jsonl"
                if training_path.exists():
                    training_data = self.load_holdout_data(training_path)
                    try:
                        cv_result = self.run_cross_validation(
                            agent_name=agent_name,
                            training_data=training_data,
                            metric_fn=metric_fn,
                            k=k_folds,
                            verbose=verbose,
                        )
                        cv_results.append(cv_result)
                    except Exception as e:
                        if verbose:
                            print(f"  ERROR in cross-validation: {e}")
                else:
                    if verbose:
                        print(f"  No training data found at {training_path}")

        # Generate summary
        passed_count = sum(1 for r in holdout_results if r.passed)
        regressed_count = sum(1 for r in regression_results if r.regressed)

        summary = {
            "total_agents": len(agent_names),
            "holdout_passed": passed_count,
            "holdout_failed": len(holdout_results) - passed_count,
            "regressions": regressed_count,
            "average_holdout_score": (
                sum(r.holdout_score for r in holdout_results) / len(holdout_results)
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
            print(f"Agents verified: {len(agent_names)}")
            print(f"Holdout passed: {passed_count}/{len(holdout_results)}")
            print(f"Regressions detected: {regressed_count}")
            print(f"Average holdout score: {summary['average_holdout_score']:.3f}")
            if cv_results:
                print(f"Average CV score: {summary['average_cv_score']:.3f}")

        return VerificationReport(
            timestamp=datetime.now().isoformat(),
            agents_verified=len(agent_names),
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
) -> Tuple[bool, float, Optional[float]]:
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
        Tuple of (should_deploy, new_score, existing_score).
        existing_score is None if no prior optimization exists.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"PRE-FLIGHT HOLDOUT CHECK: {agent_name}")
        print(f"{'='*50}")

    # Evaluate the new optimization on holdout
    new_prompt = new_optimized.to_prompt()
    new_scores = []
    for i, example in enumerate(holdout_data):
        full_prompt = f"{new_prompt}\n\n## New Input\n\n{example.input_text}"
        result = runner.run(full_prompt)
        if result.success:
            score = metric_fn(example.expected_output, result.output)
            new_scores.append(score)
            if verbose:
                print(f"  [new {i+1}/{len(holdout_data)}] Score: {score:.3f}")
        else:
            # A failed evaluation scores 0 rather than vanishing from the mean —
            # dropping it silently inflates the average of whatever remains.
            new_scores.append(0.0)
            if verbose:
                print(f"  [new {i+1}/{len(holdout_data)}] FAILED (scored 0.0): {result.error}")

    new_score = sum(new_scores) / len(new_scores) if new_scores else 0.0

    # Load and evaluate existing optimization (if any)
    existing = storage.load_optimized_prompt(agent_name)
    existing_score = None

    if existing and existing.demos:
        existing_prompt = existing.to_prompt()
        existing_scores = []
        for i, example in enumerate(holdout_data):
            full_prompt = f"{existing_prompt}\n\n## New Input\n\n{example.input_text}"
            result = runner.run(full_prompt)
            if result.success:
                score = metric_fn(example.expected_output, result.output)
                existing_scores.append(score)
                if verbose:
                    print(f"  [existing {i+1}/{len(holdout_data)}] Score: {score:.3f}")
            else:
                # Same rule as the new arm: failures score 0, symmetrically.
                existing_scores.append(0.0)
                if verbose:
                    print(f"  [existing {i+1}/{len(holdout_data)}] FAILED (scored 0.0): {result.error}")

        existing_score = sum(existing_scores) / len(existing_scores) if existing_scores else 0.0

    # Decision
    if existing_score is None:
        # No prior optimization — always deploy
        should_deploy = True
        if verbose:
            print(f"\n  No existing optimization. New score: {new_score:.3f}")
            print(f"  Decision: DEPLOY (first optimization)")
    else:
        improvement = new_score - existing_score
        should_deploy = improvement >= min_improvement

        if verbose:
            print(f"\n  Existing score: {existing_score:.3f}")
            print(f"  New score:      {new_score:.3f}")
            print(f"  Improvement:    {improvement:+.3f}")
            print(f"  Min required:   {min_improvement:+.3f}")
            if should_deploy:
                print(f"  Decision: DEPLOY (improvement >= threshold)")
            else:
                print(f"  Decision: SKIP (improvement below threshold, keeping existing)")

    return should_deploy, new_score, existing_score


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
        f"| Regressions | {report.summary.get('regressions', 0)} |",
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
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"| {r.agent_name} | {r.holdout_score:.3f} | {r.holdout_size} | {status} |")
    else:
        lines.append("No holdout results.")

    lines.extend(["", "## Regression Check", ""])

    if report.regression_results:
        lines.extend([
            "| Agent | Baseline | Current | Change | Status |",
            "|-------|----------|---------|--------|--------|",
        ])
        for r in report.regression_results:
            status = "REGRESSED" if r.regressed else "OK"
            lines.append(
                f"| {r.agent_name} | {r.baseline_score:.3f} | "
                f"{r.current_score:.3f} | {r.improvement:+.3f} | {status} |"
            )
    else:
        lines.append("No regression checks performed.")

    if report.cross_validation_results:
        lines.extend(["", "## Cross-Validation Results", ""])
        lines.extend([
            "| Agent | Mean | Std | K |",
            "|-------|------|-----|---|",
        ])
        for r in report.cross_validation_results:
            lines.append(f"| {r.agent_name} | {r.mean_score:.3f} | {r.std_score:.3f} | {r.k} |")

    content = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)

    return content
