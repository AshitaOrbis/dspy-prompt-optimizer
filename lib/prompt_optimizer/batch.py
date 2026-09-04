"""
Batch optimization utilities for processing multiple agents/skills.

Provides parallel and sequential optimization capabilities for
efficiently running BootstrapFewShot across multiple targets.
"""

import hashlib
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .bootstrap import BootstrapFewShot, BootstrapResult, TrainingExample
from .claude_runner import ClaudeRunner
from .storage import DemoStorage
from .utils import parse_markdown_prompt

logger = logging.getLogger(__name__)


@dataclass
class BatchTarget:
    """A single optimization target (agent or skill)."""
    name: str
    prompt_path: Path
    training_data_path: Path
    metric_fn: Callable[[str, str], float]
    threshold: float = 0.7
    max_demos: int = 3
    min_demos: int = 1
    allow_zero_demos: bool = False
    # Phase 6: Transformer support
    demo_transformer: Optional[Callable] = None
    format_instruction: str = ""

    def __post_init__(self) -> None:
        if self.max_demos < 0:
            raise ValueError("max_demos must be non-negative")
        if self.min_demos < 0:
            raise ValueError("min_demos must be non-negative")
        if not self.allow_zero_demos and self.min_demos == 0:
            raise ValueError("min_demos=0 requires allow_zero_demos=True")
        if not self.allow_zero_demos and self.min_demos > self.max_demos:
            raise ValueError("min_demos cannot exceed max_demos")


@dataclass(frozen=True)
class ArtifactIdentity:
    """Identity of the exact candidate written by this batch run."""

    run_id: str
    created_at: str
    content_hash: str
    file_hash: str
    path: Path


@dataclass
class BatchResult:
    """Result from a batch optimization run."""
    target: BatchTarget
    result: Optional[BootstrapResult]
    error: Optional[str]
    duration_seconds: float
    artifact: Optional[ArtifactIdentity] = None

    @property
    def succeeded(self) -> bool:
        """A batch target succeeds only with its verified, run-bound artifact."""
        return self.error is None and self.result is not None and self.artifact is not None


@dataclass
class BatchSummary:
    """Summary of a complete batch optimization run."""
    total_targets: int
    successful: int
    failed: int
    results: List[BatchResult]
    start_time: str
    end_time: str
    total_duration_seconds: float


def load_prompt(path: Path) -> str:
    """Load prompt content from a file, handling YAML frontmatter."""
    with open(path) as f:
        content = f.read()
    return parse_markdown_prompt(content)


def load_training_data(path: Path) -> List[TrainingExample]:
    """Load training data from JSONL file."""
    import json

    examples = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "input" not in data:
                    raise ValueError("Missing 'input' field")
                if "expected" not in data:
                    raise ValueError("Missing 'expected' field")
                examples.append(TrainingExample(
                    input_text=data["input"],
                    expected_output=data["expected"],
                    metadata=data.get("metadata"),
                ))
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                raise ValueError(f"Error on line {line_num}: {e}") from e
    return examples


_ARTIFACT_METADATA_KEYS = {
    "artifact_run_id",
    "artifact_created_at",
    "artifact_content_hash",
}


def _candidate_content_hash(candidate) -> str:
    """Hash candidate semantics without the self-referential identity fields."""
    payload = asdict(candidate)
    metadata = dict(payload.get("metadata") or {})
    for key in _ARTIFACT_METADATA_KEYS:
        metadata.pop(key, None)
    payload["metadata"] = metadata or None
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _persist_and_verify_artifact(
    target: BatchTarget,
    result: BootstrapResult,
    storage: DemoStorage,
    run_id: str,
    created_at: str,
) -> ArtifactIdentity:
    """Write and reload the exact candidate, rejecting stale or absent artifacts."""
    candidate = result.optimized_prompt
    content_hash = _candidate_content_hash(candidate)
    metadata = dict(candidate.metadata or {})
    metadata.update(
        {
            "artifact_run_id": run_id,
            "artifact_created_at": created_at,
            "artifact_content_hash": content_hash,
        }
    )
    candidate.metadata = metadata

    storage.save_demos(target.name, candidate.demos)
    paths = storage.save_optimized_prompt(target.name, candidate)
    latest_path = paths.get("latest") if isinstance(paths, dict) else None
    if latest_path is None:
        raise RuntimeError("storage did not return a latest artifact path")
    latest_path = Path(latest_path)
    if not latest_path.is_file():
        raise RuntimeError(f"new artifact was not written: {latest_path}")

    persisted = storage.load_optimized_prompt(target.name)
    if persisted is None:
        raise RuntimeError("new artifact could not be reloaded")
    persisted_metadata = persisted.metadata or {}
    expected_metadata = {
        "artifact_run_id": run_id,
        "artifact_created_at": created_at,
        "artifact_content_hash": content_hash,
    }
    for key, expected in expected_metadata.items():
        if persisted_metadata.get(key) != expected:
            raise RuntimeError(
                f"artifact identity mismatch for {key}; stale artifact refused"
            )
    if _candidate_content_hash(persisted) != content_hash:
        raise RuntimeError("artifact content hash mismatch")

    file_hash = hashlib.sha256(latest_path.read_bytes()).hexdigest()
    return ArtifactIdentity(
        run_id=run_id,
        created_at=created_at,
        content_hash=content_hash,
        file_hash=file_hash,
        path=latest_path,
    )


def optimize_single_target(
    target: BatchTarget,
    runner: ClaudeRunner,
    storage: DemoStorage,
    verbose: bool = False,
) -> BatchResult:
    """
    Optimize a single target (agent or skill).

    Args:
        target: The optimization target
        runner: ClaudeRunner instance
        storage: DemoStorage instance
        verbose: Print progress

    Returns:
        BatchResult with success/failure info
    """
    import time

    start_time = time.time()
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        # Load prompt and training data
        base_prompt = load_prompt(target.prompt_path)
        training_data = load_training_data(target.training_data_path)

        if verbose:
            print(f"[{target.name}] Starting optimization ({len(training_data)} examples)")
            if target.format_instruction:
                print(f"[{target.name}] Format instruction: enabled")
            if target.demo_transformer:
                print(f"[{target.name}] Demo transformer: enabled")

        # Create optimizer with optional transformer support
        optimizer = BootstrapFewShot(
            runner=runner,
            max_demos=target.max_demos,
            storage=storage,
            demo_transformer=target.demo_transformer,
            format_instruction=target.format_instruction,
        )

        # Run optimization
        result = optimizer.optimize(
            base_prompt=base_prompt,
            training_data=training_data,
            metric_fn=target.metric_fn,
            threshold=target.threshold,
            agent_name=target.name,
            verbose=verbose,
            persist=False,
        )

        successful_evaluations = result.total_examples - result.failed_examples
        if successful_evaluations <= 0:
            raise RuntimeError(
                "optimization produced no successful model evaluations"
            )

        demo_count = len(result.optimized_prompt.demos)
        required_demos = 0 if target.allow_zero_demos else target.min_demos
        if demo_count < required_demos:
            raise RuntimeError(
                f"optimization selected {demo_count} demos; "
                f"at least {required_demos} required"
            )

        artifact = _persist_and_verify_artifact(
            target=target,
            result=result,
            storage=storage,
            run_id=run_id,
            created_at=created_at,
        )

        duration = time.time() - start_time

        if verbose:
            print(f"[{target.name}] Complete: {result.successful_examples}/{result.total_examples} demos, avg score {result.avg_score:.3f}")

        return BatchResult(
            target=target,
            result=result,
            error=None,
            duration_seconds=duration,
            artifact=artifact,
        )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)

        if verbose:
            print(f"[{target.name}] FAILED: {error_msg}")

        logger.exception(f"Failed to optimize {target.name}")

        return BatchResult(
            target=target,
            result=None,
            error=error_msg,
            duration_seconds=duration,
            artifact=None,
        )


def optimize_batch_sequential(
    targets: List[BatchTarget],
    runner: ClaudeRunner,
    storage: Optional[DemoStorage] = None,
    verbose: bool = True,
) -> BatchSummary:
    """
    Optimize multiple targets sequentially.

    Args:
        targets: List of optimization targets
        runner: ClaudeRunner instance
        storage: Optional DemoStorage (creates new if not provided)
        verbose: Print progress

    Returns:
        BatchSummary with all results
    """
    import time

    storage = storage or DemoStorage()
    start_time = datetime.now()
    start_ts = time.time()

    if verbose:
        print(f"Starting batch optimization of {len(targets)} targets...")
        print(f"Mode: Sequential")
        print("-" * 50)

    results: List[BatchResult] = []

    for i, target in enumerate(targets):
        if verbose:
            print(f"\n[{i+1}/{len(targets)}] {target.name}")

        result = optimize_single_target(
            target=target,
            runner=runner,
            storage=storage,
            verbose=verbose,
        )
        results.append(result)

    end_time = datetime.now()
    total_duration = time.time() - start_ts

    successful = sum(1 for r in results if r.succeeded)
    failed = len(results) - successful

    summary = BatchSummary(
        total_targets=len(targets),
        successful=successful,
        failed=failed,
        results=results,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        total_duration_seconds=total_duration,
    )

    if verbose:
        print("\n" + "=" * 50)
        print("BATCH OPTIMIZATION COMPLETE")
        print("=" * 50)
        print(f"Total targets: {summary.total_targets}")
        print(f"Successful: {summary.successful}")
        print(f"Failed: {summary.failed}")
        print(f"Duration: {summary.total_duration_seconds:.1f}s")

    return summary


def optimize_batch_parallel(
    targets: List[BatchTarget],
    runner: ClaudeRunner,
    storage: Optional[DemoStorage] = None,
    max_workers: int = 3,
    verbose: bool = True,
) -> BatchSummary:
    """
    Optimize multiple targets in parallel.

    NOTE: Use with caution - parallel API calls may hit rate limits.

    Args:
        targets: List of optimization targets
        runner: ClaudeRunner instance
        storage: Optional DemoStorage (creates new if not provided)
        max_workers: Maximum parallel workers (default: 3)
        verbose: Print progress

    Returns:
        BatchSummary with all results
    """
    import time

    storage = storage or DemoStorage()
    start_time = datetime.now()
    start_ts = time.time()

    if verbose:
        print(f"Starting batch optimization of {len(targets)} targets...")
        print(f"Mode: Parallel (max_workers={max_workers})")
        print("-" * 50)

    results: List[BatchResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_target = {
            executor.submit(
                optimize_single_target,
                target,
                runner,
                storage,
                verbose,
            ): target
            for target in targets
        }

        # Collect results as they complete
        for future in as_completed(future_to_target):
            target = future_to_target[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Handle unexpected exceptions
                results.append(BatchResult(
                    target=target,
                    result=None,
                    error=str(e),
                    duration_seconds=0.0,
                ))

    end_time = datetime.now()
    total_duration = time.time() - start_ts

    successful = sum(1 for r in results if r.succeeded)
    failed = len(results) - successful

    summary = BatchSummary(
        total_targets=len(targets),
        successful=successful,
        failed=failed,
        results=results,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        total_duration_seconds=total_duration,
    )

    if verbose:
        print("\n" + "=" * 50)
        print("BATCH OPTIMIZATION COMPLETE")
        print("=" * 50)
        print(f"Total targets: {summary.total_targets}")
        print(f"Successful: {summary.successful}")
        print(f"Failed: {summary.failed}")
        print(f"Duration: {summary.total_duration_seconds:.1f}s")

    return summary


def generate_batch_report(
    summary: BatchSummary,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a markdown report from batch optimization results.

    Args:
        summary: BatchSummary from optimization run
        output_path: Optional path to save report

    Returns:
        Markdown report content
    """
    lines = [
        "# Batch Optimization Report",
        "",
        f"**Generated**: {summary.end_time}",
        f"**Duration**: {summary.total_duration_seconds:.1f} seconds",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Targets | {summary.total_targets} |",
        f"| Successful | {summary.successful} |",
        f"| Failed | {summary.failed} |",
        f"| Success Rate | {(100 * summary.successful / summary.total_targets) if summary.total_targets > 0 else 0.0:.1f}% |",
        "",
        "## Results by Target",
        "",
    ]

    # Successful targets
    successful_results = [r for r in summary.results if r.succeeded]
    if successful_results:
        lines.extend([
            "### Successful",
            "",
            "| Target | Demos | Avg Score | Duration |",
            "|--------|-------|-----------|----------|",
        ])

        for r in successful_results:
            demos = len(r.result.optimized_prompt.demos) if r.result else 0
            score = r.result.avg_score if r.result else 0.0
            lines.append(
                f"| {r.target.name} | {demos} | {score:.3f} | {r.duration_seconds:.1f}s |"
            )
        lines.append("")

    # Failed targets
    failed_results = [r for r in summary.results if not r.succeeded]
    if failed_results:
        lines.extend([
            "### Failed",
            "",
            "| Target | Error |",
            "|--------|-------|",
        ])

        for r in failed_results:
            error = r.error or "missing verified artifact"
            error_short = error[:50] + "..." if len(error) > 50 else error
            lines.append(f"| {r.target.name} | {error_short} |")
        lines.append("")

    # Detailed results for successful targets
    if successful_results:
        lines.extend([
            "## Detailed Results",
            "",
        ])

        for r in successful_results:
            if r.result:
                lines.extend([
                    f"### {r.target.name}",
                    "",
                    f"- **Total examples**: {r.result.total_examples}",
                    f"- **Successful demos**: {r.result.successful_examples}",
                    f"- **Failed runs**: {r.result.failed_examples}",
                    f"- **Average score**: {r.result.avg_score:.3f}",
                    f"- **Demos selected**: {len(r.result.optimized_prompt.demos)}",
                    "",
                ])

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report
