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

import fcntl
import hashlib
import json
import logging
import math
import os
import random
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any

from .claude_runner import ClaudeRunner
from .storage import Demo, OptimizedPrompt, DemoStorage, SafeJSONEncoder
from .validation import contained_path, validate_name

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example for optimization."""
    input_text: str
    expected_output: str
    metadata: Optional[Dict[str, Any]] = None


class HoldoutGateError(RuntimeError):
    """Raised when required holdout evidence is absent, incomplete, or failing."""


@dataclass(frozen=True)
class GatePromotionResult:
    """Evidence returned after a candidate passes and is atomically promoted."""

    new_score: float
    existing_score: float
    evaluated_examples: int
    latest_path: Path
    corpus_sha256: str
    corpus_cardinality: int
    incumbent_artifact_sha256: str


@dataclass(frozen=True)
class HoldoutCorpusIdentity:
    """Stable identity of an ordered, parsed holdout corpus."""

    sha256: str
    cardinality: int

    def as_dict(self) -> Dict[str, Any]:
        return {"sha256": self.sha256, "cardinality": self.cardinality}


def holdout_corpus_identity(
    examples: List[TrainingExample],
) -> HoldoutCorpusIdentity:
    """Hash every ordered example plus its expected output and metadata."""
    if not examples:
        raise HoldoutGateError("Holdout data is required and must be non-empty")
    ordered_examples = [
        {
            "input": example.input_text,
            "expected": example.expected_output,
            "metadata": example.metadata,
        }
        for example in examples
    ]
    try:
        encoded = json.dumps(
            ordered_examples,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HoldoutGateError(
            f"Holdout corpus cannot be assigned a stable identity: {exc}"
        ) from exc
    return HoldoutCorpusIdentity(
        sha256=hashlib.sha256(encoded).hexdigest(),
        cardinality=len(examples),
    )


def _parse_corpus_identity(value: Any) -> HoldoutCorpusIdentity:
    if not isinstance(value, dict):
        raise HoldoutGateError(
            "Incumbent artifact has no recorded holdout corpus identity"
        )
    digest = value.get("sha256")
    cardinality = value.get("cardinality")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(cardinality, bool)
        or not isinstance(cardinality, int)
        or cardinality <= 0
    ):
        raise HoldoutGateError(
            "Incumbent artifact has no valid recorded holdout corpus identity"
        )
    return HoldoutCorpusIdentity(sha256=digest, cardinality=cardinality)


def _recorded_corpus_identity(incumbent: OptimizedPrompt) -> HoldoutCorpusIdentity:
    metadata = incumbent.metadata
    gate_evidence = metadata.get("holdout_gate") if isinstance(metadata, dict) else None
    value = (
        gate_evidence.get("corpus_identity")
        if isinstance(gate_evidence, dict)
        else None
    )
    return _parse_corpus_identity(value)


_ARTIFACT_IDENTITY_KEYS = {
    "artifact_run_id",
    "artifact_created_at",
    "artifact_content_hash",
}


def _prompt_semantic_hash(candidate: OptimizedPrompt) -> str:
    """Mirror the batch artifact hash while excluding self-identity fields."""
    payload = asdict(candidate)
    metadata = dict(payload.get("metadata") or {})
    for key in _ARTIFACT_IDENTITY_KEYS:
        metadata.pop(key, None)
    payload["metadata"] = metadata or None
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        cls=SafeJSONEncoder,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encode_prompt(candidate: OptimizedPrompt) -> bytes:
    return json.dumps(asdict(candidate), indent=2, cls=SafeJSONEncoder).encode("utf-8")


def _decode_prompt_snapshot(payload: bytes) -> OptimizedPrompt:
    """Decode the exact incumbent bytes captured for compare-and-swap."""
    try:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("prompt artifact must be an object")
        raw_demos = data.get("demos")
        if not isinstance(raw_demos, list):
            raise ValueError("prompt demos must be a list")
        demos = []
        for raw_demo in raw_demos:
            if not isinstance(raw_demo, dict):
                raise ValueError("prompt demo must be an object")
            demos.append(
                Demo(
                    input_text=raw_demo["input_text"],
                    output_text=raw_demo["output_text"],
                    score=raw_demo["score"],
                    metadata=raw_demo.get("metadata"),
                )
            )
        return OptimizedPrompt(
            base_prompt=data["base_prompt"],
            demos=demos,
            optimization_date=data["optimization_date"],
            metric_name=data["metric_name"],
            threshold=data["threshold"],
            avg_score=data["avg_score"],
            metadata=data.get("metadata"),
            format_instruction=data.get("format_instruction", ""),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HoldoutGateError(
            f"Incumbent/backup is unreadable or malformed: {exc}"
        ) from exc


def load_holdout_jsonl(path: Path) -> List[TrainingExample]:
    """Load a required, non-empty holdout JSONL file or fail closed."""
    path = Path(path)
    if not path.exists():
        raise HoldoutGateError(f"Holdout file is missing: {path}")
    if not path.is_file():
        raise HoldoutGateError(f"Holdout path is not a readable file: {path}")

    examples: List[TrainingExample] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HoldoutGateError(
                        f"Malformed holdout JSON at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise HoldoutGateError(
                        f"Malformed holdout entry at {path}:{line_number}: expected object"
                    )
                input_text = payload.get("input")
                expected_output = payload.get("expected")
                if not isinstance(input_text, str) or not isinstance(expected_output, str):
                    raise HoldoutGateError(
                        f"Malformed holdout entry at {path}:{line_number}: "
                        "'input' and 'expected' must be strings"
                    )
                examples.append(
                    TrainingExample(
                        input_text=input_text,
                        expected_output=expected_output,
                        metadata=payload.get("metadata"),
                    )
                )
    except HoldoutGateError:
        raise
    except OSError as exc:
        raise HoldoutGateError(f"Holdout file is unreadable: {path}: {exc}") from exc

    if not examples:
        raise HoldoutGateError(f"Holdout file is empty: {path}")
    return examples


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes via an fsynced same-directory temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def _promotion_lock(latest_path: Path):
    """Serialize cooperating writers while an exact latest snapshot is checked."""
    lock_path = latest_path.with_name(f"{latest_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def promote_candidate_atomic(
    agent_name: str,
    candidate: OptimizedPrompt,
    storage: DemoStorage,
    *,
    expected_latest_bytes: Optional[bytes] = None,
    expected_latest_absent: bool = False,
) -> Path:
    """Persist history and replace latest under a compare-and-swap lock."""
    validate_name(agent_name, kind="agent name")
    if not isinstance(candidate, OptimizedPrompt) or not candidate.demos:
        raise HoldoutGateError("Candidate prompt is missing or contains no demonstrations")

    prompts_dir = storage.prompts_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    history_json = contained_path(prompts_dir, f"{agent_name}_{timestamp}.json")
    history_markdown = contained_path(prompts_dir, f"{agent_name}_{timestamp}.md")
    latest_path = contained_path(prompts_dir, f"{agent_name}_latest.json")
    encoded = _encode_prompt(candidate)

    with _promotion_lock(latest_path):
        if expected_latest_bytes is not None and expected_latest_absent:
            raise ValueError(
                "expected_latest_bytes and expected_latest_absent are mutually exclusive"
            )
        if expected_latest_absent and latest_path.exists():
            raise HoldoutGateError(
                "Incumbent changed during holdout evaluation; stale promotion refused"
            )
        if expected_latest_bytes is not None:
            try:
                current_bytes = latest_path.read_bytes()
            except OSError as exc:
                raise HoldoutGateError(
                    f"Incumbent changed during holdout evaluation: {exc}"
                ) from exc
            if current_bytes != expected_latest_bytes:
                raise HoldoutGateError(
                    "Incumbent changed during holdout evaluation; stale promotion refused"
                )

        # A failure while writing demos/history leaves the incumbent latest untouched.
        storage.save_demos(agent_name, candidate.demos)
        _atomic_write(history_json, encoded)
        _atomic_write(history_markdown, candidate.to_markdown().encode("utf-8"))

        # Re-check after the ancillary writes, immediately before replacement.
        if (
            expected_latest_bytes is not None
            and latest_path.read_bytes() != expected_latest_bytes
        ):
            raise HoldoutGateError(
                "Incumbent changed during holdout evaluation; stale promotion refused"
            )
        _atomic_write(latest_path, encoded)
    return latest_path


def _rollback_promoted_latest(
    latest_path: Path,
    promoted_bytes: bytes,
    incumbent_bytes: bytes,
) -> None:
    """Restore only if latest still equals the artifact this gate promoted."""
    with _promotion_lock(latest_path):
        try:
            current_bytes = latest_path.read_bytes()
        except OSError as exc:
            logger.warning("Could not inspect latest for gate rollback: %s", exc)
            return
        if current_bytes != promoted_bytes:
            logger.warning(
                "Latest changed after gate promotion; refusing to overwrite concurrent update"
            )
            return
        _atomic_write(latest_path, incumbent_bytes)


class _CompleteCoverageRunner:
    """Track every gate call so failed/partial evaluation cannot look complete."""

    def __init__(self, runner: ClaudeRunner):
        self._runner = runner
        self.calls = 0
        self.failures: List[str] = []

    def run(self, prompt: str):
        self.calls += 1
        try:
            result = self._runner.run(prompt)
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise
        if result is None or not getattr(result, "success", False):
            error = getattr(result, "error", "missing runner result")
            self.failures.append(str(error))
        return result


def promote_candidate_with_holdout(
    agent_name: str,
    candidate: OptimizedPrompt,
    holdout_data: List[TrainingExample],
    metric_fn: Callable[[str, str], float],
    runner: ClaudeRunner,
    storage: DemoStorage,
    min_improvement: float = 0.0,
    verbose: bool = True,
    on_promoted: Optional[Callable[[float, float], None]] = None,
) -> GatePromotionResult:
    """Evaluate a staged candidate and atomically promote only on a complete pass.

    Gate policy is deliberately strict: runner failures abort the comparison rather
    than being averaged over a smaller denominator. Both candidate and incumbent
    must produce one finite score for every example in the same non-empty holdout.
    An incumbent is required because a promotion gate cannot make a symmetric
    comparison without its rollback artifact.
    """
    corpus_identity = holdout_corpus_identity(holdout_data)
    if not callable(metric_fn):
        raise HoldoutGateError("A callable holdout metric is required")
    if not isinstance(candidate, OptimizedPrompt) or not candidate.demos:
        raise HoldoutGateError("Candidate prompt is missing or contains no demonstrations")

    validate_name(agent_name, kind="agent name")
    latest_path = contained_path(storage.prompts_dir, f"{agent_name}_latest.json")
    backup_path = contained_path(storage.prompts_dir, f"{agent_name}_latest.json.pre-gate")
    if not latest_path.exists():
        raise HoldoutGateError(
            f"Incumbent/backup is missing for holdout-gated promotion: {latest_path}"
        )

    try:
        incumbent_bytes = latest_path.read_bytes()
    except OSError as exc:
        raise HoldoutGateError(f"Incumbent/backup is unreadable: {exc}") from exc
    incumbent = _decode_prompt_snapshot(incumbent_bytes)
    if incumbent is None or not incumbent.demos:
        raise HoldoutGateError("Incumbent/backup is missing a valid optimized prompt")
    recorded_identity = _recorded_corpus_identity(incumbent)
    if recorded_identity != corpus_identity:
        raise HoldoutGateError(
            "Holdout corpus identity mismatch: "
            f"incumbent={recorded_identity.sha256}/{recorded_identity.cardinality}, "
            f"current={corpus_identity.sha256}/{corpus_identity.cardinality}"
        )
    incumbent_artifact_sha256 = hashlib.sha256(incumbent_bytes).hexdigest()

    # Keep an audit/rollback artifact until the atomic promotion has succeeded.
    _atomic_write(backup_path, incumbent_bytes)
    try:
        if not backup_path.exists() or backup_path.read_bytes() != incumbent_bytes:
            raise HoldoutGateError("Incumbent backup could not be created or verified")
        json.loads(backup_path.read_text(encoding="utf-8"))
    except HoldoutGateError:
        raise
    except Exception as exc:
        raise HoldoutGateError(f"Incumbent backup is unreadable or malformed: {exc}") from exc

    tracking_runner = _CompleteCoverageRunner(runner)
    metric_failures: List[str] = []

    def checked_metric(expected: str, actual: str) -> float:
        score = metric_fn(expected, actual)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            metric_failures.append(f"non-numeric score: {score!r}")
            raise HoldoutGateError(f"Holdout metric returned a non-numeric score: {score!r}")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            metric_failures.append(f"non-finite score: {score!r}")
            raise HoldoutGateError(f"Holdout metric returned a non-finite score: {score!r}")
        return numeric_score

    promoted_bytes: Optional[bytes] = None
    try:
        # Local import avoids bootstrap <-> verification import initialization cycles.
        from .verification import pre_flight_holdout_check

        should_deploy, new_score, existing_score = pre_flight_holdout_check(
            agent_name=agent_name,
            new_optimized=candidate,
            holdout_data=holdout_data,
            metric_fn=checked_metric,
            runner=tracking_runner,
            storage=storage,
            min_improvement=min_improvement,
            verbose=verbose,
        )
        expected_calls = len(holdout_data) * 2
        if tracking_runner.failures:
            raise HoldoutGateError(
                "Holdout evaluation failed; complete coverage is required: "
                + "; ".join(tracking_runner.failures)
            )
        if metric_failures:
            raise HoldoutGateError(
                "Holdout metric failed; complete numeric coverage is required: "
                + "; ".join(metric_failures)
            )
        if tracking_runner.calls != expected_calls or existing_score is None:
            raise HoldoutGateError(
                f"Holdout coverage incomplete: expected {expected_calls} calls, "
                f"observed {tracking_runner.calls}"
            )
        if not should_deploy:
            raise HoldoutGateError(
                f"Candidate did not clear the holdout gate: new={new_score:.3f}, "
                f"incumbent={existing_score:.3f}, required={min_improvement:+.3f}"
            )

        identity_payload = corpus_identity.as_dict()
        candidate.metadata = {
            **(candidate.metadata or {}),
            "holdout_gate": {
                "schema_version": 1,
                "gate_type": "comparative",
                "corpus_identity": identity_payload,
                "candidate_evaluation": {
                    "corpus_identity": identity_payload,
                    "evaluated_examples": corpus_identity.cardinality,
                    "score": float(new_score),
                },
                "incumbent_evaluation": {
                    "corpus_identity": identity_payload,
                    "evaluated_examples": corpus_identity.cardinality,
                    "score": float(existing_score),
                    "artifact_sha256": incumbent_artifact_sha256,
                },
                "min_improvement": float(min_improvement),
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        if "artifact_content_hash" in candidate.metadata:
            candidate.metadata["artifact_content_hash"] = _prompt_semantic_hash(candidate)

        promoted_path = promote_candidate_atomic(
            agent_name,
            candidate,
            storage,
            expected_latest_bytes=incumbent_bytes,
        )
        promoted_bytes = promoted_path.read_bytes()
        if on_promoted is not None:
            on_promoted(float(new_score), float(existing_score))
        try:
            backup_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Promoted %s but could not remove gate backup: %s", agent_name, exc)
        return GatePromotionResult(
            new_score=float(new_score),
            existing_score=float(existing_score),
            evaluated_examples=len(holdout_data),
            latest_path=promoted_path,
            corpus_sha256=corpus_identity.sha256,
            corpus_cardinality=corpus_identity.cardinality,
            incumbent_artifact_sha256=incumbent_artifact_sha256,
        )
    except Exception as exc:
        if promoted_bytes is not None:
            _rollback_promoted_latest(latest_path, promoted_bytes, incumbent_bytes)
        if isinstance(exc, HoldoutGateError):
            raise
        raise HoldoutGateError(f"Holdout gate failed: {type(exc).__name__}: {exc}") from exc


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
        persist: bool = True,
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
            persist: Whether to persist the candidate. Set False when a caller
                must validate it before promotion.

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
            metadata={"algorithm": "bootstrap"},
            format_instruction=self.format_instruction,  # Phase 6: Save for evaluation
        )

        # Save to storage
        if persist and demo_objects:
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
        persist_candidate: bool = True,
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
            persist_candidate: Promote only after complete holdout coverage clears
                the explicit ``threshold``. False returns a staged candidate.

        Returns:
            Tuple of (BootstrapResult, holdout_score)
        """
        import random

        expected_latest_bytes: Optional[bytes] = None
        expected_latest_absent = False
        if persist_candidate:
            validate_name(agent_name, kind="agent name")
            latest_path = contained_path(
                self.storage.prompts_dir, f"{agent_name}_latest.json"
            )
            if latest_path.exists():
                try:
                    expected_latest_bytes = latest_path.read_bytes()
                except OSError as exc:
                    raise HoldoutGateError(
                        f"Incumbent is unreadable before holdout evaluation: {exc}"
                    ) from exc
            else:
                expected_latest_absent = True

        # Split data
        shuffled = training_data.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - holdout_ratio))
        train_set = shuffled[:split_idx]
        holdout_set = shuffled[split_idx:]

        if not train_set:
            raise HoldoutGateError("Training split is empty")
        if not holdout_set:
            raise HoldoutGateError("Holdout split is empty")
        corpus_identity = holdout_corpus_identity(holdout_set)

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
            persist=False,
        )

        # Evaluate on holdout
        if verbose:
            print(f"\nEvaluating on holdout set...")

        holdout_scores = []
        optimized_prompt = result.optimized_prompt.to_prompt()

        for example in holdout_set:
            full_prompt = f"{optimized_prompt}\n\n## New Input\n\n{example.input_text}"
            try:
                run_result = self.runner.run(full_prompt)
            except Exception as exc:
                raise HoldoutGateError(
                    f"Holdout evaluation raised {type(exc).__name__}: {exc}"
                ) from exc
            if run_result is None or not run_result.success:
                error = getattr(run_result, "error", "missing runner result")
                raise HoldoutGateError(
                    f"Holdout evaluation failed; complete coverage is required: {error}"
                )
            score = metric_fn(example.expected_output, run_result.output)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise HoldoutGateError(f"Holdout metric returned invalid score: {score!r}")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise HoldoutGateError(f"Holdout metric returned invalid score: {score!r}")
            holdout_scores.append(numeric_score)

        if len(holdout_scores) != len(holdout_set):
            raise HoldoutGateError(
                f"Holdout coverage incomplete: {len(holdout_scores)}/{len(holdout_set)}"
            )
        holdout_avg = sum(holdout_scores) / len(holdout_scores)

        if verbose:
            print(f"Holdout score: {holdout_avg:.2f} ({len(holdout_scores)} evaluated)")

        if holdout_avg < threshold:
            raise HoldoutGateError(
                f"Candidate holdout score {holdout_avg:.3f} is below threshold {threshold:.3f}"
            )
        if persist_candidate:
            identity_payload = corpus_identity.as_dict()
            result.optimized_prompt.metadata = {
                **(result.optimized_prompt.metadata or {}),
                "holdout_gate": {
                    "schema_version": 1,
                    "gate_type": "absolute_threshold",
                    "corpus_identity": identity_payload,
                    "candidate_evaluation": {
                        "corpus_identity": identity_payload,
                        "evaluated_examples": corpus_identity.cardinality,
                        "score": holdout_avg,
                    },
                    "threshold": float(threshold),
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            if "artifact_content_hash" in result.optimized_prompt.metadata:
                result.optimized_prompt.metadata["artifact_content_hash"] = (
                    _prompt_semantic_hash(result.optimized_prompt)
                )
            promote_candidate_atomic(
                agent_name,
                result.optimized_prompt,
                self.storage,
                expected_latest_bytes=expected_latest_bytes,
                expected_latest_absent=expected_latest_absent,
            )

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
