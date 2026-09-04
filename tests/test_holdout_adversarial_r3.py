"""Adversarial regressions for corpus-bound holdout promotion."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from lib.prompt_optimizer.bootstrap import (
    BootstrapFewShot,
    HoldoutGateError,
    TrainingExample,
    promote_candidate_with_holdout,
)
from lib.prompt_optimizer.batch import BatchSummary
from lib.prompt_optimizer.storage import Demo, DemoStorage, OptimizedPrompt


PROJECT_ROOT = Path(__file__).parent.parent


def _load_batch_module():
    script = PROJECT_ROOT / "scripts" / "batch_optimize.py"
    spec = importlib.util.spec_from_file_location("batch_optimize_r3_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeRunResult:
    output: str = "ok"
    success: bool = True
    error: str = ""


class FakeRunner:
    def __init__(self, on_call=None):
        self.calls = 0
        self.on_call = on_call

    def run(self, _prompt):
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)
        return FakeRunResult()


def _corpus_identity(examples: list[TrainingExample]) -> dict[str, object]:
    ordered_examples = [
        {
            "input": example.input_text,
            "expected": example.expected_output,
            "metadata": example.metadata,
        }
        for example in examples
    ]
    encoded = json.dumps(
        ordered_examples,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "cardinality": len(examples),
    }


def _prompt(
    label: str,
    *,
    corpus_identity: dict[str, object] | None = None,
) -> OptimizedPrompt:
    metadata: dict[str, object] = {"algorithm": "bootstrap"}
    if corpus_identity is not None:
        metadata["holdout_gate"] = {
            "schema_version": 1,
            "corpus_identity": corpus_identity,
        }
    return OptimizedPrompt(
        base_prompt=label,
        demos=[Demo("input", "output", 1.0)],
        optimization_date="2026-08-23T00:00:00Z",
        metric_name="fixture",
        threshold=0.7,
        avg_score=1.0,
        metadata=metadata,
    )


def _latest(storage: DemoStorage) -> OptimizedPrompt:
    prompt = storage.load_optimized_prompt("target")
    assert prompt is not None
    return prompt


def test_shrunk_holdout_is_rejected_against_recorded_corpus(tmp_path):
    full_holdout = [
        TrainingExample("one", "expected-one"),
        TrainingExample("two", "expected-two"),
    ]
    storage = DemoStorage(str(tmp_path / "storage"))
    storage.save_optimized_prompt(
        "target",
        _prompt("incumbent", corpus_identity=_corpus_identity(full_holdout)),
    )

    with pytest.raises(HoldoutGateError, match="corpus identity"):
        promote_candidate_with_holdout(
            "target",
            _prompt("candidate"),
            full_holdout[:1],
            lambda _expected, _actual: 1.0,
            FakeRunner(),
            storage,
            verbose=False,
        )

    assert _latest(storage).base_prompt == "incumbent"


def test_same_length_substitute_holdout_is_rejected(tmp_path):
    recorded_holdout = [TrainingExample("recorded", "expected")]
    substituted_holdout = [TrainingExample("substituted", "expected")]
    storage = DemoStorage(str(tmp_path / "storage"))
    storage.save_optimized_prompt(
        "target",
        _prompt("incumbent", corpus_identity=_corpus_identity(recorded_holdout)),
    )

    with pytest.raises(HoldoutGateError, match="corpus identity"):
        promote_candidate_with_holdout(
            "target",
            _prompt("candidate"),
            substituted_holdout,
            lambda _expected, _actual: 1.0,
            FakeRunner(),
            storage,
            verbose=False,
        )

    assert _latest(storage).base_prompt == "incumbent"


def test_unbound_incumbent_is_refused(tmp_path):
    holdout = [TrainingExample("one", "expected")]
    storage = DemoStorage(str(tmp_path / "storage"))
    storage.save_optimized_prompt("target", _prompt("incumbent"))

    with pytest.raises(HoldoutGateError, match="recorded holdout corpus identity"):
        promote_candidate_with_holdout(
            "target",
            _prompt("candidate"),
            holdout,
            lambda _expected, _actual: 1.0,
            FakeRunner(),
            storage,
            verbose=False,
        )

    assert _latest(storage).base_prompt == "incumbent"


def test_promotion_artifact_records_same_corpus_for_both_arms(tmp_path):
    holdout = [
        TrainingExample("one", "expected-one", {"ordinal": 1}),
        TrainingExample("two", "expected-two", {"ordinal": 2}),
    ]
    identity = _corpus_identity(holdout)
    storage = DemoStorage(str(tmp_path / "storage"))
    storage.save_optimized_prompt(
        "target",
        _prompt("incumbent", corpus_identity=identity),
    )

    result = promote_candidate_with_holdout(
        "target",
        _prompt("candidate"),
        holdout,
        lambda _expected, _actual: 1.0,
        FakeRunner(),
        storage,
        verbose=False,
    )

    latest = _latest(storage)
    evidence = latest.metadata["holdout_gate"]
    assert evidence["gate_type"] == "comparative"
    assert evidence["corpus_identity"] == identity
    assert evidence["candidate_evaluation"]["corpus_identity"] == identity
    assert evidence["incumbent_evaluation"]["corpus_identity"] == identity
    assert evidence["candidate_evaluation"]["evaluated_examples"] == len(holdout)
    assert evidence["incumbent_evaluation"]["evaluated_examples"] == len(holdout)
    assert result.corpus_sha256 == identity["sha256"]
    assert result.corpus_cardinality == identity["cardinality"]


def test_concurrent_latest_replacement_is_not_overwritten(tmp_path):
    holdout = [TrainingExample("one", "expected")]
    identity = _corpus_identity(holdout)
    storage = DemoStorage(str(tmp_path / "storage"))
    storage.save_optimized_prompt(
        "target",
        _prompt("incumbent", corpus_identity=identity),
    )

    def concurrent_write(call_number: int) -> None:
        if call_number == 2:
            storage.save_optimized_prompt(
                "target",
                _prompt("concurrent-winner", corpus_identity=identity),
            )

    with pytest.raises(HoldoutGateError, match="changed during holdout evaluation"):
        promote_candidate_with_holdout(
            "target",
            _prompt("candidate"),
            holdout,
            lambda _expected, _actual: 1.0,
            FakeRunner(concurrent_write),
            storage,
            verbose=False,
        )

    assert _latest(storage).base_prompt == "concurrent-winner"


def test_wrong_metric_type_still_fails_without_replacing_incumbent(tmp_path):
    holdout = [TrainingExample("one", "expected")]
    identity = _corpus_identity(holdout)
    storage = DemoStorage(str(tmp_path / "storage"))
    storage.save_optimized_prompt(
        "target",
        _prompt("incumbent", corpus_identity=identity),
    )

    with pytest.raises(HoldoutGateError, match="non-numeric"):
        promote_candidate_with_holdout(
            "target",
            _prompt("candidate"),
            holdout,
            lambda _expected, _actual: "1.0",
            FakeRunner(),
            storage,
            verbose=False,
        )

    assert _latest(storage).base_prompt == "incumbent"


def test_optimize_with_holdout_records_absolute_gate_corpus(monkeypatch, tmp_path):
    storage = DemoStorage(str(tmp_path / "storage"))
    optimizer = BootstrapFewShot(FakeRunner(), storage=storage, max_demos=1)
    monkeypatch.setattr("lib.prompt_optimizer.bootstrap.random.shuffle", lambda _items: None)

    optimizer.optimize_with_holdout(
        base_prompt="candidate",
        training_data=[
            TrainingExample("train", "ok"),
            TrainingExample("holdout", "ok"),
        ],
        metric_fn=lambda _expected, _actual: 1.0,
        holdout_ratio=0.5,
        threshold=0.5,
        agent_name="target",
        verbose=False,
    )

    latest = _latest(storage)
    evidence = latest.metadata["holdout_gate"]
    expected_identity = _corpus_identity([TrainingExample("holdout", "ok")])
    assert evidence["gate_type"] == "absolute_threshold"
    assert evidence["corpus_identity"] == expected_identity
    assert evidence["candidate_evaluation"]["corpus_identity"] == expected_identity
    assert evidence["candidate_evaluation"]["evaluated_examples"] == 1


def test_optimize_with_holdout_refuses_concurrent_latest_replacement(
    monkeypatch, tmp_path
):
    storage = DemoStorage(str(tmp_path / "storage"))
    storage.save_optimized_prompt("target", _prompt("incumbent"))

    def concurrent_write(call_number: int) -> None:
        if call_number == 2:
            storage.save_optimized_prompt("target", _prompt("concurrent-winner"))

    optimizer = BootstrapFewShot(
        FakeRunner(concurrent_write), storage=storage, max_demos=1
    )
    monkeypatch.setattr("lib.prompt_optimizer.bootstrap.random.shuffle", lambda _items: None)

    with pytest.raises(HoldoutGateError, match="changed during holdout evaluation"):
        optimizer.optimize_with_holdout(
            base_prompt="candidate",
            training_data=[
                TrainingExample("train", "ok"),
                TrainingExample("holdout", "ok"),
            ],
            metric_fn=lambda _expected, _actual: 1.0,
            holdout_ratio=0.5,
            threshold=0.5,
            agent_name="target",
            verbose=False,
        )

    assert _latest(storage).base_prompt == "concurrent-winner"


def test_allow_unverified_cannot_mark_target_or_run_complete(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "code-reviews.jsonl").write_text(
        '{"input":"i","expected":"e"}\n', encoding="utf-8"
    )
    output = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_marker = tmp_path / "python-invoked"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"touch {invocation_marker}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    process = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "run_optimization.sh"),
            "--foreground",
            "--targets",
            "code-reviewer",
            "--datasets-dir",
            str(datasets),
            "--output-dir",
            str(output),
            "--allow-unverified",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert process.returncode == 2
    assert status["current"] == "unverified"
    assert status["completed"] == []
    assert status["unverified"] == ["code-reviewer"]
    assert "TERMINAL_SIGNAL: OPTIMIZATION_COMPLETE" not in process.stdout
    assert not invocation_marker.exists()


def test_batch_result_identity_matches_corpus_bound_artifact(monkeypatch, tmp_path):
    module = _load_batch_module()
    from prompt_optimizer.batch import _candidate_content_hash as batch_content_hash
    from prompt_optimizer.storage import (
        Demo as BatchDemo,
        DemoStorage as BatchDemoStorage,
        OptimizedPrompt as BatchOptimizedPrompt,
    )

    def batch_prompt(
        label: str, corpus_identity: dict[str, object] | None = None
    ) -> BatchOptimizedPrompt:
        metadata: dict[str, object] = {"algorithm": "bootstrap"}
        if corpus_identity is not None:
            metadata["holdout_gate"] = {
                "schema_version": 1,
                "corpus_identity": corpus_identity,
            }
        return BatchOptimizedPrompt(
            base_prompt=label,
            demos=[BatchDemo("input", "output", 1.0)],
            optimization_date="2026-08-23T00:00:00Z",
            metric_name="fixture",
            threshold=0.7,
            avg_score=1.0,
            metadata=metadata,
        )

    target = module.BatchTarget(
        name="code-reviewer",
        prompt_path=tmp_path / "prompt.md",
        training_data_path=tmp_path / "training.jsonl",
        metric_fn=lambda _expected, _actual: 1.0,
    )
    candidate = batch_prompt("candidate")
    bootstrap_result = module.BootstrapResult(
        optimized_prompt=candidate,
        total_examples=1,
        successful_examples=1,
        failed_examples=0,
        avg_score=1.0,
        traces=[],
    )
    summary = module.BatchSummary(
        total_targets=1,
        successful=1,
        failed=0,
        results=[module.BatchResult(target, bootstrap_result, None, 0.01)],
        start_time="2026-08-23T00:00:00Z",
        end_time="2026-08-23T00:00:01Z",
        total_duration_seconds=0.01,
    )
    output_dir = tmp_path / "output"
    storage = BatchDemoStorage(str(output_dir))
    holdout = [TrainingExample("h", "e")]
    identity = _corpus_identity(holdout)
    storage.save_optimized_prompt(
        target.name,
        batch_prompt("incumbent", corpus_identity=identity),
    )
    holdout_dir = tmp_path / "holdouts"
    holdout_dir.mkdir()
    (holdout_dir / "code-reviews-holdout.jsonl").write_text(
        '{"input":"h","expected":"e"}\n', encoding="utf-8"
    )
    captured: dict[str, BatchSummary] = {}

    monkeypatch.setattr(module, "build_agent_target", lambda *_args, **_kwargs: target)
    monkeypatch.setattr(module, "ClaudeRunner", lambda **_kwargs: FakeRunner())

    def storage_factory(storage_dir=None, *_args, **_kwargs):
        if storage_dir is not None and Path(storage_dir) == output_dir:
            return storage
        return BatchDemoStorage(storage_dir)

    monkeypatch.setattr(module, "DemoStorage", storage_factory)
    monkeypatch.setattr(module, "optimize_batch_sequential", lambda **_kwargs: summary)

    def capture_report(result_summary, output_path=None):
        captured["summary"] = result_summary
        return "report"

    monkeypatch.setattr(module, "generate_batch_report", capture_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_optimize.py",
            "--agents",
            target.name,
            "--holdout-gate",
            "--holdout-dir",
            str(holdout_dir),
            "--output",
            str(output_dir),
            "--quiet",
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        detail = captured.get("summary", summary).results[0].error
        raise AssertionError(f"batch gate unexpectedly exited {exc.code}: {detail}") from exc

    result = captured["summary"].results[0]
    persisted = storage.load_optimized_prompt(target.name)
    assert persisted is not None
    assert result.artifact is not None
    assert result.artifact.content_hash == persisted.metadata["artifact_content_hash"]
    assert result.artifact.content_hash == batch_content_hash(persisted)
