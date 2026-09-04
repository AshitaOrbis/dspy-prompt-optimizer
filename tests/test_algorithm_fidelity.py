"""Regression tests for algorithm fidelity, tier staging, and non-persistence."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
LIB_DIR = PROJECT_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from prompt_optimizer.batch import BatchResult, BatchSummary, BatchTarget
from prompt_optimizer.bootstrap import BootstrapFewShot, BootstrapResult, TrainingExample
from prompt_optimizer.storage import Demo, DemoStorage, OptimizedPrompt


def _load_batch_module():
    script = PROJECT_ROOT / "scripts" / "batch_optimize.py"
    spec = importlib.util.spec_from_file_location("batch_optimize_algorithm_test", script)
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
    def __init__(self, model="haiku", **_kwargs):
        self.model = model

    def run(self, _prompt):
        return FakeRunResult()


def _prompt(base_prompt: str) -> OptimizedPrompt:
    return OptimizedPrompt(
        base_prompt=base_prompt,
        demos=[Demo(input_text="in", output_text="out", score=0.8)],
        optimization_date="2026-08-23T00:00:00Z",
        metric_name="metric",
        threshold=0.5,
        avg_score=0.8,
    )


def _bootstrap_result(candidate: OptimizedPrompt) -> BootstrapResult:
    return BootstrapResult(
        optimized_prompt=candidate,
        total_examples=1,
        successful_examples=1,
        failed_examples=0,
        avg_score=0.8,
        traces=[],
    )


def _summary(target: BatchTarget, candidate: OptimizedPrompt) -> BatchSummary:
    result = BatchResult(target, _bootstrap_result(candidate), None, 0.01)
    return BatchSummary(1, 1, 0, [result], "start", "end", 0.01)


@pytest.mark.parametrize("algorithm", ["bootstrap", "copro", "iterative"])
def test_selected_algorithm_runs_once_and_persists_its_metadata(
    monkeypatch, tmp_path, algorithm
):
    module = _load_batch_module()
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("base")
    training_path = tmp_path / "training.jsonl"
    training_path.write_text(json.dumps({"input": "in", "expected": "out"}) + "\n")
    target = BatchTarget(
        name="test-target",
        prompt_path=prompt_path,
        training_data_path=training_path,
        metric_fn=lambda _expected, _actual: 0.8,
    )
    output_dir = tmp_path / "output"
    real_storage = DemoStorage(str(output_dir))
    calls = {"bootstrap": 0, "copro": 0, "iterative": 0}

    def fake_bootstrap(*, storage, **_kwargs):
        calls["bootstrap"] += 1
        candidate = _prompt("bootstrap-selected")
        storage.save_optimized_prompt(target.name, candidate)
        return _summary(target, candidate)

    class FakeCOPRO:
        def __init__(self, _runner, n_variants, storage):
            self.storage = storage

        def optimize_with_bootstrap(self, **_kwargs):
            calls["copro"] += 1
            candidate = _prompt("copro-selected")
            self.storage.save_optimized_prompt(target.name, candidate)
            return SimpleNamespace(improvement=0.2), _bootstrap_result(candidate)

    class FakeIterative:
        def __init__(self, _runner, max_rounds, storage):
            self.storage = storage

        def optimize(self, **_kwargs):
            calls["iterative"] += 1
            candidate = _prompt("iterative-selected")
            self.storage.save_optimized_prompt(target.name, candidate)
            return SimpleNamespace(
                final_prompt=candidate,
                final_score=0.8,
                rounds=[SimpleNamespace(score=0.7)],
                total_rounds=1,
                converged=False,
            )

    monkeypatch.setattr(module, "build_agent_target", lambda *_args, **_kwargs: target)
    monkeypatch.setattr(module, "ClaudeRunner", FakeRunner)
    def storage_factory(storage_dir=None, *_args, **_kwargs):
        if storage_dir is not None and Path(storage_dir) == output_dir:
            return real_storage
        return DemoStorage(storage_dir)

    monkeypatch.setattr(module, "DemoStorage", storage_factory)
    monkeypatch.setattr(module, "optimize_batch_sequential", fake_bootstrap)
    monkeypatch.setattr(module, "COPROOptimizer", FakeCOPRO)
    monkeypatch.setattr(module, "IterativeOptimizer", FakeIterative)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_optimize.py",
            "--agents",
            target.name,
            "--algorithm",
            algorithm,
            "--output",
            str(output_dir),
            "--quiet",
        ],
    )

    module.main()

    expected_calls = {"bootstrap": 0, "copro": 0, "iterative": 0}
    expected_calls[algorithm] = 1
    assert calls == expected_calls
    persisted = real_storage.load_optimized_prompt(target.name)
    assert persisted is not None
    assert persisted.base_prompt == f"{algorithm}-selected"
    assert persisted.metadata and persisted.metadata["algorithm"] == algorithm


def test_tiered_run_uses_real_batch_success_condition_and_promotes_only_final_phase(
    monkeypatch, tmp_path
):
    module = _load_batch_module()
    target = BatchTarget(
        name="tier-target",
        prompt_path=tmp_path / "prompt.md",
        training_data_path=tmp_path / "training.jsonl",
        metric_fn=lambda _expected, _actual: 0.8,
    )
    output_dir = tmp_path / "output"
    real_storage = DemoStorage(str(output_dir))
    phases = []

    def fake_phase(*, runner, storage, **_kwargs):
        phases.append(runner.model)
        candidate = _prompt(f"{runner.model}-selected")
        storage.save_optimized_prompt(target.name, candidate)
        return _summary(target, candidate)

    monkeypatch.setattr(module, "build_agent_target", lambda *_args, **_kwargs: target)
    monkeypatch.setattr(module, "ClaudeRunner", FakeRunner)
    def storage_factory(storage_dir=None, *_args, **_kwargs):
        if storage_dir is not None and Path(storage_dir) == output_dir:
            return real_storage
        return DemoStorage(storage_dir)

    monkeypatch.setattr(module, "DemoStorage", storage_factory)
    monkeypatch.setattr(module, "optimize_batch_sequential", fake_phase)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_optimize.py",
            "--agents",
            target.name,
            "--tier",
            "tiered",
            "--output",
            str(output_dir),
            "--quiet",
        ],
    )

    module.main()

    assert phases == ["haiku", "sonnet", "opus"]
    persisted = real_storage.load_optimized_prompt(target.name)
    assert persisted is not None
    assert persisted.base_prompt == "opus-selected"
    assert persisted.metadata and persisted.metadata["algorithm"] == "tiered"


def test_optimize_with_holdout_can_leave_latest_unchanged(monkeypatch, tmp_path):
    storage = DemoStorage(str(tmp_path / "storage"))
    storage.save_optimized_prompt("agent", _prompt("incumbent"))
    latest = storage.prompts_dir / "agent_latest.json"
    before = latest.read_bytes()
    optimizer = BootstrapFewShot(FakeRunner(), storage=storage, max_demos=1)
    monkeypatch.setattr("prompt_optimizer.bootstrap.random.shuffle", lambda _items: None)

    result, _score = optimizer.optimize_with_holdout(
        base_prompt="candidate",
        training_data=[
            TrainingExample("train", "ok"),
            TrainingExample("holdout", "ok"),
        ],
        metric_fn=lambda _expected, _actual: 1.0,
        holdout_ratio=0.5,
        threshold=0.5,
        agent_name="agent",
        verbose=False,
        persist_candidate=False,
    )

    assert result.optimized_prompt.base_prompt == "candidate"
    assert latest.read_bytes() == before
