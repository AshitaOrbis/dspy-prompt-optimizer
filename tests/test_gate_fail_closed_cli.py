"""CLI regressions for staged, fail-closed holdout promotion."""

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
from prompt_optimizer.bootstrap import (
    BootstrapResult,
    TrainingExample,
    holdout_corpus_identity,
)
from prompt_optimizer.storage import Demo, DemoStorage, OptimizedPrompt


def _load_batch_module():
    script = PROJECT_ROOT / "scripts" / "batch_optimize.py"
    spec = importlib.util.spec_from_file_location("batch_optimize_gate_test", script)
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
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.calls = 0

    def run(self, _prompt):
        self.calls += 1
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return FakeRunResult()


def _prompt(base_prompt: str, *, algorithm: str = "bootstrap") -> OptimizedPrompt:
    return OptimizedPrompt(
        base_prompt=base_prompt,
        demos=[Demo(input_text="input", output_text="output", score=0.8)],
        optimization_date="2026-08-23T00:00:00Z",
        metric_name="metric",
        threshold=0.5,
        avg_score=0.8,
        metadata={"algorithm": algorithm},
    )


def _summary(target: BatchTarget, candidate: OptimizedPrompt) -> BatchSummary:
    result = BootstrapResult(
        optimized_prompt=candidate,
        total_examples=1,
        successful_examples=1,
        failed_examples=0,
        avg_score=0.8,
        traces=[],
    )
    batch_result = BatchResult(target=target, result=result, error=None, duration_seconds=0.01)
    return BatchSummary(
        total_targets=1,
        successful=1,
        failed=0,
        results=[batch_result],
        start_time="2026-08-23T00:00:00Z",
        end_time="2026-08-23T00:00:01Z",
        total_duration_seconds=0.01,
    )


def _run_gate_case(
    monkeypatch,
    tmp_path: Path,
    *,
    holdout_kind: str = "valid",
    metric=None,
    outcomes=None,
    with_incumbent: bool = True,
):
    module = _load_batch_module()
    target_name = "code-reviewer" if metric is not None else "missing-metric-target"
    metric_fn = metric
    target = BatchTarget(
        name=target_name,
        prompt_path=tmp_path / "prompt.md",
        training_data_path=tmp_path / "train.jsonl",
        metric_fn=metric_fn,
    )
    candidate = _prompt("candidate")
    output_dir = tmp_path / "output"
    storage = DemoStorage(str(output_dir))
    latest_path = storage.prompts_dir / f"{target_name}_latest.json"
    if with_incumbent:
        incumbent = _prompt("incumbent")
        if holdout_kind == "valid":
            identity = holdout_corpus_identity([TrainingExample("h", "e")])
            incumbent.metadata["holdout_gate"] = {
                "schema_version": 1,
                "corpus_identity": identity.as_dict(),
            }
        storage.save_optimized_prompt(target_name, incumbent)
    before = latest_path.read_bytes() if latest_path.exists() else None

    holdout_dir = tmp_path / "holdouts"
    holdout_dir.mkdir()
    basename = "code-reviews" if target_name == "code-reviewer" else target_name
    holdout_path = holdout_dir / f"{basename}-holdout.jsonl"
    if holdout_kind == "valid":
        holdout_path.write_text(json.dumps({"input": "h", "expected": "e"}) + "\n")
    elif holdout_kind == "empty":
        holdout_path.write_text("")
    elif holdout_kind == "malformed":
        holdout_path.write_text("{not-json}\n")
    elif holdout_kind == "unreadable":
        holdout_path.mkdir()
    elif holdout_kind != "absent":
        raise AssertionError(f"unknown holdout kind: {holdout_kind}")

    runner = FakeRunner(outcomes)

    def fake_optimize(*, storage, **_kwargs):
        # This writes to the production store in the old implementation and to
        # an isolated staging store in the fixed implementation.
        storage.save_optimized_prompt(target_name, candidate)
        return _summary(target, candidate)

    monkeypatch.setattr(module, "build_agent_target", lambda *_args, **_kwargs: target)
    monkeypatch.setattr(module, "ClaudeRunner", lambda **_kwargs: runner)
    def storage_factory(storage_dir=None, *_args, **_kwargs):
        if storage_dir is not None and Path(storage_dir) == output_dir:
            return storage
        return DemoStorage(storage_dir)

    monkeypatch.setattr(module, "DemoStorage", storage_factory)
    monkeypatch.setattr(module, "optimize_batch_sequential", fake_optimize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_optimize.py",
            "--agents",
            target_name,
            "--holdout-gate",
            "--holdout-dir",
            str(holdout_dir),
            "--output",
            str(output_dir),
            "--quiet",
        ],
    )

    exit_code = 0
    try:
        module.main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception:
        exit_code = 1

    after = latest_path.read_bytes() if latest_path.exists() else None
    return exit_code, before, after


@pytest.mark.parametrize("holdout_kind", ["absent", "empty", "unreadable", "malformed"])
def test_holdout_input_failure_exits_nonzero_without_changing_latest(
    monkeypatch, tmp_path, holdout_kind
):
    code, before, after = _run_gate_case(
        monkeypatch,
        tmp_path,
        holdout_kind=holdout_kind,
        metric=lambda _expected, _actual: 0.8,
    )
    assert code != 0
    assert after == before


def test_missing_metric_exits_nonzero_without_changing_latest(monkeypatch, tmp_path):
    code, before, after = _run_gate_case(monkeypatch, tmp_path, metric=None)
    assert code != 0
    assert after == before


def test_one_evaluation_raising_exits_nonzero_without_changing_latest(monkeypatch, tmp_path):
    code, before, after = _run_gate_case(
        monkeypatch,
        tmp_path,
        metric=lambda _expected, _actual: 0.8,
        outcomes=[RuntimeError("runner outage")],
    )
    assert code != 0
    assert after == before


def test_all_evaluations_failing_exits_nonzero_without_changing_latest(monkeypatch, tmp_path):
    failures = [FakeRunResult(success=False, error="outage") for _ in range(2)]
    code, before, after = _run_gate_case(
        monkeypatch,
        tmp_path,
        metric=lambda _expected, _actual: 0.8,
        outcomes=failures,
    )
    assert code != 0
    assert after == before


def test_none_score_exits_nonzero_without_changing_latest(monkeypatch, tmp_path):
    code, before, after = _run_gate_case(
        monkeypatch,
        tmp_path,
        metric=lambda _expected, _actual: None,
    )
    assert code != 0
    assert after == before


def test_missing_backup_exits_nonzero_without_creating_latest(monkeypatch, tmp_path):
    code, before, after = _run_gate_case(
        monkeypatch,
        tmp_path,
        metric=lambda _expected, _actual: 0.8,
        with_incumbent=False,
    )
    assert before is None
    assert code != 0
    assert after is None


def test_complete_symmetric_gate_promotes_the_staged_candidate(monkeypatch, tmp_path):
    code, before, after = _run_gate_case(
        monkeypatch,
        tmp_path,
        metric=lambda _expected, _actual: 0.8,
    )
    assert code == 0
    assert after != before
    assert json.loads(after)["base_prompt"] == "candidate"
