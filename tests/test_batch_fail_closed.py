"""Fail-closed batch optimization success and artifact identity tests."""

from pathlib import Path

from lib.prompt_optimizer import batch
from lib.prompt_optimizer.batch import BatchTarget, optimize_single_target
from lib.prompt_optimizer.bootstrap import BootstrapResult, TrainingExample
from lib.prompt_optimizer.storage import Demo, DemoStorage, OptimizedPrompt


def _prompt(demo_count: int) -> OptimizedPrompt:
    return OptimizedPrompt(
        base_prompt="Base prompt",
        demos=[Demo(f"input-{i}", f"output-{i}", 0.9) for i in range(demo_count)],
        optimization_date="2026-08-23T00:00:00Z",
        metric_name="metric",
        threshold=0.7,
        avg_score=0.9 if demo_count else 0.0,
    )


def _bootstrap_result(
    *, total: int, failed: int, selected: int, successful_demos: int | None = None
) -> BootstrapResult:
    return BootstrapResult(
        optimized_prompt=_prompt(selected),
        total_examples=total,
        successful_examples=(selected if successful_demos is None else successful_demos),
        failed_examples=failed,
        avg_score=0.9 if selected else 0.0,
        traces=[],
    )


def _target(tmp_path: Path, **kwargs) -> BatchTarget:
    prompt_path = tmp_path / "prompt.md"
    training_path = tmp_path / "training.jsonl"
    prompt_path.write_text("Base prompt")
    training_path.write_text('{"input":"x","expected":"y"}\n')
    return BatchTarget(
        name="test-agent",
        prompt_path=prompt_path,
        training_data_path=training_path,
        metric_fn=lambda expected, actual: 1.0,
        **kwargs,
    )


def _install_fake_optimizer(monkeypatch, result: BootstrapResult):
    class FakeOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, *args, **kwargs):
            return result

    monkeypatch.setattr(batch, "BootstrapFewShot", FakeOptimizer)


def test_all_model_calls_failed_is_batch_failure_with_stale_artifact(
    tmp_path, monkeypatch
):
    storage = DemoStorage(storage_dir=str(tmp_path / "storage"))
    storage.save_optimized_prompt("test-agent", _prompt(1))
    _install_fake_optimizer(
        monkeypatch, _bootstrap_result(total=2, failed=2, selected=0)
    )

    result = optimize_single_target(
        _target(tmp_path), runner=object(), storage=storage, verbose=False
    )

    assert result.error is not None
    assert result.result is None
    assert getattr(result, "artifact", None) is None


def test_zero_demo_run_is_batch_failure(tmp_path, monkeypatch):
    storage = DemoStorage(storage_dir=str(tmp_path / "storage"))
    _install_fake_optimizer(
        monkeypatch, _bootstrap_result(total=2, failed=0, selected=0)
    )

    result = optimize_single_target(
        _target(tmp_path), runner=object(), storage=storage, verbose=False
    )

    assert result.error is not None
    assert result.result is None


def test_required_demo_count_must_be_met(tmp_path, monkeypatch):
    assert "min_demos" in BatchTarget.__dataclass_fields__
    storage = DemoStorage(storage_dir=str(tmp_path / "storage"))
    _install_fake_optimizer(
        monkeypatch, _bootstrap_result(total=2, failed=0, selected=1)
    )

    result = optimize_single_target(
        _target(tmp_path, min_demos=2),
        runner=object(),
        storage=storage,
        verbose=False,
    )

    assert result.error is not None
    assert result.result is None


class NoWriteStorage(DemoStorage):
    def save_demos(self, agent_name, demos):
        return self.demos_dir / f"{agent_name}.json"

    def save_optimized_prompt(self, agent_name, optimized, format="both"):
        return {"latest": self.prompts_dir / f"{agent_name}_latest.json"}


def test_no_artifact_written_is_batch_failure(tmp_path, monkeypatch):
    storage = NoWriteStorage(storage_dir=str(tmp_path / "storage"))
    _install_fake_optimizer(
        monkeypatch, _bootstrap_result(total=2, failed=0, selected=1)
    )

    result = optimize_single_target(
        _target(tmp_path), runner=object(), storage=storage, verbose=False
    )

    assert result.error is not None
    assert result.result is None
    assert getattr(result, "artifact", None) is None


def test_success_is_bound_to_new_artifact_identity(tmp_path, monkeypatch):
    storage = DemoStorage(storage_dir=str(tmp_path / "storage"))
    _install_fake_optimizer(
        monkeypatch, _bootstrap_result(total=2, failed=0, selected=1)
    )

    result = optimize_single_target(
        _target(tmp_path), runner=object(), storage=storage, verbose=False
    )

    assert result.error is None
    assert result.artifact is not None
    persisted = storage.load_optimized_prompt("test-agent")
    assert persisted is not None
    assert persisted.metadata["artifact_run_id"] == result.artifact.run_id
    assert persisted.metadata["artifact_created_at"] == result.artifact.created_at
    assert persisted.metadata["artifact_content_hash"] == result.artifact.content_hash


def test_zero_demo_mode_must_be_explicit_and_have_successful_evaluation(
    tmp_path, monkeypatch
):
    assert "allow_zero_demos" in BatchTarget.__dataclass_fields__
    storage = DemoStorage(storage_dir=str(tmp_path / "storage"))
    _install_fake_optimizer(
        monkeypatch, _bootstrap_result(total=2, failed=0, selected=0)
    )

    result = optimize_single_target(
        _target(tmp_path, allow_zero_demos=True, max_demos=0),
        runner=object(),
        storage=storage,
        verbose=False,
    )

    assert result.error is None
    assert result.artifact is not None
