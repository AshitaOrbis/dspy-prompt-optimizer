"""Publication-review gate and checkpoint resume regressions."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
LIB_DIR = PROJECT_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from prompt_optimizer.bootstrap import TrainingExample, holdout_corpus_identity
from prompt_optimizer.storage import Demo, DemoStorage, OptimizedPrompt


def _load_module():
    script = PROJECT_ROOT / "scripts" / "optimize_publication_review.py"
    spec = importlib.util.spec_from_file_location("publication_review_gate_test", script)
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
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self, _prompt):
        self.calls += 1
        return self.outcomes.pop(0)


def _prompt(base_prompt: str, *, holdout=None) -> OptimizedPrompt:
    metadata = {"algorithm": "bootstrap"}
    if holdout is not None:
        metadata["holdout_gate"] = {
            "schema_version": 1,
            "corpus_identity": holdout_corpus_identity(holdout).as_dict(),
        }
    return OptimizedPrompt(
        base_prompt=base_prompt,
        demos=[Demo(input_text="in", output_text="out", score=0.8)],
        optimization_date="2026-08-23T00:00:00Z",
        metric_name="publication_review_match",
        threshold=0.0,
        avg_score=0.8,
        metadata=metadata,
    )


def test_publication_gate_missing_holdout_exits_nonzero_and_retains_incumbent(
    monkeypatch, tmp_path
):
    module = _load_module()
    storage = DemoStorage(str(tmp_path / "storage"))
    target = "publication-review-gpt"
    storage.save_optimized_prompt(target, _prompt("incumbent"))
    latest = storage.prompts_dir / f"{target}_latest.json"
    before = latest.read_bytes()
    skill = tmp_path / "SKILL.md"
    skill.write_text("placeholder")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"

    monkeypatch.setattr(module, "SKILL_MD_PATH", skill)
    monkeypatch.setattr(module, "DATASETS_DIR", datasets)
    monkeypatch.setattr(module, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(module, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(module, "DemoStorage", lambda *_args, **_kwargs: storage)
    monkeypatch.setattr(module, "extract_prompt_section", lambda *_args: "candidate")
    monkeypatch.setattr(
        module,
        "load_training_data",
        lambda _model: [TrainingExample("in", "out")],
    )
    monkeypatch.setattr(
        module,
        "run_training_with_checkpoints",
        lambda **_kwargs: (
            [{"input_text": "in", "output_text": "out", "score": 0.8}],
            {"0": 0.8},
            0.8,
        ),
    )
    monkeypatch.setattr(module, "get_runner_for_model", lambda _model: FakeRunner([]))
    monkeypatch.setattr(sys, "argv", ["optimize_publication_review.py", "--model", "gpt", "--holdout-gate", "--threshold", "0"])

    code = 0
    try:
        module.main()
    except SystemExit as exc:
        code = int(exc.code or 0)

    assert code != 0
    assert latest.read_bytes() == before


def test_publication_status_write_failure_rolls_back_promoted_prompt(monkeypatch, tmp_path):
    module = _load_module()
    storage = DemoStorage(str(tmp_path / "storage"))
    target = "publication-review-gpt"
    gate_holdout = [TrainingExample("holdout", "expected")]
    storage.save_optimized_prompt(target, _prompt("incumbent", holdout=gate_holdout))
    latest = storage.prompts_dir / f"{target}_latest.json"
    before = latest.read_bytes()
    skill = tmp_path / "SKILL.md"
    skill.write_text("placeholder")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "publication-review-gpt-holdout.jsonl").write_text(
        '{"input":"holdout","expected":"expected"}\n'
    )

    monkeypatch.setattr(module, "SKILL_MD_PATH", skill)
    monkeypatch.setattr(module, "DATASETS_DIR", datasets)
    monkeypatch.setattr(module, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(module, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(module, "DemoStorage", lambda *_args, **_kwargs: storage)
    monkeypatch.setattr(module, "extract_prompt_section", lambda *_args: "candidate")
    monkeypatch.setattr(
        module,
        "load_training_data",
        lambda _model: [TrainingExample("in", "out")],
    )
    monkeypatch.setattr(
        module,
        "run_training_with_checkpoints",
        lambda **_kwargs: (
            [{"input_text": "in", "output_text": "out", "score": 0.8}],
            {"0": 0.8},
            0.8,
        ),
    )
    runner = FakeRunner([FakeRunResult(), FakeRunResult()])
    monkeypatch.setattr(module, "get_runner_for_model", lambda _model: runner)
    monkeypatch.setattr(module, "publication_review_match", lambda *_args: 0.8)
    monkeypatch.setattr(
        module,
        "update_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("status write failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_publication_review.py",
            "--model",
            "gpt",
            "--holdout-gate",
            "--threshold",
            "0",
        ],
    )

    code = 0
    try:
        module.main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:
        code = 1

    assert code != 0
    assert runner.calls == 2
    assert latest.read_bytes() == before


def test_resume_retries_an_index_that_failed_transiently(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(module, "CONSECUTIVE_FAILURE_LIMIT", 1)
    example = TrainingExample("in", "out", {"post_slug": "example"})

    first_runner = FakeRunner([FakeRunResult(success=False, error="quota")])
    module.run_training_with_checkpoints(
        model="gpt",
        runner=first_runner,
        base_prompt="base",
        training_data=[example],
        format_instruction="format",
        threshold=0.0,
        max_demos=1,
        verbose=False,
    )
    assert first_runner.calls == 1

    second_runner = FakeRunner([FakeRunResult(success=True, output="out")])
    _demos, scores, avg_score = module.run_training_with_checkpoints(
        model="gpt",
        runner=second_runner,
        base_prompt="base",
        training_data=[example],
        format_instruction="format",
        threshold=0.0,
        max_demos=1,
        verbose=False,
    )

    assert second_runner.calls == 1
    assert scores["0"] >= 0.0
    assert avg_score >= 0.0
    checkpoint = module.Checkpoint.load("gpt")
    assert checkpoint is not None
    assert checkpoint.completed_indices == [0]
