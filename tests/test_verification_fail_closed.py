"""Fail-closed verification and regression accounting tests."""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lib.prompt_optimizer.bootstrap import TrainingExample
from lib.prompt_optimizer.storage import Demo, DemoStorage, OptimizedPrompt
from lib.prompt_optimizer import verification as verification_module
from lib.prompt_optimizer.verification import (
    VerificationReport,
    VerificationSuite,
    pre_flight_holdout_check,
)


@dataclass
class FakeRunResult:
    output: str = "output"
    success: bool = True
    error: str = ""


def _status_value(result) -> str | None:
    status = getattr(result, "status", None)
    return getattr(status, "value", status)


def _storage_with_prompt(tmp_path: Path, agent_name: str = "test-agent") -> DemoStorage:
    storage = DemoStorage(storage_dir=str(tmp_path / "storage"))
    storage.save_optimized_prompt(
        agent_name,
        OptimizedPrompt(
            base_prompt="Base prompt",
            demos=[Demo("input", "output", 0.8)],
            optimization_date="2026-08-23T00:00:00Z",
            metric_name="metric",
            threshold=0.7,
            avg_score=0.8,
        ),
    )
    return storage


@pytest.mark.parametrize(
    ("contents", "create_file"),
    [
        (None, False),
        ("[]", True),
        ("{broken", True),
        ('{"other": 0.8}', True),
        ('{"test-agent": null}', True),
        ('{"test-agent": "x"}', True),
        ('{"test-agent": NaN}', True),
        ('{"test-agent": Infinity}', True),
        ('{"test-agent": -0.1}', True),
        ('{"test-agent": 1.1}', True),
    ],
)
def test_requested_regression_requires_assessable_exact_baseline(
    tmp_path, contents, create_file
):
    baseline_path = tmp_path / "baseline.json"
    if create_file:
        baseline_path.write_text(contents)

    suite = VerificationSuite(
        runner=MagicMock(), baseline_scores_path=baseline_path
    )
    result = suite.check_regression("test-agent", 0.8, verbose=False)

    assert _status_value(result) == "NOT_ASSESSABLE"
    assert result.baseline_score is None
    assert result.regressed is False


def test_regression_is_a_typed_failure(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"test-agent": 0.9}))
    suite = VerificationSuite(
        runner=MagicMock(), baseline_scores_path=baseline_path
    )

    result = suite.check_regression(
        "test-agent", 0.8, regression_threshold=0.02, verbose=False
    )

    assert _status_value(result) == "FAIL"
    assert result.regressed is True


def test_non_finite_regression_threshold_is_error(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"test-agent": 0.9}))
    suite = VerificationSuite(
        runner=MagicMock(), baseline_scores_path=baseline_path
    )

    result = suite.check_regression(
        "test-agent", 0.1, regression_threshold=float("nan"), verbose=False
    )

    assert _status_value(result) == "ERROR"
    assert result.regressed is False


def test_allow_missing_does_not_waive_regression_errors(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"test-agent": 0.9}))
    suite = VerificationSuite(
        runner=MagicMock(), baseline_scores_path=baseline_path
    )
    regression = suite.check_regression("test-agent", float("nan"), verbose=False)
    report = VerificationReport(
        timestamp="2026-08-23T00:00:00Z",
        agents_verified=0,
        holdout_results=[],
        regression_results=[regression],
        cross_validation_results=[],
        summary={},
    )

    assert _status_value(regression) == "ERROR"
    assert report.has_blocking_failures(allow_missing=True) is True


def test_allow_missing_does_not_waive_cross_validation_evidence_errors(tmp_path):
    suite = VerificationSuite(runner=MagicMock())
    report = suite.run_full_verification(
        agent_names=["test-agent"],
        holdout_data_dir=tmp_path,
        training_data_dir=tmp_path,
        metric_fns={"test-agent": lambda expected, actual: 1.0},
        check_regression=False,
        run_cross_validation=True,
        verbose=False,
    )

    assert report.cross_validation_results[0].error_category == "evidence"
    assert report.has_blocking_failures(allow_missing=True) is True


def test_mid_set_exception_is_recorded_without_dropping_example(tmp_path):
    runner = MagicMock()
    runner.run.side_effect = [
        FakeRunResult(output="good"),
        RuntimeError("runner exploded"),
    ]
    suite = VerificationSuite(
        runner=runner, storage=_storage_with_prompt(tmp_path)
    )
    holdout = [
        TrainingExample("one", "expected"),
        TrainingExample("two", "expected"),
    ]

    try:
        result = suite.run_holdout_evaluation(
            "test-agent", holdout, lambda expected, actual: 0.9, verbose=False
        )
    except RuntimeError:
        result = None

    assert result is not None, "a raised evaluation must become an explicit result"
    assert _status_value(result) == "ERROR"
    assert len(result.evaluations) == 2
    assert [_status_value(item) for item in result.evaluations] == ["PASS", "ERROR"]
    assert result.scores == [0.9, 0.0]
    assert result.holdout_score == pytest.approx(0.45)
    assert result.successful_evals == 1
    assert result.failed_evals == 1


def test_empty_holdout_is_not_run(tmp_path):
    suite = VerificationSuite(
        runner=MagicMock(), storage=_storage_with_prompt(tmp_path)
    )

    result = suite.run_holdout_evaluation(
        "test-agent", [], lambda expected, actual: 1.0, verbose=False
    )

    assert _status_value(result) == "NOT_RUN"
    assert result.passed is False


def test_allow_missing_does_not_waive_missing_optimized_prompt(tmp_path):
    suite = VerificationSuite(
        runner=MagicMock(), storage=DemoStorage(storage_dir=str(tmp_path / "storage"))
    )
    result = suite.run_holdout_evaluation(
        "test-agent",
        [TrainingExample("input", "expected")],
        lambda expected, actual: 1.0,
        verbose=False,
    )
    report = VerificationReport(
        timestamp="2026-08-23T00:00:00Z",
        agents_verified=0,
        holdout_results=[result],
        regression_results=[],
        cross_validation_results=[],
        summary={},
    )

    assert _status_value(result) == "NOT_RUN"
    assert report.has_blocking_failures(allow_missing=True) is True


def test_prompt_render_error_retains_every_requested_example(tmp_path):
    storage = MagicMock()
    optimized = MagicMock()
    optimized.to_prompt.side_effect = ValueError("malformed prompt")
    storage.load_optimized_prompt.return_value = optimized
    suite = VerificationSuite(runner=MagicMock(), storage=storage)
    holdout = [
        TrainingExample("one", "expected"),
        TrainingExample("two", "expected"),
    ]

    try:
        result = suite.run_holdout_evaluation(
            "test-agent", holdout, lambda expected, actual: 1.0, verbose=False
        )
    except ValueError:
        result = None

    assert result is not None
    assert _status_value(result) == "ERROR"
    assert len(result.evaluations) == 2


def test_requested_target_that_is_skipped_remains_in_report(tmp_path):
    suite = VerificationSuite(runner=MagicMock())

    report = suite.run_full_verification(
        agent_names=["missing-target"],
        holdout_data_dir=tmp_path / "holdout",
        training_data_dir=tmp_path / "training",
        metric_fns={},
        verbose=False,
    )

    assert len(report.holdout_results) == 1
    assert report.holdout_results[0].agent_name == "missing-target"
    assert _status_value(report.holdout_results[0]) == "NOT_RUN"
    assert report.agents_verified == 0
    assert report.summary["incomplete"] == 1


def test_report_average_keeps_incomplete_targets_in_denominator(tmp_path):
    storage = _storage_with_prompt(tmp_path, agent_name="completed")
    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    (holdout_dir / "completed-holdout.jsonl").write_text(
        '{"input":"x","expected":"y"}\n'
    )
    runner = MagicMock()
    runner.run.return_value = FakeRunResult(output="good")
    suite = VerificationSuite(runner=runner, storage=storage)

    report = suite.run_full_verification(
        agent_names=["completed", "skipped"],
        holdout_data_dir=holdout_dir,
        training_data_dir=tmp_path / "training",
        metric_fns={"completed": lambda expected, actual: 1.0},
        check_regression=False,
        verbose=False,
    )

    assert report.summary["average_holdout_score"] == 0.5


def test_cross_validation_optimizer_error_retains_every_requested_example(
    tmp_path, monkeypatch
):
    class FailingOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, *args, **kwargs):
            raise RuntimeError("optimizer failed")

    monkeypatch.setattr(verification_module, "BootstrapFewShot", FailingOptimizer)
    suite = VerificationSuite(
        runner=MagicMock(), storage=_storage_with_prompt(tmp_path)
    )
    training = [
        TrainingExample(f"input-{index}", f"expected-{index}")
        for index in range(4)
    ]

    try:
        result = suite.run_cross_validation(
            "test-agent",
            training,
            lambda expected, actual: 1.0,
            k=2,
            verbose=False,
        )
    except RuntimeError:
        result = None

    assert result is not None
    assert _status_value(result) == "ERROR"
    assert result.total_examples == 4
    assert len(result.evaluations) == 4
    assert all(_status_value(item) == "ERROR" for item in result.evaluations)


def test_cross_validation_prompt_error_retains_every_requested_example(tmp_path):
    storage = MagicMock()
    storage.load_optimized_prompt.side_effect = ValueError("malformed prompt")
    suite = VerificationSuite(runner=MagicMock(), storage=storage)
    training = [TrainingExample(f"input-{index}", "expected") for index in range(4)]

    try:
        result = suite.run_cross_validation(
            "test-agent",
            training,
            lambda expected, actual: 1.0,
            k=2,
            verbose=False,
        )
    except ValueError:
        result = None

    assert result is not None
    assert _status_value(result) == "ERROR"
    assert len(result.evaluations) == 4


def test_first_candidate_preflight_refuses_failed_evaluations(tmp_path):
    storage = DemoStorage(storage_dir=str(tmp_path / "storage"))
    runner = MagicMock()
    runner.run.return_value = FakeRunResult(
        output="", success=False, error="model outage"
    )
    candidate = OptimizedPrompt(
        base_prompt="candidate",
        demos=[Demo("input", "output", 0.8)],
        optimization_date="2026-08-23T00:00:00Z",
        metric_name="metric",
        threshold=0.7,
        avg_score=0.8,
    )
    holdout = [
        TrainingExample("one", "expected"),
        TrainingExample("two", "expected"),
    ]

    result = pre_flight_holdout_check(
        "test-agent",
        candidate,
        holdout,
        lambda expected, actual: 1.0,
        runner,
        storage,
        verbose=False,
    )
    should_deploy, new_score, existing_score = result

    assert _status_value(result) == "ERROR"
    assert should_deploy is False
    assert new_score == 0.0
    assert existing_score is None
    assert len(result.new_evaluations) == 2
    assert all(_status_value(item) == "ERROR" for item in result.new_evaluations)


def _load_verify_cli_module():
    script = Path(__file__).parents[1] / "scripts" / "verify_optimizations.py"
    spec = importlib.util.spec_from_file_location("verify_optimizations_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_exits_nonzero_when_regression_detected(tmp_path, monkeypatch):
    cli = _load_verify_cli_module()
    holdout_path = tmp_path / "holdout.jsonl"
    holdout_path.write_text('{"input":"x","expected":"y"}\n')
    suite = MagicMock()
    suite.load_holdout_data.return_value = [object()]
    suite.run_holdout_evaluation.return_value = SimpleNamespace(
        status="PASS", passed=True, holdout_score=0.8
    )
    suite.check_regression.return_value = SimpleNamespace(
        status="FAIL", regressed=True
    )
    monkeypatch.setattr(cli, "VerificationSuite", lambda *args, **kwargs: suite)
    monkeypatch.setattr(cli, "ClaudeRunner", MagicMock)
    monkeypatch.setattr(cli, "DemoStorage", MagicMock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_optimizations.py",
            "--agent",
            "code-reviewer",
            "--holdout",
            str(holdout_path),
            "--check-regression",
            "--quiet",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1


def test_cli_missing_holdout_is_nonzero_by_default(tmp_path, monkeypatch):
    cli = _load_verify_cli_module()
    monkeypatch.setattr(cli, "ClaudeRunner", MagicMock)
    monkeypatch.setattr(cli, "DemoStorage", MagicMock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_optimizations.py",
            "--agent",
            "code-reviewer",
            "--holdout",
            str(tmp_path / "missing.jsonl"),
            "--quiet",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1


def test_cli_allow_missing_is_explicit_advisory_mode(tmp_path, monkeypatch):
    cli = _load_verify_cli_module()
    monkeypatch.setattr(cli, "ClaudeRunner", MagicMock)
    monkeypatch.setattr(cli, "DemoStorage", MagicMock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_optimizations.py",
            "--agent",
            "code-reviewer",
            "--holdout",
            str(tmp_path / "missing.jsonl"),
            "--allow-missing",
            "--quiet",
        ],
    )

    cli.main()


def test_cli_exits_nonzero_for_cross_validation_error(tmp_path, monkeypatch):
    cli = _load_verify_cli_module()
    holdout_path = tmp_path / "holdout.jsonl"
    training_path = tmp_path / "training.jsonl"
    holdout_path.write_text('{"input":"x","expected":"y"}\n')
    training_path.write_text('{"input":"x","expected":"y"}\n')
    suite = MagicMock()
    suite.load_holdout_data.return_value = [object()]
    suite.run_holdout_evaluation.return_value = SimpleNamespace(
        status="PASS", passed=True, holdout_score=0.8
    )
    suite.run_cross_validation.return_value = SimpleNamespace(status="ERROR")
    monkeypatch.setattr(cli, "VerificationSuite", lambda *args, **kwargs: suite)
    monkeypatch.setattr(cli, "ClaudeRunner", MagicMock)
    monkeypatch.setattr(cli, "DemoStorage", MagicMock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_optimizations.py",
            "--agent",
            "code-reviewer",
            "--holdout",
            str(holdout_path),
            "--cross-validate",
            "--training-data",
            str(training_path),
            "--quiet",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1


@pytest.mark.parametrize(
    ("holdout_status", "regression_status"),
    [("NOT_RUN", None), ("PASS", "ERROR")],
)
def test_cli_allow_missing_does_not_waive_configuration_or_regression_errors(
    tmp_path, monkeypatch, holdout_status, regression_status
):
    cli = _load_verify_cli_module()
    holdout_path = tmp_path / "holdout.jsonl"
    holdout_path.write_text('{"input":"x","expected":"y"}\n')
    suite = MagicMock()
    suite.load_holdout_data.return_value = [object()]
    suite.run_holdout_evaluation.return_value = SimpleNamespace(
        status=holdout_status,
        passed=holdout_status == "PASS",
        holdout_score=0.8,
        error_category="configuration",
    )
    suite.check_regression.return_value = SimpleNamespace(
        status=regression_status,
        regressed=False,
    )
    monkeypatch.setattr(cli, "VerificationSuite", lambda *args, **kwargs: suite)
    monkeypatch.setattr(cli, "ClaudeRunner", MagicMock)
    monkeypatch.setattr(cli, "DemoStorage", MagicMock)
    arguments = [
        "verify_optimizations.py",
        "--agent",
        "code-reviewer",
        "--holdout",
        str(holdout_path),
        "--allow-missing",
        "--quiet",
    ]
    if regression_status is not None:
        arguments.append("--check-regression")
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
