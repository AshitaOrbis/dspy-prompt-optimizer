"""Tests for pre-flight holdout gate functionality."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer.verification import pre_flight_holdout_check
from prompt_optimizer.storage import DemoStorage, OptimizedPrompt, Demo
from prompt_optimizer.bootstrap import TrainingExample


@dataclass
class FakeRunResult:
    output: str
    success: bool
    error: str = ""
    raw_output: str = ""


def make_optimized_prompt(demos_score=0.8):
    """Create a test OptimizedPrompt."""
    return OptimizedPrompt(
        base_prompt="You are a reviewer.",
        demos=[Demo(input_text="test input", output_text="test output", score=demos_score)],
        optimization_date="2026-04-14",
        metric_name="test_metric",
        threshold=0.5,
        avg_score=demos_score,
    )


def make_holdout_data():
    """Create simple holdout examples."""
    return [
        TrainingExample(input_text="holdout input 1", expected_output="expected 1"),
        TrainingExample(input_text="holdout input 2", expected_output="expected 2"),
    ]


def test_gate_passes_when_new_is_better():
    """Gate should return should_deploy=True when new score beats existing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DemoStorage(storage_dir=tmpdir)

        # Save an existing "old" optimization with low score
        old_prompt = make_optimized_prompt(demos_score=0.4)
        storage.save_optimized_prompt("test-agent", old_prompt)

        # New optimization with higher score
        new_prompt = make_optimized_prompt(demos_score=0.8)

        # Mock runner that returns output scoring 0.7
        runner = MagicMock()
        runner.run.return_value = FakeRunResult(output="expected 1", success=True)

        # Metric that returns 0.7 for any match
        def mock_metric(expected, actual):
            return 0.7

        holdout = make_holdout_data()

        should_deploy, new_score, existing_score = pre_flight_holdout_check(
            agent_name="test-agent",
            new_optimized=new_prompt,
            holdout_data=holdout,
            metric_fn=mock_metric,
            runner=runner,
            storage=storage,
            verbose=False,
        )

        assert should_deploy is True
        assert new_score == 0.7
        # existing_score is also 0.7 because same mock runner is used
        assert existing_score == 0.7


def test_gate_blocks_when_new_is_worse():
    """Gate should return should_deploy=False when new score is significantly worse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DemoStorage(storage_dir=tmpdir)

        # Save an existing optimization
        old_prompt = make_optimized_prompt(demos_score=0.8)
        storage.save_optimized_prompt("test-agent", old_prompt)

        new_prompt = make_optimized_prompt(demos_score=0.3)

        call_count = 0

        def side_effect(prompt):
            nonlocal call_count
            call_count += 1
            # First 2 calls (new optimization) return low score output
            # Next 2 calls (existing optimization) return high score output
            if call_count <= 2:
                return FakeRunResult(output="bad output", success=True)
            else:
                return FakeRunResult(output="expected 1", success=True)

        runner = MagicMock()
        runner.run.side_effect = side_effect

        def mock_metric(expected, actual):
            if actual == "bad output":
                return 0.2
            return 0.8

        holdout = make_holdout_data()

        should_deploy, new_score, existing_score = pre_flight_holdout_check(
            agent_name="test-agent",
            new_optimized=new_prompt,
            holdout_data=holdout,
            metric_fn=mock_metric,
            runner=runner,
            storage=storage,
            min_improvement=0.0,
            verbose=False,
        )

        assert should_deploy is False
        assert new_score == 0.2
        assert existing_score == 0.8


def test_gate_auto_passes_first_optimization():
    """Gate should always deploy when there's no existing optimization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DemoStorage(storage_dir=tmpdir)
        # Don't save any existing optimization

        new_prompt = make_optimized_prompt(demos_score=0.5)

        runner = MagicMock()
        runner.run.return_value = FakeRunResult(output="output", success=True)

        def mock_metric(expected, actual):
            return 0.5

        holdout = make_holdout_data()

        should_deploy, new_score, existing_score = pre_flight_holdout_check(
            agent_name="test-agent",
            new_optimized=new_prompt,
            holdout_data=holdout,
            metric_fn=mock_metric,
            runner=runner,
            storage=storage,
            verbose=False,
        )

        assert should_deploy is True
        assert new_score == 0.5
        assert existing_score is None


if __name__ == "__main__":
    test_gate_passes_when_new_is_better()
    print("PASS: test_gate_passes_when_new_is_better")

    test_gate_blocks_when_new_is_worse()
    print("PASS: test_gate_blocks_when_new_is_worse")

    test_gate_auto_passes_first_optimization()
    print("PASS: test_gate_auto_passes_first_optimization")

    print("\nAll holdout gate tests passed!")
