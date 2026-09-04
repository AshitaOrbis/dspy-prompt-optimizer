"""Execute the README algorithm examples to catch public API drift."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import prompt_optimizer
from prompt_optimizer.bootstrap import TrainingExample
from prompt_optimizer.claude_runner import RunResult


README_PATH = Path(__file__).parent.parent / "README.md"
ALGORITHMS = README_PATH.read_text(encoding="utf-8").split("## Algorithms", 1)[1].split(
    "## Metrics", 1
)[0]
PYTHON_EXAMPLES = re.findall(r"```python\n(.*?)```", ALGORITHMS, flags=re.DOTALL)


class StubRunner:
    """Offline runner used only to execute documentation examples."""

    def __init__(self, **_: object) -> None:
        pass

    def run(self, _: str) -> RunResult:
        return RunResult(success=True, output="Expected output")


@pytest.mark.parametrize("example", PYTHON_EXAMPLES, ids=["bootstrap", "copro", "iterative"])
def test_readme_algorithm_example_executes(
    example: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert len(PYTHON_EXAMPLES) == 3
    monkeypatch.setenv("PROMPT_OPTIMIZER_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(prompt_optimizer, "ClaudeRunner", StubRunner)
    namespace = {
        "ClaudeRunner": StubRunner,
        "examples": [
            TrainingExample(
                input_text="Example input",
                expected_output="Expected output",
            )
        ],
        "your_metric": lambda expected, actual: float(expected == actual),
    }

    exec(compile(example, str(README_PATH), "exec"), namespace)
