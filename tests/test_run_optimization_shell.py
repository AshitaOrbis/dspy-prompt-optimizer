"""Shell-level orchestration gate regression tests."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def _always_success_python(fake_bin: Path) -> None:
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n")
    fake_python.chmod(0o755)


def test_validation_failure_is_failed_not_completed_and_exits_nonzero(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "code-reviews.jsonl").write_text('{"input":"i","expected":"e"}\n')
    (datasets / "code-reviews-holdout.jsonl").write_text('{"input":"h","expected":"e"}\n')
    output = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"$1\" == *verify_optimizations.py ]]; then exit 7; fi\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$arg\" == --holdout-gate ]]; then exit 7; fi\n"
        "done\n"
        "exit 0\n"
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
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
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    status = json.loads((output / "status.json").read_text())
    assert status["completed"] == []
    assert status["failed"] == ["code-reviewer"]
    assert status["current"] == "failed"
    assert "--holdout-gate" in result.stdout
    assert "OPTIMIZATION COMPLETE" not in result.stdout


def test_missing_holdout_is_unverified_without_explicit_override(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "code-reviews.jsonl").write_text('{"input":"i","expected":"e"}\n')
    output = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _always_success_python(fake_bin)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
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
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    status = json.loads((output / "status.json").read_text())
    assert status["completed"] == []
    assert status["failed"] == []
    assert status["unverified"] == ["code-reviewer"]


def test_allow_unverified_cannot_override_required_holdout(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "code-reviews.jsonl").write_text('{"input":"i","expected":"e"}\n')
    output = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _always_success_python(fake_bin)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
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

    assert result.returncode == 2
    status = json.loads((output / "status.json").read_text())
    assert status["current"] == "unverified"
    assert status["completed"] == []
    assert status["failed"] == []
    assert status["unverified"] == ["code-reviewer"]
    assert "TERMINAL_SIGNAL: OPTIMIZATION_COMPLETE" not in result.stdout
