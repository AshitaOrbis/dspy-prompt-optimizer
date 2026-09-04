"""Regression tests for the shared hardened model-runner subprocess boundary."""

import json
import stat
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from prompt_optimizer import claude_runner
from prompt_optimizer.claude_runner import ClaudeRunner
from prompt_optimizer.model_runners import CodexModelRunner, GeminiModelRunner


PROMPT = "SENSITIVE_PROMPT_PAYLOAD"
CONTEXT = "SENSITIVE_CONTEXT_PAYLOAD"


def _stdout_for(runner, output="ok"):
    if isinstance(runner, GeminiModelRunner):
        return json.dumps({"type": "result", "result": output}) + "\n"
    return output


def _capture_subprocess(monkeypatch, *, returncode=0, output="ok", stderr=""):
    calls = []

    def fake_capture(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return claude_runner._ProcessCapture(
            returncode=returncode,
            stdout=output.encode("utf-8"),
            stderr=stderr.encode("utf-8"),
        )

    monkeypatch.setattr(claude_runner, "_capture_process", fake_capture)
    return calls


@pytest.mark.parametrize(
    "runner",
    [ClaudeRunner(), CodexModelRunner(), GeminiModelRunner()],
    ids=["claude", "codex", "agy"],
)
def test_user_prompt_and_context_are_absent_from_subprocess_argv(monkeypatch, runner):
    calls = _capture_subprocess(monkeypatch, output=_stdout_for(runner))

    runner.run(PROMPT, CONTEXT)

    argv = calls[0][0]
    assert all(PROMPT not in arg and CONTEXT not in arg for arg in argv)


def test_claude_system_prompt_uses_private_file_not_argv(monkeypatch):
    system_prompt = "SENSITIVE_SYSTEM_PROMPT"
    observation = {}

    def fake_capture(cmd, **kwargs):
        observation["argv"] = cmd
        file_index = cmd.index("--system-prompt-file") + 1
        path = Path(cmd[file_index])
        observation["path"] = path
        observation["mode"] = stat.S_IMODE(path.stat().st_mode)
        observation["content"] = path.read_text(encoding="utf-8")
        return claude_runner._ProcessCapture(returncode=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(claude_runner, "_capture_process", fake_capture)
    result = ClaudeRunner().run_with_system(PROMPT, system_prompt)

    assert result.success
    assert system_prompt not in observation["argv"]
    assert observation["content"] == system_prompt
    assert observation["mode"] == 0o600
    assert not observation["path"].exists()


@pytest.mark.parametrize(
    "runner",
    [ClaudeRunner(), CodexModelRunner(), GeminiModelRunner()],
    ids=["claude", "codex", "agy"],
)
def test_scrubbed_environment_is_an_explicit_allowlist(monkeypatch, runner):
    monkeypatch.setenv("PROMPT_OPTIMIZER_SCRUB_ENV", "1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "AMBIENT_SECRET_VALUE")
    monkeypatch.setenv("PROMPT_OPTIMIZER_ENV_PASSTHROUGH", "RUNNER_SCOPED_TOKEN")
    monkeypatch.setenv("RUNNER_SCOPED_TOKEN", "allowed-scoped-value")
    calls = _capture_subprocess(monkeypatch, output=_stdout_for(runner))

    runner.run(PROMPT)

    child_env = calls[0][1]["env"]
    expected_names = claude_runner.SUBPROCESS_ENV_ALLOWLIST | {"RUNNER_SCOPED_TOKEN"}
    assert set(child_env) <= expected_names
    assert "AWS_SECRET_ACCESS_KEY" not in child_env
    assert child_env["RUNNER_SCOPED_TOKEN"] == "allowed-scoped-value"


@pytest.mark.parametrize(
    "runner",
    [ClaudeRunner(), CodexModelRunner(), GeminiModelRunner()],
    ids=["claude", "codex", "agy"],
)
def test_permission_bypass_is_disabled_by_default(monkeypatch, runner):
    monkeypatch.delenv("PROMPT_OPTIMIZER_SKIP_PERMISSIONS", raising=False)
    calls = _capture_subprocess(monkeypatch, output=_stdout_for(runner))

    runner.run(PROMPT)

    argv = calls[0][0]
    assert "--dangerously-skip-permissions" not in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "--approve-for-me" not in argv
    assert "--full-auto" not in argv


@pytest.mark.parametrize(
    ("runner", "expected_flag"),
    [
        (ClaudeRunner(), "--dangerously-skip-permissions"),
        (CodexModelRunner(), "--approve-for-me"),
        (GeminiModelRunner(), "--dangerously-skip-permissions"),
    ],
    ids=["claude", "codex", "agy"],
)
def test_permission_automation_requires_explicit_opt_in(
    monkeypatch, runner, expected_flag
):
    monkeypatch.setenv("PROMPT_OPTIMIZER_SKIP_PERMISSIONS", "1")
    calls = _capture_subprocess(monkeypatch, output=_stdout_for(runner))

    runner.run(PROMPT)

    assert expected_flag in calls[0][0]


@pytest.mark.parametrize("runner_cls", [ClaudeRunner, CodexModelRunner, GeminiModelRunner])
def test_timeout_rejects_non_positive_and_absurd_values_and_clamps_small_values(
    runner_cls,
):
    with pytest.raises(ValueError, match="positive"):
        runner_cls(timeout=0)
    with pytest.raises(ValueError, match="at most"):
        runner_cls(timeout=claude_runner.MAX_TIMEOUT + 1)
    assert runner_cls(timeout=1).timeout == claude_runner.MIN_TIMEOUT


@pytest.mark.parametrize(
    "runner",
    [ClaudeRunner(), CodexModelRunner(), GeminiModelRunner()],
    ids=["claude", "codex", "agy"],
)
def test_captured_output_is_rejected_at_shared_byte_cap(monkeypatch, runner):
    monkeypatch.setattr(claude_runner, "MAX_OUTPUT_BYTES", 8)
    calls = _capture_subprocess(monkeypatch, output=_stdout_for(runner, "x" * 20))

    result = runner.run(PROMPT)

    assert calls
    assert not result.success
    assert "stdout truncated at 8 bytes" in result.error


def test_runaway_stdout_is_bounded_during_capture_not_afterward(monkeypatch):
    cap = 64 * 1024
    total_output = 32 * 1024 * 1024
    monkeypatch.setattr(claude_runner, "MAX_OUTPUT_BYTES", cap)
    child = (
        "import os\n"
        "chunk = b'x' * 4096\n"
        f"for _ in range({total_output // 4096}):\n"
        "    os.write(1, chunk)\n"
    )

    tracemalloc.start()
    tracemalloc.reset_peak()
    result = claude_runner.run_hardened_subprocess(
        [sys.executable, "-c", child],
        stdin_text="",
        timeout=5,
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # A post-hoc size check peaks near total_output. Streaming capture should
    # remain O(cap), with generous room for thread/process bookkeeping.
    assert peak_bytes < 4 * 1024 * 1024
    assert not result.success
    assert len(result.output.encode("utf-8")) <= cap
    assert len((result.raw_output or "").encode("utf-8")) <= cap
    assert f"stdout truncated at {cap} bytes" in result.error


def test_runaway_stderr_is_bounded_and_truncation_is_visible(monkeypatch):
    cap = 32 * 1024
    total_output = 4 * 1024 * 1024
    monkeypatch.setattr(claude_runner, "MAX_OUTPUT_BYTES", cap)
    child = (
        "import os\n"
        "chunk = b'e' * 4096\n"
        f"for _ in range({total_output // 4096}):\n"
        "    os.write(2, chunk)\n"
    )

    result = claude_runner.run_hardened_subprocess(
        [sys.executable, "-c", child],
        stdin_text="",
        timeout=5,
    )

    assert not result.success
    assert len(result.output.encode("utf-8")) <= cap
    assert f"stderr truncated at {cap} bytes" in result.error
    assert len(result.error.encode("utf-8")) <= cap + 1024


def test_output_cap_kills_child_that_ignores_termination(monkeypatch):
    cap = 32 * 1024
    monkeypatch.setattr(claude_runner, "MAX_OUTPUT_BYTES", cap)
    child = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"os.write(1, b'x' * ({cap} + 4096))\n"
        "time.sleep(30)\n"
    )

    started = time.monotonic()
    result = claude_runner.run_hardened_subprocess(
        [sys.executable, "-c", child],
        stdin_text="",
        timeout=5,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 2
    assert not result.success
    assert f"stdout truncated at {cap} bytes" in result.error


def test_timeout_still_terminates_a_quiet_child(monkeypatch):
    monkeypatch.setattr(claude_runner, "MIN_TIMEOUT", 1)
    started = time.monotonic()
    result = claude_runner.run_hardened_subprocess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin_text="",
        timeout=1,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 3
    assert not result.success
    assert result.error == "Timeout after 1 seconds"


@pytest.mark.parametrize(
    "runner",
    [ClaudeRunner(), CodexModelRunner(), GeminiModelRunner()],
    ids=["claude", "codex", "agy"],
)
def test_errors_redact_prompt_context_and_environment_values(monkeypatch, runner):
    env_secret = "AMBIENT_ENV_SECRET_VALUE"
    monkeypatch.setenv("CUSTOM_AMBIENT_VALUE", env_secret)
    stderr = f"failure: {CONTEXT} {PROMPT} token=BACKEND_SECRET {env_secret}"
    calls = _capture_subprocess(
        monkeypatch,
        returncode=2,
        output="",
        stderr=stderr,
    )

    result = runner.run(PROMPT, CONTEXT)

    assert calls
    assert not result.success
    assert PROMPT not in result.error
    assert CONTEXT not in result.error
    assert "BACKEND_SECRET" not in result.error
    assert env_secret not in result.error
    assert "<REDACTED>" in result.error


def test_gemini_mocked_schema_uses_stream_json_stdin_and_terminal_result(monkeypatch):
    """Assert the provisional repo schema, not the unverified live agy contract."""
    calls = _capture_subprocess(
        monkeypatch,
        output=json.dumps({"type": "result", "result": "structured answer"}) + "\n",
    )
    runner = GeminiModelRunner()

    result = runner.run(PROMPT, CONTEXT)

    argv, kwargs = calls[0]
    assert "--input-format=stream-json" in argv
    assert "--output-format=stream-json" in argv
    assert "-p" not in argv
    input_event = json.loads(kwargs["input"])
    assert input_event == {
        "type": "user",
        "message": f"{CONTEXT}\n\n{PROMPT}",
    }
    assert result.success
    assert result.output == "structured answer"
