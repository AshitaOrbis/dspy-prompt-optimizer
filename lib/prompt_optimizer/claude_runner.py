"""
Claude Code CLI wrapper for prompt optimization.

Provides a clean interface for running prompts through Claude Code CLI
and parsing the output for use in optimization algorithms.
"""

import os
import shutil
import subprocess
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence


# Default timeout in seconds (can be overridden via env var)
DEFAULT_TIMEOUT = 180

# Clamp the env-configurable timeout to a sane range so a poisoned environment
# cannot pin a subprocess open indefinitely (or set a uselessly tiny value).
MIN_TIMEOUT = 5
MAX_TIMEOUT = 1800

# Hard cap on captured stdout/stderr per run. A runaway / adversarial model
# response could otherwise be retained or processed without a bound.
MAX_OUTPUT_BYTES = int(os.environ.get("PROMPT_OPTIMIZER_MAX_OUTPUT_BYTES", 5_000_000))

# Pipe reads stay small and unbuffered so the parent never accumulates a child
# stream before applying MAX_OUTPUT_BYTES. Termination gets a short grace period
# before the child is killed.
_OUTPUT_READ_CHUNK_BYTES = 64 * 1024
_PROCESS_POLL_SECONDS = 0.02
_TERMINATE_GRACE_SECONDS = 0.25
_KILL_GRACE_SECONDS = 1.0

# The only ambient variables admitted when subprocess environment scrubbing is
# enabled. Backend-specific callers all use this same policy.
SUBPROCESS_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TERM",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
    }
)

_TRUE_VALUES = frozenset({"1", "true", "yes"})

# Patterns used to redact obvious secrets before any value is logged or stored.
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|bearer)\b\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]


def redact_secrets(text: Optional[str]) -> Optional[str]:
    """Redact common secret patterns from a string before logging/storage."""
    if not text:
        return text
    redacted = text
    for pat in _SECRET_PATTERNS:
        redacted = pat.sub("<REDACTED>", redacted)
    return redacted


def validate_timeout(timeout: object) -> int:
    """Validate a subprocess timeout and clamp small positive values."""
    if isinstance(timeout, bool):
        raise ValueError("timeout must be a positive integer")
    try:
        parsed = int(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a positive integer") from exc
    if isinstance(timeout, float) and not timeout.is_integer():
        raise ValueError("timeout must be a positive integer")
    if parsed <= 0:
        raise ValueError("timeout must be positive")
    if parsed > MAX_TIMEOUT:
        raise ValueError(f"timeout must be at most {MAX_TIMEOUT} seconds")
    return max(MIN_TIMEOUT, parsed)


def permission_automation_enabled() -> bool:
    """Return whether unattended permission automation was explicitly enabled."""
    return os.environ.get("PROMPT_OPTIMIZER_SKIP_PERMISSIONS", "").lower() in _TRUE_VALUES


def _build_subprocess_env() -> dict:
    """
    Build the environment for a model subprocess.

    By default the subprocess inherits the parent environment (model CLIs need
    auth/config, typically under $HOME). When PROMPT_OPTIMIZER_SCRUB_ENV is set,
    pass only an allowlisted, minimal environment so that ambient secrets
    (cloud tokens, CI variables, .env exports) are not exposed to a model that
    may attempt tool-driven exfiltration.
    """
    if os.environ.get("PROMPT_OPTIMIZER_SCRUB_ENV", "").lower() not in ("1", "true", "yes"):
        # Inherit the parent environment, minus CLAUDECODE: its presence makes
        # a nested `claude -p` invocation fail with a nested-session error.
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        return env

    # Minimal allowlist needed for the CLIs to locate config/auth and run.
    allowlist = set(SUBPROCESS_ENV_ALLOWLIST)
    # Allow explicit pass-through of named vars (e.g. a scoped token) if the
    # operator opts in, but never blanket-copy the environment.
    extra = os.environ.get("PROMPT_OPTIMIZER_ENV_PASSTHROUGH", "")
    for name in (n.strip() for n in extra.split(",") if n.strip()):
        allowlist.add(name)

    env = {k: v for k, v in os.environ.items() if k in allowlist}
    # Guarantee a usable PATH even if the parent had none.
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return env


def _redact_error_content(
    text: Optional[str],
    *,
    sensitive_values: Sequence[str],
    child_env: dict,
) -> Optional[str]:
    """Remove prompt and ambient-environment values from backend errors."""
    if not text:
        return text

    redacted = text
    values = {value for value in sensitive_values if value}
    values.update(value for value in child_env.values() if value)
    for value in sorted(values, key=len, reverse=True):
        redacted = redacted.replace(value, "<REDACTED>")
    return redact_secrets(redacted)


@contextmanager
def private_prompt_file(text: str) -> Iterator[str]:
    """Yield a mode-0600 temporary file containing sensitive prompt text."""
    fd, path = tempfile.mkstemp(prefix="prompt-optimizer-", suffix=".txt")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as prompt_file:
            prompt_file.write(text)
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _find_claude_binary() -> str:
    """Find the claude binary, checking PATH and common locations."""
    # Try PATH first
    found = shutil.which("claude")
    if found:
        return found

    # Check common nvm/node locations
    home = os.path.expanduser("~")
    for candidate in [
        os.path.join(home, ".nvm/versions/node", d, "bin/claude")
        for d in sorted(os.listdir(os.path.join(home, ".nvm/versions/node")), reverse=True)
        if os.path.isdir(os.path.join(home, ".nvm/versions/node", d))
    ] if os.path.isdir(os.path.join(home, ".nvm/versions/node")) else []:
        if os.path.isfile(candidate):
            return candidate

    # Check npm global
    npm_global = os.path.join(home, ".npm-global/bin/claude")
    if os.path.isfile(npm_global):
        return npm_global

    # Fallback to bare name (let subprocess handle the error)
    return "claude"


# Resolve claude binary path once at import time
CLAUDE_BINARY = _find_claude_binary()


def estimate_timeout(input_text: str, base_timeout: int = 180) -> int:
    """
    Estimate timeout based on input complexity.

    Longer inputs typically require more processing time. This function
    provides complexity-aware dynamic timeouts.

    Args:
        input_text: The input text to process
        base_timeout: Base timeout in seconds (default: 180)

    Returns:
        Estimated timeout in seconds
    """
    # Estimate tokens by word count (rough approximation)
    tokens = len(input_text.split())

    if tokens > 1000:
        return base_timeout * 2  # Double for very long inputs
    elif tokens > 500:
        return int(base_timeout * 1.5)  # 1.5x for medium inputs
    return base_timeout


@dataclass
class RunResult:
    """Result from a Claude Code CLI run."""
    output: str
    success: bool
    error: Optional[str] = None
    raw_output: Optional[str] = None


@dataclass
class _StreamCapture:
    """A single pipe's retained bytes and whether input crossed its cap."""

    data: bytearray = field(default_factory=bytearray)
    truncated: bool = False


@dataclass
class _ProcessCapture:
    """Internal bounded result from a child process."""

    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False


def _terminate_then_kill(process: subprocess.Popen) -> None:
    """Stop a child promptly, escalating when it ignores termination."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        # A direct child should be reaped after SIGKILL. Surface the failure to
        # the caller rather than waiting without a bound.
        raise RuntimeError("model subprocess did not exit after kill")


def _read_bounded_stream(
    stream,
    capture: _StreamCapture,
    limit: int,
    limit_reached: threading.Event,
) -> None:
    """Drain one unbuffered pipe while retaining at most ``limit`` bytes."""
    while True:
        remaining = limit - len(capture.data)
        # The one-byte probe distinguishes an exact-limit EOF from truncation;
        # it is never retained, so the capture itself remains bounded.
        read_size = min(_OUTPUT_READ_CHUNK_BYTES, max(remaining + 1, 1))
        chunk = stream.read(read_size)
        if not chunk:
            return
        if remaining > 0:
            capture.data.extend(chunk[:remaining])
        if len(chunk) > remaining:
            capture.truncated = True
            limit_reached.set()
            return


def _write_stdin(stream, payload: bytes) -> None:
    """Feed stdin without blocking concurrent stdout/stderr draining."""
    try:
        stream.write(payload)
    except (BrokenPipeError, OSError):
        # Matching subprocess.communicate(), an early child exit is reported by
        # its return code rather than by a BrokenPipe from the writer.
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _capture_process(
    cmd: Sequence[str],
    *,
    input: str,
    timeout: int,
    cwd: Optional[str],
    env: dict,
) -> _ProcessCapture:
    """Run a child with concurrent, byte-bounded stdout/stderr capture."""
    limit = max(int(MAX_OUTPUT_BYTES), 0)
    process = subprocess.Popen(
        list(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        bufsize=0,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    limit_reached = threading.Event()
    stdout_capture = _StreamCapture()
    stderr_capture = _StreamCapture()
    stdout_thread = threading.Thread(
        target=_read_bounded_stream,
        args=(process.stdout, stdout_capture, limit, limit_reached),
        name="prompt-optimizer-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_bounded_stream,
        args=(process.stderr, stderr_capture, limit, limit_reached),
        name="prompt-optimizer-stderr",
        daemon=True,
    )
    stdin_thread = threading.Thread(
        target=_write_stdin,
        args=(process.stdin, input.encode("utf-8")),
        name="prompt-optimizer-stdin",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    stdin_thread.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    while process.poll() is None:
        if limit_reached.is_set():
            _terminate_then_kill(process)
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            _terminate_then_kill(process)
            break
        limit_reached.wait(min(_PROCESS_POLL_SECONDS, remaining))

    # A child can exit between emitting the over-limit byte and the main thread
    # observing the event. Readers still finish and preserve the cap decision.
    stdout_thread.join(_KILL_GRACE_SECONDS)
    stderr_thread.join(_KILL_GRACE_SECONDS)
    stdin_thread.join(_KILL_GRACE_SECONDS)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        raise RuntimeError("model subprocess pipes did not close after exit")

    return _ProcessCapture(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=bytes(stdout_capture.data),
        stderr=bytes(stderr_capture.data),
        stdout_truncated=stdout_capture.truncated,
        stderr_truncated=stderr_capture.truncated,
        timed_out=timed_out,
    )


def _decode_captured_output(value: bytes) -> str:
    """Decode retained UTF-8 without expanding malformed input past its cap."""
    return value.decode("utf-8", errors="ignore")


def run_hardened_subprocess(
    cmd: Sequence[str],
    *,
    stdin_text: str,
    timeout: object,
    cwd: Optional[str] = None,
    sensitive_values: Sequence[str] = (),
) -> RunResult:
    """Execute one model CLI under the shared subprocess-safety policy."""
    checked_timeout = validate_timeout(timeout)
    child_env = _build_subprocess_env()

    protected_values = tuple(value for value in sensitive_values if value)
    if any(value == arg for value in protected_values for arg in cmd):
        return RunResult(
            output="",
            success=False,
            error="Refusing subprocess: sensitive prompt content found in argv",
        )

    try:
        result = _capture_process(
            list(cmd),
            input=stdin_text,
            timeout=checked_timeout,
            cwd=cwd,
            env=child_env,
        )
    except Exception as exc:
        return RunResult(
            output="",
            success=False,
            error=_redact_error_content(
                str(exc),
                sensitive_values=protected_values,
                child_env=child_env,
            ),
        )

    limit = max(int(MAX_OUTPUT_BYTES), 0)
    stdout_truncated = result.stdout_truncated or len(result.stdout) > limit
    stderr_truncated = result.stderr_truncated or len(result.stderr) > limit
    stdout = _decode_captured_output(result.stdout[:limit])
    stderr = _decode_captured_output(result.stderr[:limit])

    if stdout_truncated or stderr_truncated:
        truncated_streams = []
        if stdout_truncated:
            truncated_streams.append(f"stdout truncated at {limit} bytes")
        if stderr_truncated:
            truncated_streams.append(f"stderr truncated at {limit} bytes")
        error = "Output limit exceeded; " + " and ".join(truncated_streams)
        redacted_stderr = _redact_error_content(
            stderr,
            sensitive_values=protected_values,
            child_env=child_env,
        )
        if redacted_stderr:
            # Keep returned diagnostics bounded as well as the underlying pipe.
            error += "\n" + redacted_stderr.encode("utf-8")[:limit].decode(
                "utf-8", errors="ignore"
            )
        return RunResult(
            output=stdout,
            success=False,
            error=error,
            raw_output=stdout,
        )

    if result.timed_out:
        return RunResult(
            output=stdout,
            success=False,
            error=f"Timeout after {checked_timeout} seconds",
            raw_output=stdout,
        )

    if result.returncode != 0:
        return RunResult(
            output="",
            success=False,
            error=_redact_error_content(
                stderr,
                sensitive_values=protected_values,
                child_env=child_env,
            )
            or f"Exit code: {result.returncode}",
            raw_output=stdout,
        )

    return RunResult(output=stdout, success=True, raw_output=stdout)


class ClaudeRunner:
    """Wrapper for Claude Code CLI invocations."""

    def __init__(
        self,
        model: str = "sonnet",
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
    ):
        """
        Initialize the Claude runner.

        Args:
            model: Model to use (sonnet, opus, haiku)
            timeout: Timeout in seconds for CLI call. If None, uses
                     PROMPT_OPTIMIZER_TIMEOUT env var or DEFAULT_TIMEOUT (180s)
            working_dir: Working directory for Claude Code
        """
        self.model = model
        if timeout is None:
            timeout = int(os.environ.get("PROMPT_OPTIMIZER_TIMEOUT", DEFAULT_TIMEOUT))
        self.timeout = validate_timeout(timeout)
        self.working_dir = working_dir

        # Resolve the claude executable once, at construction, to avoid a
        # PATH-hijack where a project-local or world-writable './claude' is
        # executed instead of the real CLI. Fall back to the bare name if not
        # found (the subprocess call will then surface a clear error).
        self.claude_bin = _find_claude_binary()

    def _base_cmd(self) -> list:
        """Build the leading argv shared by run() and run_with_system()."""
        cmd = [self.claude_bin, "--print", "--model", self.model]
        if permission_automation_enabled():
            cmd.append("--dangerously-skip-permissions")
        return cmd

    def _exec(
        self,
        cmd: list,
        stdin_text: str,
        *,
        sensitive_values: Sequence[str],
    ) -> "RunResult":
        """
        Execute a claude argv with the prompt supplied on STDIN.

        Passing the prompt via stdin (rather than as a command-line argument)
        keeps potentially-sensitive prompt/context text out of the process
        argument list, which is world-readable via /proc on many systems.
        Output is size-capped and secret-redacted before being returned.
        """
        result = run_hardened_subprocess(
            cmd,
            stdin_text=stdin_text,
            timeout=self.timeout,
            cwd=self.working_dir,
            sensitive_values=sensitive_values,
        )
        if result.success:
            result.output = self._clean_output(result.output)
        return result

    def run(self, prompt: str, context: Optional[str] = None) -> RunResult:
        """
        Run a prompt through Claude Code CLI.

        Args:
            prompt: The prompt to send
            context: Optional additional context to prepend

        Returns:
            RunResult with output, success status, and any errors
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        # Prompt delivered on STDIN (see _exec) rather than as an argv element.
        return self._exec(
            self._base_cmd(),
            full_prompt,
            sensitive_values=tuple(
                value for value in (full_prompt, prompt, context) if value
            ),
        )

    def _clean_output(self, output: str) -> str:
        """Remove ANSI escape codes and clean up output."""
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned = ansi_escape.sub('', output)
        return cleaned.strip()

    def run_with_system(
        self,
        prompt: str,
        system_prompt: str,
        context: Optional[str] = None,
    ) -> RunResult:
        """
        Run with a specific system prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt to use
            context: Optional additional context

        Returns:
            RunResult with output
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        sensitive_values = tuple(
            value for value in (full_prompt, prompt, context, system_prompt) if value
        )
        with private_prompt_file(system_prompt) as system_prompt_path:
            cmd = self._base_cmd()
            cmd += ["--system-prompt-file", system_prompt_path]
            return self._exec(
                cmd,
                full_prompt,
                sensitive_values=sensitive_values,
            )

    def with_model(self, model: str) -> 'ClaudeRunner':
        """
        Return a new runner with a different model.

        Useful for tiered optimization (Haiku -> Sonnet -> Opus).

        Args:
            model: New model to use. Accepts either a CLI alias ("haiku",
                "sonnet", "opus") which tracks the current generation, or
                a pinned full model ID ("claude-opus-4-7", "claude-opus-4-6",
                etc.) for longitudinal studies where a specific generation
                is load-bearing. The ``claude -p --model`` flag accepts
                both forms, so the string is passed through as-is.

        Returns:
            New ClaudeRunner instance with the specified model
        """
        return ClaudeRunner(
            model=model,
            timeout=self.timeout,
            working_dir=self.working_dir,
        )
