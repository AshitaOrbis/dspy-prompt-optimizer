"""
Claude Code CLI wrapper for prompt optimization.

Provides a clean interface for running prompts through Claude Code CLI
and parsing the output for use in optimization algorithms.
"""

import os
import shutil
import subprocess
import json
import re
from dataclasses import dataclass
from typing import Optional


# Default timeout in seconds (can be overridden via env var)
DEFAULT_TIMEOUT = 180

# Clamp the env-configurable timeout to a sane range so a poisoned environment
# cannot pin a subprocess open indefinitely (or set a uselessly tiny value).
MIN_TIMEOUT = 5
MAX_TIMEOUT = 1800

# Hard cap on captured stdout/stderr per run. A runaway / adversarial model
# response could otherwise buffer unbounded output in memory.
MAX_OUTPUT_BYTES = int(os.environ.get("PROMPT_OPTIMIZER_MAX_OUTPUT_BYTES", 5_000_000))

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


def _build_subprocess_env() -> Optional[dict]:
    """
    Build the environment for the claude subprocess.

    By default the subprocess inherits the parent environment (claude needs its
    auth/config, typically under $HOME). When PROMPT_OPTIMIZER_SCRUB_ENV is set,
    pass only an allowlisted, minimal environment so that ambient secrets
    (cloud tokens, CI variables, .env exports) are not exposed to a model that
    may attempt tool-driven exfiltration. Returns None to mean "inherit".
    """
    if os.environ.get("PROMPT_OPTIMIZER_SCRUB_ENV", "").lower() not in ("1", "true", "yes"):
        # Inherit the parent environment, minus CLAUDECODE: its presence makes
        # a nested `claude -p` invocation fail with a nested-session error.
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        return env

    # Minimal allowlist needed for claude to locate config/auth and run.
    allowlist = {
        "HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR",
        "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    }
    # Allow explicit pass-through of named vars (e.g. a scoped token) if the
    # operator opts in, but never blanket-copy the environment.
    extra = os.environ.get("PROMPT_OPTIMIZER_ENV_PASSTHROUGH", "")
    for name in (n.strip() for n in extra.split(",") if n.strip()):
        allowlist.add(name)

    env = {k: v for k, v in os.environ.items() if k in allowlist}
    # Guarantee a usable PATH even if the parent had none.
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return env


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
        # Clamp to a sane range (defends against a hostile env pinning a huge or
        # tiny timeout).
        self.timeout = max(MIN_TIMEOUT, min(int(timeout), MAX_TIMEOUT))
        self.working_dir = working_dir

        # Resolve the claude executable once, at construction, to avoid a
        # PATH-hijack where a project-local or world-writable './claude' is
        # executed instead of the real CLI. Fall back to the bare name if not
        # found (the subprocess call will then surface a clear error).
        self.claude_bin = _find_claude_binary()

        # Whether to pass --dangerously-skip-permissions. This bypass is
        # INHERENT to unattended/non-interactive optimization (the CLI would
        # otherwise block on a permission prompt). It is documented in SECURITY.md.
        # Operators who run only trusted datasets in a sandbox can disable it by
        # setting PROMPT_OPTIMIZER_SKIP_PERMISSIONS=0, at the cost of the run
        # hanging if the model attempts a gated tool call.
        self.skip_permissions = os.environ.get(
            "PROMPT_OPTIMIZER_SKIP_PERMISSIONS", "1"
        ).lower() not in ("0", "false", "no")

        self._env = _build_subprocess_env()

    def _base_cmd(self) -> list:
        """Build the leading argv shared by run() and run_with_system()."""
        cmd = [self.claude_bin, "--print", "--model", self.model]
        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        return cmd

    def _exec(self, cmd: list, stdin_text: str) -> "RunResult":
        """
        Execute a claude argv with the prompt supplied on STDIN.

        Passing the prompt via stdin (rather than as a command-line argument)
        keeps potentially-sensitive prompt/context text out of the process
        argument list, which is world-readable via /proc on many systems.
        Output is size-capped and secret-redacted before being returned.
        """
        try:
            result = subprocess.run(
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
                env=self._env,
            )
        except subprocess.TimeoutExpired:
            return RunResult(output="", success=False, error=f"Timeout after {self.timeout} seconds")
        except Exception as e:
            return RunResult(output="", success=False, error=redact_secrets(str(e)))

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if len(stdout) > MAX_OUTPUT_BYTES or len(stderr) > MAX_OUTPUT_BYTES:
            return RunResult(
                output="",
                success=False,
                error=f"Output exceeded {MAX_OUTPUT_BYTES} bytes; treating run as failed",
            )

        if result.returncode != 0:
            return RunResult(
                output="",
                success=False,
                error=redact_secrets(stderr) or f"Exit code: {result.returncode}",
                raw_output=stdout,
            )

        return RunResult(
            output=self._clean_output(stdout),
            success=True,
            raw_output=stdout,
        )

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
        return self._exec(self._base_cmd(), full_prompt)

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
        cmd = self._base_cmd()
        cmd += ["--system-prompt", system_prompt]
        # Prompt delivered on STDIN (see _exec) rather than as an argv element.
        return self._exec(cmd, full_prompt)

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
