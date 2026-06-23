"""
Model runners for multi-model prompt optimization.

Provides CodexModelRunner (GPT-5.4) and GeminiModelRunner alongside
the existing ClaudeRunner. All return RunResult for drop-in compatibility
with BootstrapFewShot.optimize().

CLI invocations:
  - codex exec: passes prompt via stdin (large prompts break as CLI args)
  - agy -p: passes the prompt via the -p flag with stdin closed (the standalone
    `gemini` CLI is deprecated — its personal OAuth tier is no longer eligible,
    so Gemini access now goes through the Antigravity `agy` agentic CLI)
  - claude --print: existing ClaudeRunner handles this
"""

import os
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

from .claude_runner import RunResult


class ModelRunner(ABC):
    """Abstract base class for model runners."""

    @abstractmethod
    def run(self, prompt: str, context: Optional[str] = None) -> RunResult:
        """Run a prompt and return the result."""
        ...


class CodexModelRunner(ModelRunner):
    """
    Runner for GPT-5.4 via Codex CLI.

    Uses `codex exec` with prompt piped via stdin (avoids CLI arg length limits).
    The exec subcommand runs the agent non-interactively.
    """

    def __init__(
        self,
        model: str = "gpt-5.4",
        timeout: int = 480,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.timeout = timeout
        self.retry_delay = retry_delay
        self._binary = self._find_codex_binary()

    @staticmethod
    def _find_codex_binary() -> str:
        """Find the codex binary."""
        found = shutil.which("codex")
        if found:
            return found

        home = os.path.expanduser("~")
        nvm_dir = os.path.join(home, ".nvm/versions/node")
        if os.path.isdir(nvm_dir):
            for d in sorted(os.listdir(nvm_dir), reverse=True):
                candidate = os.path.join(nvm_dir, d, "bin/codex")
                if os.path.isfile(candidate):
                    return candidate

        return "codex"

    @staticmethod
    def _get_configured_mcp_servers() -> list:
        """Read config.toml and return names of defined MCP servers."""
        config_path = os.path.expanduser("~/.codex/config.toml")
        if not os.path.isfile(config_path):
            return []
        servers = []
        try:
            with open(config_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("[mcp_servers.") and line.endswith("]"):
                        name = line[len("[mcp_servers."):-1]
                        if name and "." not in name:
                            servers.append(name)
        except OSError:
            pass
        return servers

    def run(self, prompt: str, context: Optional[str] = None) -> RunResult:
        """Run prompt through Codex CLI via stdin."""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        # Use stdin (-) for the prompt to avoid CLI arg length limits.
        # codex exec reads from stdin when prompt arg is "-" or omitted.
        # Disable all MCP servers to avoid startup overhead and token waste.
        # Without MCP: ~1.7K tokens, instant startup.
        # With MCP: ~11.5K tokens, ~10s startup per call.
        # Build command. Disable MCP servers that exist in config.toml to avoid
        # startup overhead and token waste (~10K tokens saved per call).
        # Only disable servers that are actually defined — Codex rejects
        # config overrides for non-existent servers.
        cmd = [
            self._binary,
            "exec",
            "--full-auto",
            "-c", f'model="{self.model}"',
        ]
        for server_name in self._get_configured_mcp_servers():
            cmd.extend(["-c", f"mcp_servers.{server_name}.enabled=false"])
        cmd.extend([
            "--ephemeral",  # Don't persist session files
            "-",  # Read prompt from stdin
        ])

        env = {k: v for k, v in os.environ.items()}

        try:
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )

            if result.returncode != 0:
                return RunResult(
                    output="",
                    success=False,
                    error=result.stderr or f"Exit code: {result.returncode}",
                    raw_output=result.stdout,
                )

            clean_output = self._clean_output(result.stdout)
            return RunResult(
                output=clean_output,
                success=True,
                raw_output=result.stdout,
            )

        except subprocess.TimeoutExpired:
            return RunResult(
                output="",
                success=False,
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            return RunResult(
                output="",
                success=False,
                error=str(e),
            )

    @staticmethod
    def _clean_output(output: str) -> str:
        """Remove ANSI codes and clean up."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned = ansi_escape.sub('', output)
        return cleaned.strip()


class GeminiModelRunner(ModelRunner):
    """
    Runner for Gemini 3.1 Pro via the Antigravity `agy` agentic CLI.

    The standalone `gemini` CLI is deprecated (its personal OAuth tier is no
    longer eligible), so Gemini access now goes through `agy`. Uses `agy -p`
    for non-interactive print mode with stdin closed (the prompt is passed via
    the -p flag; agy has no stdin-append mode).
    """

    def __init__(
        self,
        model: str = "gemini-3.1-pro",
        timeout: int = 480,
    ):
        # Default to gemini-3.1-pro. The old gemini-CLI runner defaulted to None
        # to dodge a capacity issue when explicitly requesting
        # gemini-3.1-pro-preview; that no longer applies on agy, which accepts
        # the stable "gemini-3.1-pro" id.
        self.model = model
        self.timeout = timeout
        self._binary = self._find_agy_binary()

    @staticmethod
    def _find_agy_binary() -> str:
        """Find the agy binary."""
        found = shutil.which("agy")
        if found:
            return found

        home = os.path.expanduser("~")
        candidate = os.path.join(home, ".local/bin/agy")
        if os.path.isfile(candidate):
            return candidate

        nvm_dir = os.path.join(home, ".nvm/versions/node")
        if os.path.isdir(nvm_dir):
            for d in sorted(os.listdir(nvm_dir), reverse=True):
                candidate = os.path.join(nvm_dir, d, "bin/agy")
                if os.path.isfile(candidate):
                    return candidate

        return "agy"

    def run(self, prompt: str, context: Optional[str] = None) -> RunResult:
        """Run prompt through the agy CLI.

        agy has no stdin-append mode, so the full prompt is passed via -p with
        stdin closed (input="") to keep the run non-interactive.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        # agy print mode: --dangerously-skip-permissions auto-approves any tool
        # calls so the run never blocks on a permission prompt. --print-timeout
        # governs agy's own wait; keep it just under our hard timeout.
        cmd = [
            self._binary,
            "--dangerously-skip-permissions",
            "--model", self.model,
            "--print-timeout", f"{max(self.timeout - 30, 30)}s",
            "-p", full_prompt,
        ]

        env = {k: v for k, v in os.environ.items()}

        try:
            result = subprocess.run(
                cmd,
                input="",  # close stdin so agy stays non-interactive
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )

            if result.returncode != 0:
                return RunResult(
                    output="",
                    success=False,
                    error=result.stderr or f"Exit code: {result.returncode}",
                    raw_output=result.stdout,
                )

            clean_output = self._clean_output(result.stdout)
            return RunResult(
                output=clean_output,
                success=True,
                raw_output=result.stdout,
            )

        except subprocess.TimeoutExpired:
            return RunResult(
                output="",
                success=False,
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            return RunResult(
                output="",
                success=False,
                error=str(e),
            )

    @staticmethod
    def _clean_output(output: str) -> str:
        """Remove ANSI codes and strip agy CLI preamble lines.

        agy print mode may emit credential/model banner lines to stdout
        before the actual model output. These contaminate parsed responses,
        so strip any leading lines that match known patterns.
        """
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned = ansi_escape.sub('', output)

        preamble_patterns = (
            "Loaded cached credentials",
            "Data collection is disabled",
            "Using model:",
            "Auto-approve",
        )
        lines = cleaned.splitlines()
        start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(p) for p in preamble_patterns):
                start = i + 1
                continue
            start = i
            break
        return "\n".join(lines[start:]).strip()


def get_runner_for_model(model: str) -> ModelRunner:
    """
    Factory function to get the appropriate runner for a target model.

    Aliases vs pinned IDs:

    - ``"opus"`` uses the CLI alias — drifts with Anthropic releases. Pick this
      when you want the experiment to track whatever the current Opus generation
      is (normal optimization runs).
    - ``"opus-4-7"`` pins the exact model ID. Pick this for longitudinal
      studies where reproducibility across future model releases matters
      (e.g., baselines you'll re-run in six months to compare deltas).

    The same pattern is used in the Psyche benchmark (`benchmark/run_eval.py`).

    Args:
        model: One of "gpt", "gemini", "opus" (alias), or "opus-4-7" (pinned).

    Returns:
        Appropriate ModelRunner instance.
    """
    from .claude_runner import ClaudeRunner

    if model == "gpt":
        return CodexModelRunner()
    elif model == "gemini":
        return GeminiModelRunner()
    elif model == "opus":
        return ClaudeRunner(model="opus", timeout=480)
    elif model == "opus-4-7":
        # Pinned Opus 4.7 for longitudinal comparison. The CLI accepts full
        # model IDs as well as aliases.
        return ClaudeRunner(model="claude-opus-4-7", timeout=480)
    else:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Expected 'gpt', 'gemini', 'opus', or 'opus-4-7'."
        )
