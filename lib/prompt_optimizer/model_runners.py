"""
Model runners for multi-model prompt optimization.

Provides CodexModelRunner (GPT-5.4) and GeminiModelRunner alongside
the existing ClaudeRunner. All return RunResult for drop-in compatibility
with BootstrapFewShot.optimize().

CLI invocations:
  - codex exec: passes prompt via stdin (large prompts break as CLI args)
  - agy --input-format=stream-json: passes the repository's provisional typed
    prompt event via stdin. Local help confirms NDJSON transport, but the exact
    live request envelope remains unverified (see SECURITY.md).
  - claude --print: ClaudeRunner uses the same hardened subprocess primitive
"""

import json
import os
import re
import shutil
from abc import ABC, abstractmethod
from typing import Optional

from .claude_runner import (
    RunResult,
    permission_automation_enabled,
    run_hardened_subprocess,
    validate_timeout,
)


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
        self.timeout = validate_timeout(timeout)
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
            "-c", f'model="{self.model}"',
        ]
        if permission_automation_enabled():
            # Current Codex CLI's bounded approval automation. Unlike the old
            # --full-auto flag, this retains the workspace-write sandbox.
            cmd.append("--approve-for-me")
        for server_name in self._get_configured_mcp_servers():
            cmd.extend(["-c", f"mcp_servers.{server_name}.enabled=false"])
        cmd.extend([
            "--ephemeral",  # Don't persist session files
            "-",  # Read prompt from stdin
        ])

        result = run_hardened_subprocess(
            cmd,
            stdin_text=full_prompt,
            timeout=self.timeout,
            sensitive_values=tuple(
                value for value in (full_prompt, prompt, context) if value
            ),
        )
        if result.success:
            result.output = self._clean_output(result.output)
        return result

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
    longer eligible), so Gemini access now goes through `agy`. Local agy help
    and changelog text confirm NDJSON stdin mode and named stream-json output
    events. They do not establish the exact request envelope used below; that
    schema is provisional and covered by mocks pending a networked contract run.
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
        self.timeout = validate_timeout(timeout)
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
        """Run the provisional prompt event through agy's stream-json mode."""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        cmd = [
            self._binary,
            "--model", self.model,
            "--print-timeout", f"{max(self.timeout - 1, 1)}s",
            "--input-format=stream-json",
            "--output-format=stream-json",
            "--print=",
        ]
        if permission_automation_enabled():
            cmd.append("--dangerously-skip-permissions")

        # Provisional schema: local help documents one NDJSON message per line,
        # but offline evidence does not specify the message object's exact keys.
        # The mocked tests assert this repository-side expectation only.
        stdin_payload = json.dumps(
            {"type": "user", "message": full_prompt},
            ensure_ascii=False,
        ) + "\n"
        result = run_hardened_subprocess(
            cmd,
            stdin_text=stdin_payload,
            timeout=self.timeout,
            sensitive_values=tuple(
                value for value in (full_prompt, prompt, context) if value
            ),
        )
        if not result.success:
            return result
        try:
            result.output = self._clean_output(result.output)
        except ValueError as exc:
            return RunResult(output="", success=False, error=str(exc))
        return result

    @staticmethod
    def _clean_output(output: str) -> str:
        """Extract model text from the provisionally expected result event."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        terminal_result = None
        for line_number, line in enumerate(output.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid agy stream-json event on line {line_number}"
                ) from exc
            if not isinstance(event, dict) or event.get("type") != "result":
                continue
            if event.get("is_error"):
                raise ValueError("agy stream-json terminal result reported an error")
            value = event.get("result")
            if not isinstance(value, str):
                raise ValueError("agy stream-json terminal result has no text result")
            terminal_result = value

        if terminal_result is None:
            raise ValueError("agy stream-json output has no terminal result event")
        return ansi_escape.sub("", terminal_result).strip()


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
        model: One of "gpt", "gemini", "opus"/"opus-4-7", or "fable"/"fable-5".

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
    elif model == "fable":
        # CLI alias — tracks the current Fable (Mythos-class) generation.
        # Verified `claude -p --model fable` is accepted (2026-07-06).
        return ClaudeRunner(model="fable", timeout=480)
    elif model == "fable-5":
        # Pinned Fable 5 for longitudinal comparison.
        return ClaudeRunner(model="claude-fable-5", timeout=480)
    else:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Expected 'gpt', 'gemini', 'opus', 'opus-4-7', 'fable', or 'fable-5'."
        )
