"""
Model runners for multi-model prompt optimization.

Provides CodexModelRunner (GPT-5.4) and GeminiModelRunner alongside
the existing ClaudeRunner. All return RunResult for drop-in compatibility
with BootstrapFewShot.optimize().

CLI invocations:
  - codex exec: passes prompt via stdin (large prompts break as CLI args)
  - gemini -p: passes prompt via stdin piped to the -p flag
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

    def run(self, prompt: str, context: Optional[str] = None) -> RunResult:
        """Run prompt through Codex CLI via stdin."""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        # Use stdin (-) for the prompt to avoid CLI arg length limits.
        # codex exec reads from stdin when prompt arg is "-" or omitted.
        # Disable all MCP servers to avoid startup overhead and token waste.
        # Without MCP: ~1.7K tokens, instant startup.
        # With MCP: ~11.5K tokens, ~10s startup per call.
        cmd = [
            self._binary,
            "exec",
            "--full-auto",
            "-c", f'model="{self.model}"',
            "-c", "mcp_servers.brave-search.enabled=false",
            "-c", "mcp_servers.exa.enabled=false",
            "-c", "mcp_servers.filesystem.enabled=false",
            "-c", "mcp_servers.memory.enabled=false",
            "--ephemeral",  # Don't persist session files
            "-",  # Read prompt from stdin
        ]

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
    Runner for Gemini 3.1 Pro via Gemini CLI.

    Uses `gemini -p` for non-interactive mode. Writes prompt to a temp file
    and passes via stdin to handle large prompts.
    """

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        timeout: int = 480,
    ):
        self.model = model
        self.timeout = timeout
        self._binary = self._find_gemini_binary()

    @staticmethod
    def _find_gemini_binary() -> str:
        """Find the gemini binary."""
        found = shutil.which("gemini")
        if found:
            return found

        home = os.path.expanduser("~")
        nvm_dir = os.path.join(home, ".nvm/versions/node")
        if os.path.isdir(nvm_dir):
            for d in sorted(os.listdir(nvm_dir), reverse=True):
                candidate = os.path.join(nvm_dir, d, "bin/gemini")
                if os.path.isfile(candidate):
                    return candidate

        return "gemini"

    def run(self, prompt: str, context: Optional[str] = None) -> RunResult:
        """Run prompt through Gemini CLI.

        Pipes the prompt via stdin. The -p flag tells gemini to run
        non-interactively; stdin content is appended to the prompt.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        # Pass a short -p flag; pipe the actual content via stdin.
        # Gemini CLI docs: "Appended to input on stdin (if any)"
        cmd = [
            self._binary,
            "-p", "Review the following document:",
            "-m", self.model,
            "--yolo",
        ]

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


def get_runner_for_model(model: str) -> ModelRunner:
    """
    Factory function to get the appropriate runner for a target model.

    Args:
        model: One of "gpt", "gemini", "opus"

    Returns:
        Appropriate ModelRunner instance
    """
    from .claude_runner import ClaudeRunner

    if model == "gpt":
        return CodexModelRunner()
    elif model == "gemini":
        return GeminiModelRunner()
    elif model == "opus":
        return ClaudeRunner(model="opus", timeout=480)
    else:
        raise ValueError(f"Unknown model: {model}. Expected 'gpt', 'gemini', or 'opus'")
