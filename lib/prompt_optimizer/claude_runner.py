"""
Claude Code CLI wrapper for prompt optimization.

Provides a clean interface for running prompts through Claude Code CLI
and parsing the output for use in optimization algorithms.
"""

import os
import subprocess
import json
import re
from dataclasses import dataclass
from typing import Optional


# Default timeout in seconds (can be overridden via env var)
DEFAULT_TIMEOUT = 180


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
        self.timeout = timeout
        self.working_dir = working_dir

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

        cmd = [
            "claude",
            "--print",  # Non-interactive, just print output
            "--model", self.model,
            "--dangerously-skip-permissions",  # Non-interactive mode
            "--",  # End of options marker (prevents prompts starting with "-" from being parsed as flags)
            full_prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )

            if result.returncode != 0:
                return RunResult(
                    output="",
                    success=False,
                    error=result.stderr or f"Exit code: {result.returncode}",
                    raw_output=result.stdout,
                )

            # Clean output - remove any ANSI codes
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
                error=f"Timeout after {self.timeout} seconds",
            )
        except Exception as e:
            return RunResult(
                output="",
                success=False,
                error=str(e),
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
        cmd = [
            "claude",
            "--print",
            "--model", self.model,
            "--dangerously-skip-permissions",
            "--system-prompt", system_prompt,
            "--",  # End of options marker
            full_prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )

            if result.returncode != 0:
                return RunResult(
                    output="",
                    success=False,
                    error=result.stderr or f"Exit code: {result.returncode}",
                    raw_output=result.stdout,
                )

            clean_output = self._clean_output(result.stdout)
            return RunResult(output=clean_output, success=True, raw_output=result.stdout)

        except subprocess.TimeoutExpired:
            return RunResult(output="", success=False, error=f"Timeout after {self.timeout}s")
        except Exception as e:
            return RunResult(output="", success=False, error=str(e))

    def with_model(self, model: str) -> 'ClaudeRunner':
        """
        Return a new runner with a different model.

        Useful for tiered optimization (Haiku -> Sonnet -> Opus).

        Args:
            model: New model to use (sonnet, opus, haiku)

        Returns:
            New ClaudeRunner instance with the specified model
        """
        return ClaudeRunner(
            model=model,
            timeout=self.timeout,
            working_dir=self.working_dir,
        )
