"""
Storage utilities for persisting optimized prompts and demo examples.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common non-serializable types."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


@dataclass
class Demo:
    """A successful demonstration example."""
    input_text: str
    output_text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OptimizedPrompt:
    """An optimized prompt with attached demonstrations."""
    base_prompt: str
    demos: List[Demo]
    optimization_date: str
    metric_name: str
    threshold: float
    avg_score: float
    metadata: Optional[Dict[str, Any]] = None
    format_instruction: str = ""  # Phase 6: Store format instruction for evaluation

    def to_markdown(self) -> str:
        """Convert to markdown format suitable for CLAUDE.md or skills."""
        lines = [
            f"# Optimized Prompt",
            f"",
            f"**Optimized**: {self.optimization_date}",
            f"**Metric**: {self.metric_name}",
            f"**Threshold**: {self.threshold}",
            f"**Avg Demo Score**: {self.avg_score:.2f}",
            f"",
            f"## Base Prompt",
            f"",
            f"```",
            self.base_prompt,
            f"```",
            f"",
            f"## Few-Shot Examples",
            f"",
        ]

        for i, demo in enumerate(self.demos, 1):
            lines.extend([
                f"### Example {i} (Score: {demo.score:.2f})",
                f"",
                f"**Input:**",
                f"```",
                demo.input_text[:500] + ("..." if len(demo.input_text) > 500 else ""),
                f"```",
                f"",
                f"**Output:**",
                f"```",
                demo.output_text[:1000] + ("..." if len(demo.output_text) > 1000 else ""),
                f"```",
                f"",
            ])

        return "\n".join(lines)

    def to_prompt(self) -> str:
        """Generate the full prompt with few-shot examples."""
        parts = [self.base_prompt, "", "## Examples", ""]

        for i, demo in enumerate(self.demos, 1):
            parts.extend([
                f"### Example {i}",
                f"",
                f"Input:",
                demo.input_text,
                f"",
                f"Output:",
                demo.output_text,
                f"",
            ])

        # Add format instruction if present (Phase 6)
        if self.format_instruction:
            parts.append(self.format_instruction)
            parts.append("")

        parts.append("Now apply this pattern to the new input.")
        return "\n".join(parts)


class DemoStorage:
    """Persistent storage for demos and optimized prompts."""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize storage.

        Args:
            storage_dir: Directory to store optimization artifacts.
                        Defaults to PROMPT_OPTIMIZER_DIR env var or ~/.claude/prompt_optimizer
        """
        if storage_dir is None:
            storage_dir = os.environ.get(
                "PROMPT_OPTIMIZER_DIR",
                os.path.expanduser("~/.claude/prompt_optimizer")
            )
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.demos_dir = self.storage_dir / "demos"
        self.prompts_dir = self.storage_dir / "prompts"
        self.demos_dir.mkdir(exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)

    def save_demos(self, agent_name: str, demos: List[Demo]) -> Path:
        """
        Save demos for an agent.

        Args:
            agent_name: Name of the agent/task
            demos: List of Demo objects

        Returns:
            Path to saved file
        """
        file_path = self.demos_dir / f"{agent_name}.json"
        data = {
            "agent": agent_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "demos": [asdict(d) for d in demos],
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, cls=SafeJSONEncoder)
        return file_path

    def load_demos(self, agent_name: str) -> List[Demo]:
        """
        Load demos for an agent.

        Args:
            agent_name: Name of the agent/task

        Returns:
            List of Demo objects
        """
        file_path = self.demos_dir / f"{agent_name}.json"
        if not file_path.exists():
            return []

        with open(file_path) as f:
            data = json.load(f)

        demos = []
        for d in data.get("demos", []):
            # Validate required fields
            required = {"input_text", "output_text", "score"}
            missing = required - set(d.keys())
            if missing:
                raise ValueError(f"Demo missing required fields: {missing}")
            # Only pass known fields to Demo constructor
            demo_data = {k: v for k, v in d.items() if k in Demo.__dataclass_fields__}
            demos.append(Demo(**demo_data))
        return demos

    def save_optimized_prompt(
        self,
        agent_name: str,
        optimized: OptimizedPrompt,
        format: str = "both",
    ) -> Dict[str, Path]:
        """
        Save an optimized prompt.

        Args:
            agent_name: Name of the agent
            optimized: OptimizedPrompt object
            format: "json", "markdown", or "both"

        Returns:
            Dict of format -> path
        """
        valid_formats = {"json", "markdown", "both"}
        if format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}, got '{format}'")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        paths = {}

        if format in ("json", "both"):
            json_path = self.prompts_dir / f"{agent_name}_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(asdict(optimized), f, indent=2, cls=SafeJSONEncoder)
            paths["json"] = json_path

        if format in ("markdown", "both"):
            md_path = self.prompts_dir / f"{agent_name}_{timestamp}.md"
            with open(md_path, "w") as f:
                f.write(optimized.to_markdown())
            paths["markdown"] = md_path

        # Also save as "latest"
        latest_json = self.prompts_dir / f"{agent_name}_latest.json"
        with open(latest_json, "w") as f:
            json.dump(asdict(optimized), f, indent=2, cls=SafeJSONEncoder)
        paths["latest"] = latest_json

        return paths

    def load_optimized_prompt(self, agent_name: str) -> Optional[OptimizedPrompt]:
        """
        Load the latest optimized prompt for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            OptimizedPrompt or None if not found
        """
        file_path = self.prompts_dir / f"{agent_name}_latest.json"
        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)

        demos = [Demo(**d) for d in data.get("demos", [])]
        return OptimizedPrompt(
            base_prompt=data["base_prompt"],
            demos=demos,
            optimization_date=data["optimization_date"],
            metric_name=data["metric_name"],
            threshold=data["threshold"],
            avg_score=data["avg_score"],
            metadata=data.get("metadata"),
            format_instruction=data.get("format_instruction", ""),  # Phase 6
        )

    def list_optimized_prompts(self) -> List[str]:
        """List all agents with optimized prompts."""
        agents = set()
        for path in self.prompts_dir.glob("*_latest.json"):
            agent = path.stem.replace("_latest", "")
            agents.add(agent)
        return sorted(agents)
