"""Regression tests for optimized-prompt Markdown serialization."""

from pathlib import Path

from lib.prompt_optimizer.storage import OptimizedPrompt


def _extract_base_prompt(markdown: str) -> str:
    lines = markdown.splitlines()
    heading_index = lines.index("## Base Prompt")
    opening_index = heading_index + 2
    fence = lines[opening_index]
    assert len(fence) >= 3 and set(fence) == {"`"}
    closing_index = lines.index(fence, opening_index + 1)
    return "\n".join(lines[opening_index + 1 : closing_index])


def test_base_prompt_round_trips_with_nested_fences_and_headings() -> None:
    base_prompt = (
        "System instructions.\n"
        "## Embedded Heading\n"
        "```python\n"
        "print('inside triple fence')\n"
        "```\n"
        "````text\n"
        "inside four-backtick fence\n"
        "````"
    )
    optimized = OptimizedPrompt(
        base_prompt=base_prompt,
        demos=[],
        optimization_date="2026-08-23T00:00:00Z",
        metric_name="example_metric",
        threshold=0.7,
        avg_score=0.0,
    )

    markdown = optimized.to_markdown()

    assert _extract_base_prompt(markdown) == base_prompt
    assert "\n`````\n" in markdown
