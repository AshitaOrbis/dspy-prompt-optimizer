#!/usr/bin/env python3
"""
Convert review-audit data into JSONL training files for publication-review optimization.

Uses review-state.json as the primary data source (complete reviewer attribution),
with fix-manifests for DISCARDED items only.

Reads:
- Blog posts from shared/content/posts/{slug}.md
- Review state from shared/content/review-audit/review-state.json (primary: findings + reviewers)
- Fix manifests from shared/content/review-audit/rounds/{slug}/fix-manifest.md (DISCARDED only)

Produces per-model JSONL files (training + holdout) in datasets/.

Usage:
    python scripts/convert_review_audit.py
    python scripts/convert_review_audit.py --output-dir datasets/ --dry-run
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Paths
ASHITA_ROOT = Path.home() / "claudeworkspace" / "applications" / "ashitaorbis"
POSTS_DIR = ASHITA_ROOT / "shared" / "content" / "posts"
AUDIT_DIR = ASHITA_ROOT / "shared" / "content" / "review-audit"
ROUNDS_DIR = AUDIT_DIR / "rounds"
REVIEW_STATE = AUDIT_DIR / "review-state.json"

DEFAULT_OUTPUT = Path(__file__).parent.parent / "datasets"

# Posts excluded from audit
EXCLUDED_POSTS = {"004-dead-blog-theory", "035-the-revision-tax"}

# Train/holdout split: 001-027 = training, 028-034 = holdout
HOLDOUT_PREFIXES = {"028", "029", "030", "031", "032", "033", "034"}

TIER_ORDER = ["MUST", "SHOULD", "NICE"]

# Model name aliases for manifest Source column parsing
MODEL_ALIASES = {
    "gpt": ["GPT", "gpt", "GPT-5.4"],
    "gemini": ["Gemini", "gemini", "Gemini 3.1 Pro"],
    "opus": ["Opus", "opus", "Opus 4.6"],
}


@dataclass
class Finding:
    tier: str  # MUST, SHOULD, NICE
    description: str
    reviewers: List[str]  # ["gpt", "gemini", "opus"]


@dataclass
class PostData:
    slug: str
    findings: List[Finding] = field(default_factory=list)
    discarded_count: int = 0


def _extract_sources_from_text(text: str) -> List[str]:
    """Extract model names from a source string like 'GPT, Opus'."""
    sources = []
    text_lower = text.lower()
    for model, aliases in MODEL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                sources.append(model)
                break
    return sources


def _parse_manifest_sources(slug: str) -> Dict[str, List[str]]:
    """Parse manifest Source(s) column to get additional reviewer attribution.

    Returns dict mapping finding description (first 60 chars, lowered) to reviewer list.
    This supplements review-state.json which only has original reviewers, not
    cross-model triage attribution from phase 2.
    """
    manifest_path = ROUNDS_DIR / slug / "fix-manifest.md"
    if not manifest_path.exists():
        return {}

    text = manifest_path.read_text()
    sources_map = {}

    # Parse table rows from MUST FIX, SHOULD FIX, NICE TO HAVE sections
    for tier_pattern in [r"##\s*MUST\s*FIX", r"##\s*SHOULD\s*FIX", r"##\s*NICE\s*(?:TO\s*)?HAVE"]:
        match = re.search(tier_pattern, text, re.IGNORECASE)
        if not match:
            continue

        start = match.end()
        next_header = re.search(r'\n##\s', text[start:])
        section = text[start:start + next_header.start()] if next_header else text[start:]

        # Parse rows: | # | Finding | Source(s) | ...
        for m in re.finditer(r'\|\s*\d+\s*\|([^|]+)\|([^|]+)', section):
            desc = m.group(1).strip()
            sources_cell = m.group(2).strip()
            if desc and not desc.startswith("Finding") and not desc.startswith("---"):
                key = desc[:60].lower().strip()
                sources_map[key] = _extract_sources_from_text(sources_cell)

    return sources_map


def _parse_manifest_findings(slug: str) -> List[Finding]:
    """Parse manifest tables directly for findings with Source attribution.

    The manifest has the triaged Source(s) column which includes all models
    that flagged each finding (including cross-model triage from phase 2).
    This is more complete than review-state.json's original reviewer lists.
    """
    manifest_path = ROUNDS_DIR / slug / "fix-manifest.md"
    if not manifest_path.exists():
        return []

    text = manifest_path.read_text()
    findings = []

    tier_patterns = {
        "MUST": r"##\s*MUST\s*FIX",
        "SHOULD": r"##\s*SHOULD\s*FIX",
        "NICE": r"##\s*NICE\s*(?:TO\s*)?HAVE",
    }

    for tier, pattern in tier_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        start = match.end()
        next_header = re.search(r'\n##\s', text[start:])
        section = text[start:start + next_header.start()] if next_header else text[start:]

        # Parse rows: | # | Finding | Source(s) | ...
        for m in re.finditer(r'\|\s*(\d+)\s*\|([^|]+)\|([^|]+)', section):
            desc = m.group(2).strip()
            sources_cell = m.group(3).strip()

            if not desc or desc.startswith("Finding") or desc.startswith("---"):
                continue
            if len(desc) < 5:
                continue

            reviewers = _extract_sources_from_text(sources_cell)
            if reviewers:
                findings.append(Finding(
                    tier=tier,
                    description=desc,
                    reviewers=reviewers,
                ))

    return findings


def load_posts() -> Dict[str, PostData]:
    """Load post data, preferring manifest for findings (has triaged Source attribution).

    Strategy:
    - If manifest exists: parse it directly (Source column has cross-model triage)
    - If no manifest: fall back to review-state.json (original reviewer lists)
    """
    # Load review-state for fallback
    with open(REVIEW_STATE) as f:
        state = json.load(f)

    posts = {}
    for slug, post_data in state.get("posts", {}).items():
        if slug in EXCLUDED_POSTS or slug.startswith("series"):
            continue

        pd = PostData(slug=slug)

        # Try manifest first (has triaged Source attribution)
        manifest_findings = _parse_manifest_findings(slug)
        if manifest_findings:
            pd.findings = manifest_findings
        else:
            # Fallback to review-state.json
            for round_data in post_data.get("rounds", []):
                for finding in round_data.get("findings", []):
                    tier = finding.get("tier", "").upper()
                    if tier in ("MUST_FIX", "MUST"):
                        tier = "MUST"
                    elif tier in ("SHOULD_FIX", "SHOULD"):
                        tier = "SHOULD"
                    elif tier in ("NICE_TO_HAVE", "NICE"):
                        tier = "NICE"
                    else:
                        continue

                    desc = finding.get("description", "")
                    reviewers = list(finding.get("reviewers", []))

                    if desc and reviewers:
                        pd.findings.append(Finding(
                            tier=tier,
                            description=desc,
                            reviewers=reviewers,
                        ))

        posts[slug] = pd

    return posts


def count_discarded(slug: str) -> int:
    """Count DISCARDED items from fix-manifest (if it exists)."""
    manifest_path = ROUNDS_DIR / slug / "fix-manifest.md"
    if not manifest_path.exists():
        return 0

    text = manifest_path.read_text()
    match = re.search(r'##\s*DISCARDED\s*\n(.*?)(?=\n##|\Z)', text, re.DOTALL)
    if not match:
        return 0

    section = match.group(1)
    rows = re.findall(r'\|\s*\w+[-\d]*\s*\|[^|]+\|', section)
    return sum(1 for r in rows if '---' not in r and 'Finding' not in r)


def read_post_text(slug: str, max_words: int = 2000) -> Optional[str]:
    """Read a blog post, returning first max_words words."""
    post_path = POSTS_DIR / f"{slug}.md"
    if not post_path.exists():
        return None

    text = post_path.read_text()

    # Strip frontmatter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].strip()

    # Truncate to max_words
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "\n\n[...truncated...]"

    return text


def read_methodology_brief(slug: str) -> Optional[str]:
    """Read methodology brief if it exists."""
    brief_path = POSTS_DIR / f"{slug}.methodology.md"
    if brief_path.exists():
        return brief_path.read_text()
    return None


def build_expected_output(findings: List[Finding]) -> str:
    """Build the gold standard expected output from findings."""
    sections = {"MUST": [], "SHOULD": [], "NICE": []}

    for f in findings:
        sections[f.tier].append(f.description)

    parts = []
    if sections["MUST"]:
        parts.append("## MUST FIX")
        for i, desc in enumerate(sections["MUST"], 1):
            parts.append(f"{i}. **{desc[:80]}** — {desc}")

    if sections["SHOULD"]:
        parts.append("\n## SHOULD FIX")
        for i, desc in enumerate(sections["SHOULD"], 1):
            parts.append(f"{i}. **{desc[:80]}** — {desc}")

    if sections["NICE"]:
        parts.append("\n## NICE TO HAVE")
        for i, desc in enumerate(sections["NICE"], 1):
            parts.append(f"{i}. **{desc[:80]}** — {desc}")

    return "\n".join(parts)


def make_record(
    slug: str,
    post_text: str,
    post_data: PostData,
    model: str,
    methodology: Optional[str] = None,
) -> Optional[Dict]:
    """Create a JSONL record for a post/model combination.

    Uses ALL findings as the expected output — we want each model to find
    everything that's real, not just what it historically flagged first.
    The per-model differentiation comes from the prompt template, not the
    training data.
    """
    model_findings = post_data.findings

    if not model_findings:
        return None

    # Build input
    input_text = post_text
    if methodology:
        input_text += (
            "\n\n=== METHODOLOGY BRIEF (not part of the published post) ===\n"
            + methodology
            + "\n==="
        )

    expected = build_expected_output(model_findings)

    # Count by tier
    must_count = sum(1 for f in model_findings if f.tier == "MUST")
    should_count = sum(1 for f in model_findings if f.tier == "SHOULD")
    nice_count = sum(1 for f in model_findings if f.tier == "NICE")

    return {
        "input": input_text,
        "expected": expected,
        "metadata": {
            "post_slug": slug,
            "target_model": model,
            "must_count": must_count,
            "should_count": should_count,
            "nice_count": nice_count,
            "discarded_count": post_data.discarded_count,
            "total_findings": len(model_findings),
            "word_count": len(input_text.split()),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert review-audit data to JSONL")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    parser.add_argument("--max-words", type=int, default=2000, help="Max words per post input")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load from manifests (triaged Source attribution) + review-state.json fallback
    posts = load_posts()
    print(f"Loaded {len(posts)} posts")

    # Count discarded items from manifests
    for slug, pd in posts.items():
        pd.discarded_count = count_discarded(slug)

    # Build records per model
    records = {"gpt": [], "gemini": [], "opus": []}
    holdout_records = {"gpt": [], "gemini": [], "opus": []}

    for slug, post_data in sorted(posts.items()):
        if not post_data.findings:
            continue

        post_text = read_post_text(slug, max_words=args.max_words)
        if not post_text:
            print(f"  SKIP {slug}: post not found")
            continue

        methodology = read_methodology_brief(slug)
        prefix = slug[:3]
        is_holdout = prefix in HOLDOUT_PREFIXES

        for model in ["gpt", "gemini", "opus"]:
            record = make_record(
                slug=slug,
                post_text=post_text,
                post_data=post_data,
                model=model,
                methodology=methodology,
            )
            if record:
                target = holdout_records if is_holdout else records
                target[model].append(record)

    # Print stats
    print("\nTraining records:")
    for model, recs in records.items():
        total_findings = sum(r["metadata"]["total_findings"] for r in recs)
        print(f"  {model}: {len(recs)} posts, {total_findings} total findings")
    print("Holdout records:")
    for model, recs in holdout_records.items():
        total_findings = sum(r["metadata"]["total_findings"] for r in recs)
        print(f"  {model}: {len(recs)} posts, {total_findings} total findings")

    if args.dry_run:
        print("\nDry run — no files written")
        return

    # Write JSONL files
    for model in ["gpt", "gemini", "opus"]:
        # Training
        train_path = output_dir / f"publication-review-{model}.jsonl"
        with open(train_path, "w") as f:
            for record in records[model]:
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {train_path} ({len(records[model])} records)")

        # Holdout
        holdout_path = output_dir / f"publication-review-{model}-holdout.jsonl"
        with open(holdout_path, "w") as f:
            for record in holdout_records[model]:
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {holdout_path} ({len(holdout_records[model])} records)")


if __name__ == "__main__":
    main()
