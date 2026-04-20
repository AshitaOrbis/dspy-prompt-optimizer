#!/usr/bin/env python3
"""
Generate writing-review training and holdout datasets.

Reads blog posts and generates gold-standard writing reviews using
claude -p --model opus. Outputs JSONL files for optimization pipeline.

Usage:
    python3 scripts/generate_writing_review_dataset.py
    python3 scripts/generate_writing_review_dataset.py --dry-run  # first 3 only
"""

import json
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

POSTS_DIR = Path.home() / "claudeworkspace/applications/ashitaorbis/shared/content/posts"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
VOICE_GUIDE = Path.home() / ".claude/skills/writing-review/references/voice-guide.md"

# Selected posts: 20 posts stratified by length and topic diversity
# 15 training + 5 holdout (holdout marked with *)
SELECTED_POSTS = [
    # Short posts (1.4K-2K words) -- 5 posts
    "036-the-mind-is-an-instrument",       # Philosophy/personal
    "032-the-etymology-tax",               # Research/linguistics
    "004-dead-blog-theory",                # Meta/blogging
    "040-when-the-pulse-went-quiet",       # *HOLDOUT* Narrative/personal
    "003-automating-prompt-engineering",    # Technical/AI

    # Medium posts (2K-2.5K words) -- 7 posts
    "012-red-teaming-your-business-ideas",  # Business/methodology
    "017-i-let-an-ai-loose-on-an-ai-social-network",  # *HOLDOUT* AI/experiment
    "007-ai-as-mirror",                    # Philosophy/AI
    "035-the-revision-tax",                # Writing process/meta
    "023-sandboxing-ai-agents-the-embassy-pattern",  # Technical/security
    "014-behaviorisms-hidden-legacy-in-reinforcement-learning",  # Academic/AI
    "013-persona-testing",                 # Methodology/testing

    # Long posts (2.5K-3.5K words) -- 5 posts
    "021-the-algorithms-of-self-improvement",  # *HOLDOUT* AI/systems
    "034-benchmarking-bullshit-detection",  # Research/benchmarking
    "009-ai-evaluating-ai",                # Meta/AI
    "033-the-container-that-forgot-to-stop",  # Technical/narrative
    "019-what-my-ai-learned-on-the-internet",  # *HOLDOUT* AI/experiment

    # Very long posts (3.5K+ words) -- 3 posts
    "038-where-to-spend-your-context-window",  # Technical/optimization
    "029-from-text-messages-to-literary-memoir",  # *HOLDOUT* Personal/creative
    "037-the-model-generation-audit",       # Research/benchmarking
]

HOLDOUT_SLUGS = {
    "040-when-the-pulse-went-quiet",
    "017-i-let-an-ai-loose-on-an-ai-social-network",
    "021-the-algorithms-of-self-improvement",
    "019-what-my-ai-learned-on-the-internet",
    "029-from-text-messages-to-literary-memoir",
}

REVIEW_PROMPT_TEMPLATE = """You are performing a comprehensive writing review of a blog post for the Ashita Orbis blog.

## Voice Guide Reference

{voice_guide}

## Review Instructions

Analyze this blog post from ALL 5 perspectives below. For each perspective, provide specific observations with quoted evidence from the text.

### Perspective 1: Editorial Critic (Argument Structure & Originality)
Evaluate: argument structure strength, evidence quality, originality of ideas, logical flow

### Perspective 2: Target Reader (Engagement & Value)
Evaluate: hook effectiveness, value proposition clarity, engagement level, shareability, reading experience

### Perspective 3: Tone Analyst (Voice Consistency)
Evaluate each of the 7 voice dimensions:
1. Laconic Style
2. Aphorism Subversion
3. Uncomfortable Conclusions
4. Academic-Emotional Bridge
5. Power Dynamics Analysis
6. Self-Contradiction as Truth-Seeking
7. Controlled Vulnerability
Rate each as: Matches Voice / Partial Match / Diverges

### Perspective 4: Technical Accuracy
Evaluate: technical claims accuracy, code/system descriptions, methodology soundness

### Perspective 5: Fact-Check Summary
Note: any factual claims that should be verified, historical references, statistics cited

## Output Format

Structure your review as:

## Editorial Critic Review: "[title]"
[Argument structure, evidence quality, originality assessment with quoted evidence]

## Target Reader Evaluation
[Engagement breakdown with Hook/Value/Engagement/Shareability ratings and evidence]

## Tone Analyst Review: "[title]"
**Voice Match Score: X/7 dimensions matching**
[Dimension-by-dimension table with Assessment and Evidence columns]

## Technical Accuracy Review
[Technical claims assessment]

## Fact-Check Summary
[Claims requiring verification]

## Overall Assessment
**Strongest element:** [with quote]
**Weakest element:** [with quote]
**Recommendation:** [Publish as-is / Minor revisions / Major revisions]

---

## Blog Post to Review

{post_content}
"""


def generate_review(post_path: Path, voice_guide_text: str) -> str:
    """Generate a gold-standard writing review using claude -p --model opus."""
    post_content = post_path.read_text()

    prompt = REVIEW_PROMPT_TEMPLATE.format(
        voice_guide=voice_guide_text,
        post_content=post_content,
    )

    # Use claude -p (print mode) with opus
    env = {k: v for k, v in os.environ.items()}
    env.pop("CLAUDECODE", None)  # Avoid nested session error

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "opus", "--max-turns", "1"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        if result.returncode != 0:
            print(f"  ERROR: claude returned {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return ""

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return ""
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return ""


def validate_review(review: str, post_slug: str) -> bool:
    """Validate that a review meets quality requirements."""
    if len(review) < 500:
        print(f"  FAIL: {post_slug} review too short ({len(review)} chars)")
        return False

    # Check for at least 2 perspective sections
    section_markers = [
        "editorial critic",
        "target reader",
        "tone analyst",
        "technical accuracy",
        "fact-check",
    ]
    sections_found = sum(1 for m in section_markers if m.lower() in review.lower())
    if sections_found < 2:
        print(f"  FAIL: {post_slug} only {sections_found} perspectives found (need 2+)")
        return False

    # Check for quoted evidence
    if '"' not in review and '"' not in review:
        print(f"  WARN: {post_slug} no quoted evidence found (continuing anyway)")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate writing-review dataset")
    parser.add_argument("--dry-run", action="store_true", help="Process first 3 posts only")
    args = parser.parse_args()

    # Read voice guide
    if not VOICE_GUIDE.exists():
        print(f"ERROR: Voice guide not found at {VOICE_GUIDE}")
        sys.exit(1)
    voice_guide_text = VOICE_GUIDE.read_text()

    posts_to_process = SELECTED_POSTS[:3] if args.dry_run else SELECTED_POSTS

    training_examples = []
    holdout_examples = []
    failed = []

    for i, slug in enumerate(posts_to_process):
        post_path = POSTS_DIR / f"{slug}.md"
        if not post_path.exists():
            print(f"[{i+1}/{len(posts_to_process)}] SKIP: {slug} (file not found)")
            failed.append(slug)
            continue

        post_content = post_path.read_text()
        word_count = len(post_content.split())

        print(f"\n[{i+1}/{len(posts_to_process)}] {slug} ({word_count} words)")
        print(f"  Generating review via opus...")

        review = generate_review(post_path, voice_guide_text)

        if not review:
            print(f"  FAILED: Empty review")
            failed.append(slug)
            continue

        if not validate_review(review, slug):
            failed.append(slug)
            continue

        print(f"  OK: {len(review)} chars, validated")

        example = {
            "input": post_content,
            "expected": review,
            "metadata": {
                "post_slug": slug,
                "word_count": word_count,
                "review_chars": len(review),
            },
        }

        if slug in HOLDOUT_SLUGS:
            holdout_examples.append(example)
            print(f"  -> HOLDOUT")
        else:
            training_examples.append(example)
            print(f"  -> TRAINING")

        # Small delay between API calls
        time.sleep(2)

    # Write datasets
    train_path = DATASETS_DIR / "writing-reviews.jsonl"
    holdout_path = DATASETS_DIR / "writing-reviews-holdout.jsonl"

    with open(train_path, "w") as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + "\n")

    with open(holdout_path, "w") as f:
        for ex in holdout_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'='*50}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Training examples: {len(training_examples)} -> {train_path}")
    print(f"Holdout examples:  {len(holdout_examples)} -> {holdout_path}")
    print(f"Failed:            {len(failed)}")
    if failed:
        print(f"  Failed slugs: {failed}")


if __name__ == "__main__":
    main()
