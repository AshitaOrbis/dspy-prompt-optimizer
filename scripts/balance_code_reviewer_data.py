#!/usr/bin/env python3
"""
Create a category-balanced subset of code-reviewer training data.

The original dataset is security-dominated (40%). This creates a balanced
version with max 4 examples per category to prevent biased demo selection.

Usage:
    python3 scripts/balance_code_reviewer_data.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
INPUT_PATH = DATASETS_DIR / "code-reviews.jsonl"
OUTPUT_PATH = DATASETS_DIR / "code-reviews-balanced.jsonl"
MAX_PER_CATEGORY = 4

def main():
    # Load and categorize
    by_category = defaultdict(list)
    with open(INPUT_PATH) as f:
        for line in f:
            d = json.loads(line)
            meta = d.get("metadata", {})
            cat = meta.get("category", meta.get("type", "unknown"))
            by_category[cat].append(line.strip())

    print("Original distribution:")
    total_orig = 0
    for cat in sorted(by_category, key=lambda c: -len(by_category[c])):
        print(f"  {cat}: {len(by_category[cat])}")
        total_orig += len(by_category[cat])
    print(f"  Total: {total_orig}")

    # Balance: take up to MAX_PER_CATEGORY from each category
    random.seed(42)  # reproducible
    balanced = []
    for cat, examples in by_category.items():
        if len(examples) > MAX_PER_CATEGORY:
            selected = random.sample(examples, MAX_PER_CATEGORY)
        else:
            selected = examples
        balanced.extend(selected)

    # Shuffle
    random.shuffle(balanced)

    # Write
    with open(OUTPUT_PATH, "w") as f:
        for line in balanced:
            f.write(line + "\n")

    print(f"\nBalanced distribution:")
    balanced_cats = defaultdict(int)
    for line in balanced:
        d = json.loads(line)
        meta = d.get("metadata", {})
        cat = meta.get("category", meta.get("type", "unknown"))
        balanced_cats[cat] += 1
    for cat in sorted(balanced_cats, key=lambda c: -balanced_cats[c]):
        print(f"  {cat}: {balanced_cats[cat]}")
    print(f"  Total: {len(balanced)}")
    print(f"\nWritten to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
