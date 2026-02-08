"""
Diversity-based demo selection for prompt optimization.

Ensures selected demos represent diverse categories/scenarios
rather than just picking the highest-scoring similar examples.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .storage import Demo


@dataclass
class DemoWithContext:
    """Demo with additional context for diversity selection."""
    demo: Demo
    cluster: Optional[str] = None
    diversity_score: float = 0.0


def extract_cluster_from_metadata(demo: Demo) -> Optional[str]:
    """
    Extract cluster identifier from demo metadata.

    Checks common metadata fields that indicate category/type.

    Args:
        demo: Demo object with optional metadata

    Returns:
        Cluster identifier string or None
    """
    if not demo.metadata:
        return None

    # Common clustering fields in order of preference
    cluster_fields = [
        'category',
        'type',
        'query_type',
        'issue_type',
        'severity',
        'pattern',
        'tier',
        'decision',
        'tool',
    ]

    for field in cluster_fields:
        if field in demo.metadata:
            return str(demo.metadata[field])

    return None


def cluster_demos(demos: List[Demo]) -> Dict[str, List[Demo]]:
    """
    Group demos by cluster/category.

    Args:
        demos: List of Demo objects

    Returns:
        Dict mapping cluster names to lists of demos
    """
    clusters: Dict[str, List[Demo]] = defaultdict(list)

    for demo in demos:
        cluster = extract_cluster_from_metadata(demo) or "uncategorized"
        clusters[cluster].append(demo)

    return dict(clusters)


def select_diverse_demos(
    demos: List[Tuple[Any, str, float]],  # (TrainingExample, output, score)
    max_demos: int = 3,
    min_per_cluster: int = 1,
    score_weight: float = 0.7,
    diversity_weight: float = 0.3,
) -> List[Tuple[Any, str, float]]:
    """
    Select demos that balance high scores with diversity.

    Algorithm:
    1. Cluster demos by metadata category
    2. Ensure at least min_per_cluster from each cluster
    3. Fill remaining slots with highest-scoring demos

    Args:
        demos: List of (TrainingExample, output, score) tuples
        max_demos: Maximum demos to select
        min_per_cluster: Minimum demos per cluster (if available)
        score_weight: Weight for score in combined ranking
        diversity_weight: Weight for diversity in combined ranking

    Returns:
        Selected demos list
    """
    if len(demos) <= max_demos:
        return sorted(demos, key=lambda x: x[2], reverse=True)

    # Create Demo-like objects for clustering
    demo_wrappers = []
    for example, output, score in demos:
        metadata = example.metadata if hasattr(example, 'metadata') else None
        demo_wrappers.append({
            'original': (example, output, score),
            'cluster': extract_cluster_from_metadata(
                Demo(input_text="", output_text="", score=score, metadata=metadata)
            ) or "uncategorized",
            'score': score,
        })

    # Group by cluster
    clusters: Dict[str, List[dict]] = defaultdict(list)
    for wrapper in demo_wrappers:
        clusters[wrapper['cluster']].append(wrapper)

    # Sort each cluster by score
    for cluster in clusters.values():
        cluster.sort(key=lambda x: x['score'], reverse=True)

    selected: List[Tuple[Any, str, float]] = []
    selected_clusters: set = set()

    # Phase 1: Select best from each cluster (round-robin)
    cluster_iterators = {name: iter(demos) for name, demos in clusters.items()}
    rounds = 0

    while len(selected) < max_demos and cluster_iterators:
        empty_clusters = []

        for cluster_name, iterator in cluster_iterators.items():
            if len(selected) >= max_demos:
                break

            try:
                wrapper = next(iterator)
                selected.append(wrapper['original'])
                selected_clusters.add(cluster_name)
            except StopIteration:
                empty_clusters.append(cluster_name)

        # Remove exhausted clusters
        for cluster_name in empty_clusters:
            del cluster_iterators[cluster_name]

        rounds += 1
        if rounds >= max_demos:  # Safety limit
            break

    # Phase 2: Fill remaining with highest-scoring unselected
    if len(selected) < max_demos:
        selected_set = set(id(s) for s in selected)
        remaining = [
            w for w in demo_wrappers
            if id(w['original']) not in selected_set
        ]
        remaining.sort(key=lambda x: x['score'], reverse=True)

        for wrapper in remaining:
            if len(selected) >= max_demos:
                break
            selected.append(wrapper['original'])

    return selected


def compute_diversity_score(demos: List[Demo]) -> float:
    """
    Compute a diversity score for a set of demos.

    Higher score = more diverse representation.

    Args:
        demos: List of selected demos

    Returns:
        Diversity score between 0.0 and 1.0
    """
    if not demos:
        return 0.0

    clusters = cluster_demos(demos)
    num_clusters = len(clusters)
    num_demos = len(demos)

    if num_clusters == 0:
        return 0.0

    # Ideal: one demo per cluster (perfect diversity)
    # Score based on how evenly distributed demos are across clusters
    ideal_per_cluster = num_demos / num_clusters
    variance = sum(
        (len(demos_in_cluster) - ideal_per_cluster) ** 2
        for demos_in_cluster in clusters.values()
    ) / num_clusters

    # Normalize: 0 variance = 1.0 score, high variance = lower score
    max_variance = (num_demos - 1) ** 2  # Worst case: all in one cluster
    if max_variance == 0:
        return 1.0

    normalized_variance = variance / max_variance
    diversity_score = 1.0 - normalized_variance

    # Bonus for having more clusters represented
    cluster_coverage = num_clusters / num_demos if num_demos > 0 else 0
    diversity_score = 0.7 * diversity_score + 0.3 * min(1.0, cluster_coverage)

    return diversity_score


def rerank_with_diversity(
    demos: List[Demo],
    max_demos: int = 3,
) -> List[Demo]:
    """
    Re-rank demos to optimize for both score and diversity.

    Uses a greedy selection algorithm that picks demos that
    maximize combined score + diversity contribution.

    Args:
        demos: Pre-sorted list of demos (by score, descending)
        max_demos: Maximum demos to select

    Returns:
        Reranked list of demos
    """
    if len(demos) <= max_demos:
        return demos

    selected: List[Demo] = []
    remaining = list(demos)

    while len(selected) < max_demos and remaining:
        best_idx = 0
        best_combined_score = -1.0

        for i, demo in enumerate(remaining):
            # Calculate what diversity would be if we added this demo
            test_selected = selected + [demo]
            diversity = compute_diversity_score(test_selected)

            # Combined score: 70% demo score, 30% diversity contribution
            combined = 0.7 * demo.score + 0.3 * diversity
            if combined > best_combined_score:
                best_combined_score = combined
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


# Integration helper for BootstrapFewShot
def diversify_demo_selection(
    demos: List[Tuple[Any, str, float]],
    max_demos: int = 3,
) -> List[Tuple[Any, str, float]]:
    """
    Drop-in replacement for simple top-N selection.

    Use this in BootstrapFewShot._select_best_demos for diversity-aware selection.

    Args:
        demos: List of (TrainingExample, output, score) tuples
        max_demos: Maximum demos to select

    Returns:
        Diverse selection of demos
    """
    return select_diverse_demos(demos, max_demos=max_demos)
