"""
Prompt Optimizer Library

A DSPy-inspired prompt optimization framework that uses Claude Code CLI as the backend.
Implements BootstrapFewShot and related algorithms for automated few-shot example generation.

Usage:
    from prompt_optimizer import BootstrapFewShot, ClaudeRunner, metrics

    runner = ClaudeRunner()
    optimizer = BootstrapFewShot(runner, max_demos=3)

    optimized_prompt = optimizer.optimize(
        base_prompt="Evaluate this capability...",
        training_data=training_examples,
        metric_fn=metrics.score_similarity,
        threshold=0.7
    )
"""

from .claude_runner import ClaudeRunner
from .bootstrap import BootstrapFewShot
from .metrics import (
    exact_match,
    contains_keywords,
    score_similarity,
    composite_metric,
    numeric_score_match,
    evaluation_score_metric,
    # Phase 2A: Agent-specific metrics
    issue_severity_match,
    test_coverage_score,
    complexity_classification,
    security_cwe_match,
    # Phase 3: High-priority agent metrics
    root_cause_match,
    pr_quality_match,
    implementation_completeness,
    refactoring_match,
    # Plan quality metrics
    plan_quality_match,
    # Phase 2B: Skill-specific metrics
    routing_accuracy,
    binary_decision_match,
    tool_tier_classification,
    # Tier 2: Agent metrics
    discovery_quality_match,
    fact_check_quality_match,
    research_quality_match,
    # Tier 3: Skill metrics
    writing_review_quality_match,
    handoff_quality_match,
    # Publication review metric
    publication_review_match,
)
from .storage import DemoStorage
from .diversity import (
    select_diverse_demos,
    cluster_demos,
    compute_diversity_score,
    rerank_with_diversity,
)
from .batch import (
    BatchTarget,
    BatchResult,
    BatchSummary,
    optimize_batch_sequential,
    optimize_batch_parallel,
    generate_batch_report,
)
from .copro import (
    InstructionMutator,
    COPROOptimizer,
    COPROResult,
    MutationResult,
    VariantScore,
)
from .iterative import (
    SyntheticDataGenerator,
    IterativeOptimizer,
    IterativeResult,
    RoundResult,
)
from .verification import (
    VerificationSuite,
    HoldoutResult,
    CrossValidationResult,
    RegressionResult,
    VerificationReport,
    generate_verification_report,
)
from .utils import (
    parse_markdown_prompt,
    find_agent_path,
    find_skill_path,
    load_prompt_file,
)
from .validation import (
    validate_name,
    contained_path,
    ValidationError,
)
# Phase 6: Extractors, format instructions, and demo transformers
from .extractors import (
    extract_tool_from_verbose,
    normalize_tool_name,
    get_tool_family,
    extract_binary_decision,
    extract_tier,
    score_tool_match,
    # Publication review
    extract_publication_review,
    extract_review_findings,
    match_review_findings,
    ReviewFinding,
)
from .format_instructions import (
    get_format_instruction,
    get_format_type_for_target,
    append_format_instruction,
)
from .demo_transformers import (
    get_demo_transformer,
    transform_routing_demo,
    transform_binary_demo,
    transform_tier_demo,
    transform_review_demo,
    transform_demo_list,
    TransformedDemo,
)
from .model_runners import (
    ModelRunner,
    CodexModelRunner,
    GeminiModelRunner,
    get_runner_for_model,
)

__version__ = "0.5.0"  # Security hardening (2026-06-11 review)
__all__ = [
    "ClaudeRunner",
    "BootstrapFewShot",
    "DemoStorage",
    # Base metrics
    "exact_match",
    "contains_keywords",
    "score_similarity",
    "composite_metric",
    "numeric_score_match",
    "evaluation_score_metric",
    # Agent-specific metrics
    "issue_severity_match",
    "test_coverage_score",
    "complexity_classification",
    "security_cwe_match",
    # Phase 3: High-priority agent metrics
    "root_cause_match",
    "pr_quality_match",
    "implementation_completeness",
    "refactoring_match",
    # Plan quality metrics
    "plan_quality_match",
    # Skill-specific metrics
    "routing_accuracy",
    "binary_decision_match",
    "tool_tier_classification",
    # Diversity selection
    "select_diverse_demos",
    "cluster_demos",
    "compute_diversity_score",
    "rerank_with_diversity",
    # Batch optimization
    "BatchTarget",
    "BatchResult",
    "BatchSummary",
    "optimize_batch_sequential",
    "optimize_batch_parallel",
    "generate_batch_report",
    # COPRO algorithm
    "InstructionMutator",
    "COPROOptimizer",
    "COPROResult",
    "MutationResult",
    "VariantScore",
    # Iterative optimization
    "SyntheticDataGenerator",
    "IterativeOptimizer",
    "IterativeResult",
    "RoundResult",
    # Verification
    "VerificationSuite",
    "HoldoutResult",
    "CrossValidationResult",
    "RegressionResult",
    "VerificationReport",
    "generate_verification_report",
    # Utilities
    "parse_markdown_prompt",
    "find_agent_path",
    "find_skill_path",
    "load_prompt_file",
    # Validation
    "validate_name",
    "contained_path",
    "ValidationError",
    # Phase 6: Extractors
    "extract_tool_from_verbose",
    "normalize_tool_name",
    "get_tool_family",
    "extract_binary_decision",
    "extract_tier",
    "score_tool_match",
    # Publication review
    "extract_publication_review",
    "extract_review_findings",
    "match_review_findings",
    "ReviewFinding",
    "publication_review_match",
    # Phase 6: Format instructions
    "get_format_instruction",
    "get_format_type_for_target",
    "append_format_instruction",
    # Phase 6: Demo transformers
    "get_demo_transformer",
    "transform_routing_demo",
    "transform_binary_demo",
    "transform_tier_demo",
    "transform_demo_list",
    "transform_review_demo",
    "TransformedDemo",
    # Model runners
    "ModelRunner",
    "CodexModelRunner",
    "GeminiModelRunner",
    "get_runner_for_model",
]
