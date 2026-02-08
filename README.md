# DSPy Prompt Optimizer

Automated prompt optimization for Claude Code skills and agents, inspired by Stanford's [DSPy](https://github.com/stanfordnlp/dspy) framework.

Instead of hand-tuning agent prompts and skill instructions, this system automatically discovers the best few-shot examples and prompt configurations through programmatic experimentation.

## How It Works

Define a metric function that scores output quality (0.0-1.0). The optimizer runs the target prompt on training examples, keeps successful traces as few-shot demonstrations, and attaches them to the prompt. The result: measurably better prompts with no manual tuning.

```
Training Data -> Run Base Prompt -> Score Outputs -> Keep Best Traces -> Optimized Prompt
```

## Algorithms

### BootstrapFewShot

The core algorithm (from DSPy). Run the model on training examples, keep successful traces where output scores above a threshold, and attach those as few-shot demonstrations.

```python
from prompt_optimizer import BootstrapFewShot, ClaudeRunner

optimizer = BootstrapFewShot(
    runner=ClaudeRunner(model="haiku"),
    max_demos=5,
    score_threshold=0.7
)
result = optimizer.optimize(
    prompt="Your agent/skill prompt",
    training_data=examples,
    metric_fn=your_metric
)
```

### COPRO (Coordinate Prompt Optimization)

Instruction-level optimization. Generate prompt variants through mutations (paraphrase, elaborate, simplify, extend), evaluate each on training data, pick the winner.

```python
from prompt_optimizer import COPROOptimizer

optimizer = COPROOptimizer(
    runner=ClaudeRunner(model="haiku"),
    mutations=["paraphrase", "elaborate", "simplify"],
    candidates_per_round=3
)
```

### Iterative Optimization

Combines COPRO + Bootstrap across multiple rounds with cross-validation and dropout regularization. The most thorough approach.

```python
from prompt_optimizer import IterativeOptimizer

optimizer = IterativeOptimizer(
    runner=ClaudeRunner(model="haiku"),
    rounds=3,
    cv_folds=3,
    dropout=0.3
)
```

## Metrics

15+ task-specific metrics for Claude Code evaluation:

| Metric | Domain | What It Measures |
|--------|--------|------------------|
| `issue_severity_match` | Code Review | Severity classification accuracy |
| `test_coverage_score` | Test Writing | Test count, categories, assertions |
| `complexity_classification` | Performance | Pattern recognition (N+1, O(n^2)) |
| `security_cwe_match` | Security Audit | CWE identifier extraction |
| `root_cause_match` | Debugging | Cause identification, fix steps |
| `pr_quality_match` | PR Preparation | Change type, risk, sections |
| `implementation_completeness` | Features | Acceptance criteria coverage |
| `refactoring_match` | Refactoring | Smell-technique affinity |
| `routing_accuracy` | Search Routing | Tool selection accuracy |
| `binary_decision_match` | Binary Decisions | Binary classification accuracy |
| `tool_tier_classification` | Tool Selection | Core/Specialized/Deferred |
| `plan_quality_match` | Planning | Multi-dimension quality scoring |
| `evaluation_score_match` | Evaluations | Scoring accuracy |

## Project Structure

```
dspy-prompt-optimizer/
├── lib/prompt_optimizer/     # Core library (7,900+ lines)
│   ├── bootstrap.py          # BootstrapFewShot algorithm
│   ├── copro.py              # COPRO instruction optimization
│   ├── iterative.py          # Multi-round iterative optimizer
│   ├── metrics.py            # 15+ task-specific metrics
│   ├── claude_runner.py      # Claude CLI wrapper
│   ├── storage.py            # Demo/prompt persistence
│   ├── verification.py       # A/B testing and validation
│   ├── extractors.py         # Output parsing
│   ├── format_instructions.py # Prompt formatting
│   ├── demo_transformers.py  # Demo processing
│   ├── diversity.py          # Demo diversity selection
│   ├── batch.py              # Batch optimization
│   └── utils.py              # Utilities
├── datasets/                 # 12 train/holdout pairs (.jsonl)
├── scripts/                  # Optimization and deployment scripts
│   ├── optimize_agent.py     # Optimize a single agent
│   ├── optimize_skill.py     # Optimize a single skill
│   ├── batch_optimize.py     # Batch optimization
│   ├── run_ab_test.py        # A/B testing
│   ├── verify_optimizations.py # Verification
│   ├── create_training_data.py # Generate training data
│   ├── deploy_optimized_prompts.py # Deploy optimized prompts
│   └── run_optimization.sh   # Shell runner
└── reports/                  # Optimization results
```

## What's Different From DSPy

| Aspect | DSPy | This Project |
|--------|------|-------------|
| Backend | OpenAI/Anthropic API | Claude Code CLI (full tool access) |
| Output format | Structured API returns | Multi-paragraph tool-use responses |
| Metrics | Generic (exact match, F1) | 15+ task-specific metrics |
| Optimization unit | Module parameters | Skill/agent prompt files |
| Deployment | In-memory | File-based (`.md` prompts) |
| Cross-validation | Standard k-fold | With dropout regularization |

## Requirements

- Python 3.10+
- [Claude Code](https://claude.ai/claude-code) CLI installed and authenticated
- No additional Python dependencies (pure standard library)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/AshitaOrbis/dspy-prompt-optimizer.git
cd dspy-prompt-optimizer

# Optimize a skill (requires Claude Code CLI)
python scripts/optimize_skill.py --target "my-skill" --algorithm bootstrap

# Run A/B test to compare base vs optimized
python scripts/run_ab_test.py --target "my-skill" --trials 10

# Verify optimization didn't overfit
python scripts/verify_optimizations.py --target "my-skill"
```

## Results

Deployed optimizations across 11 of 13 targets:

| Target | Algorithm | Improvement | Status |
|--------|-----------|-------------|--------|
| Code Reviews | Bootstrap | +18% severity accuracy | Deployed |
| Test Writing | Bootstrap | +22% coverage score | Deployed |
| Security Audits | COPRO | +15% CWE detection | Deployed |
| Debugging | Bootstrap | +20% root cause match | Deployed |
| PR Preparation | Iterative | +12% quality score | Deployed |
| Search Routing | Bootstrap | +25% routing accuracy | Deployed |
| Tool Selection | Bootstrap | +19% tier classification | Deployed |
| Plan Quality | COPRO | +14% multi-dim score | Deployed |
| Refactoring | Bootstrap | +16% smell-technique match | Deployed |
| Performance Analysis | Bootstrap | +21% pattern recognition | Deployed |
| Feature Implementation | Iterative | +11% completeness | Deployed |

## License

MIT
