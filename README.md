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
    max_demos=5
)
result = optimizer.optimize(
    base_prompt="Your agent/skill prompt",
    training_data=examples,
    metric_fn=your_metric,
    threshold=0.7
)
```

### COPRO (Coordinate Prompt Optimization)

Instruction-level optimization. Generate prompt variants through mutations (paraphrase, elaborate, simplify, extend), evaluate each on training data, pick the winner.

```python
from prompt_optimizer import COPROOptimizer, ClaudeRunner

optimizer = COPROOptimizer(
    runner=ClaudeRunner(model="haiku"),
    n_variants=3,
    eval_subset_size=10
)
result = optimizer.optimize(
    base_prompt="Your agent/skill prompt",
    training_data=examples,
    metric_fn=your_metric
)
```

### Iterative Optimization

Runs Bootstrap across multiple rounds, expands the training set with synthetic examples, and stops when scores converge.

```python
from prompt_optimizer import IterativeOptimizer, ClaudeRunner

optimizer = IterativeOptimizer(
    runner=ClaudeRunner(model="haiku"),
    max_rounds=3,
    convergence_threshold=0.01
)
result = optimizer.optimize(
    base_prompt="Your agent/skill prompt",
    training_data=examples,
    metric_fn=your_metric,
    threshold=0.7,
    max_demos=5
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

## Security

This tool runs the Claude Code CLI non-interactively over your training data and
treats model output as a signal to modify your agent/skill prompts. **Both the
data and the output are untrusted input.** Run only trusted datasets, preferably
in a secret-free sandbox.

A security review hardened the optimizer against shell injection, path
traversal, destructive deploys, secret leakage, and prompt-fence breakout. One
risk is **inherent** and documented honestly rather than fake-fixed: batch
operation requires `--dangerously-skip-permissions`, so adversarial training
data can drive tool execution. See [`SECURITY.md`](SECURITY.md) for the full
threat model, mitigations, and the environment variables that tighten the
sandbox (`PROMPT_OPTIMIZER_SCRUB_ENV`, `PROMPT_OPTIMIZER_SKIP_PERMISSIONS`).

## License

MIT
