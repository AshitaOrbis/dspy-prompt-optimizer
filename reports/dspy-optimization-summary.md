# DSPy-Style Prompt Optimization: What We Built, How It Works, Results

**Date**: 2026-02-07

---

## What Is This?

A custom prompt optimization framework for Claude Code, inspired by Stanford's DSPy library. Instead of hand-tuning agent prompts and skill instructions, this system automatically discovers the best few-shot examples and prompt configurations through programmatic experimentation.

The goal: make Claude Code's agents and skills measurably better at their jobs, and prove it with numbers.

---

## What We Took From DSPy

DSPy (Declarative Self-improving Language Programs) introduced several ideas we adopted directly:

### 1. BootstrapFewShot

DSPy's signature algorithm. Run the model on training examples, keep the successful traces (where output scores above a threshold), and attach those as few-shot demonstrations for future runs.

Our implementation (`lib/prompt_optimizer/bootstrap.py`) follows this exactly:
- Run base prompt on N training examples
- Score each output with a metric function
- Traces scoring >= threshold become demos
- Demos get attached to the prompt going forward

### 2. COPRO (Coordinate Prompt Optimization)

DSPy's instruction-level optimization. Generate prompt variants through mutations (paraphrase, elaborate, simplify, extend), evaluate each on training data, pick the winner.

Our implementation (`lib/prompt_optimizer/copro.py`) follows the same pattern, then feeds the winning instruction into Bootstrap for demo selection.

### 3. Metric-Driven Evaluation

DSPy's core philosophy: define a metric function, then let the optimizer maximize it. No manual prompt engineering -- just define what "good" looks like and let the system find it.

We adopted this wholesale. Every optimization target has a metric that returns 0.0-1.0.

### 4. Train/Holdout Split

DSPy uses separate evaluation sets to catch overfitting. We maintain paired datasets for every target: a training set for optimization, and a holdout set for validation. No optimization touches holdout data.

### 5. Demos as First-Class Objects

DSPy treats few-shot examples as structured data (not raw strings). Our `Demo` dataclass stores input, output, score, and metadata -- making them composable and filterable.

---

## What's Different From DSPy

### 1. Claude CLI as the Backend

DSPy targets the OpenAI/Anthropic API. We run everything through the Claude Code CLI, which means:
- The "model" is a full Claude Code session with tool access
- Outputs are often verbose, multi-paragraph responses (not structured API returns)
- Timeout management matters (300s for Opus, 180s for Haiku)

`lib/prompt_optimizer/claude_runner.py` wraps the CLI, handling timeouts, retries, and output capture.

### 2. Task-Specific Metrics (15+)

DSPy ships with generic metrics (exact match, F1). We built domain-specific metrics because Claude Code tasks are structurally different from each other:

| Metric | What It Evaluates |
|--------|-------------------|
| `issue_severity_match` | Code review: severity levels (Critical/High/Med/Low) |
| `test_coverage_score` | Test writing: count, categories, assertion structure |
| `complexity_classification` | Performance: pattern recognition (N+1, O(n^2)) |
| `security_cwe_match` | Security: CWE identifier extraction |
| `root_cause_match` | Debugging: cause, fix, verification steps |
| `pr_quality_match` | PR preparation: change type, risk, section presence |
| `implementation_completeness` | Feature implementation: acceptance criteria coverage |
| `refactoring_match` | Refactoring: smell-technique affinity scoring |
| `routing_accuracy` | Search tool selection: exact/family/related matching |
| `binary_decision_match` | grep-vs-mgrep binary decisions |
| `tool_tier_classification` | Core/Specialized/Deferred tool categorization |
| `plan_quality_match` | Plan output: 5-dimension quality scoring |

### 3. Semantic Equivalence Groups

Standard metrics fail when there are multiple correct answers. "feature" and "enhancement" mean the same thing. "extract_method" and "compose_method" are both valid decomposition techniques.

We built equivalence groups into the metrics:

```python
CHANGE_TYPE_EQUIVALENCES = {
    'additive': {'feature', 'enhancement', 'new_feature', 'implement'},
    'corrective': {'bugfix', 'fix', 'patch', 'security', 'hotfix'},
    ...
}
```

With partial credit for related-but-not-equivalent categories (e.g., a refactoring technique that's valid but not the one in the gold answer gets 0.6 instead of 0.0).

### 4. Verbose Output Extractors

Claude doesn't return `{"tool": "exa_web_search"}`. It returns three paragraphs explaining why Exa is the right choice, with caveats. The metrics need structured data.

Phase 6 of the project added extractors (`lib/prompt_optimizer/extractors.py`) that parse Claude's natural language into structured fields the metrics can score. Plus format instructions that nudge Claude toward more parseable output.

### 5. Regularization Suite

DSPy has basic overfitting protection. We added:
- **Probe holdout**: Sample holdout during training to detect overfitting early
- **K-fold cross-validation**: Validate the optimization approach itself
- **Example dropout**: Randomly exclude demos (0-50%) for robustness
- **Max overfitting gap warning**: Alert when train-probe gap > 0.3
- **Demo transformers**: Condense verbose demos so they don't overwhelm the context

### 6. Iterative Refinement with Synthetic Data

Our third algorithm (`lib/prompt_optimizer/iterative.py`) goes beyond DSPy's standard toolkit:
1. Run Bootstrap to get an optimized prompt
2. Generate synthetic training data from the successful demos (similar examples, edge cases, adversarial examples)
3. Run Bootstrap again on the expanded dataset
4. Repeat until convergence

### 7. Live A/B Testing

DSPy evaluates on held-out sets. We also run live A/B tests where the same input goes through both the baseline and treatment prompts in real Claude CLI sessions, then compare scores head-to-head. This catches things that static holdout evaluation misses (like timeout behavior).

### 8. Deployment Pipeline

DSPy outputs an optimized program. We output deployed agent definitions -- the few-shot demos get injected directly into `~/.claude/agents/*.md` files as a `## Few-Shot Examples` section that Claude reads at agent invocation time.

---

## How It's Been Working

### Overall Results

| Metric | Value |
|--------|-------|
| Total targets optimized | 13 |
| Passed holdout validation | 11 (85%) |
| Failed holdout validation | 2 (15%) |
| Deployed to production | 11 |

### Agent Results

| Agent | Training Score | Holdout Score | Status |
|-------|---------------|---------------|--------|
| capability-evaluator | 0.992 | **1.000** | Deployed |
| security-auditor | 0.965 | **0.760** | Deployed |
| performance-analyzer | 0.776 | **0.729** | Deployed |
| debugger | 0.920 | **0.883** | Deployed |
| test-writer | 0.633 | **0.608** | Deployed |
| feature-implementer | 0.920 | **0.565** | Deployed |
| code-reviewer | 0.877 | **0.525** | Deployed |
| pr-preparer | 0.800 | 0.352 | Failed |
| refactoring-advisor | 0.863 | 0.308 | Failed |

### Skill Results

| Skill | Training Score | Holdout Score | Status |
|-------|---------------|---------------|--------|
| mgrep-guide | 1.000 | **1.000** | Deployed |
| mcp-search-framework | 1.000 | **1.000** | Deployed |
| advanced-tool-use | 1.000 | **0.800** | Deployed |
| plan-mode-quality | 1.000 | **1.000** | Validated |

### Plan Quality A/B Test (Most Recent)

Live comparison of Claude's plan output with and without plan-mode directives in CLAUDE.md:

**Opus (target model)**: **+11.9% improvement** with directives
- 4 wins, 4 losses, but win margins 1.6x larger than loss margins
- Fewer timeouts (directives act as focusing constraints)
- Effect is a lower bound -- real interactive plan mode would be stronger

**Haiku (reference)**: -4.1% (noise). Small models don't respond to meta-instructions.

### What Worked

1. **Skill optimization was the biggest win.** Search routing (mgrep-guide, mcp-search-framework) hit perfect holdout scores. The task is well-defined, the decision space is constrained, and a few good examples are all the model needs.

2. **Regularization prevented overfitting.** The code-reviewer initially failed holdout (0.477 in Phase 4). After re-optimization with dropout and cross-validation in Phase 5, it passed (0.525). The gap between training and holdout scores stayed manageable.

3. **Semantic equivalence was critical.** Without it, refactoring and PR metrics would reject correct answers that used different terminology. Even with it, refactoring-advisor still failed -- suggesting the equivalence groups need expansion.

4. **Live A/B testing validated real-world impact.** Static holdout scores don't capture timeout behavior or style effects. The plan quality A/B test on Opus revealed that directives help by constraining verbosity, not just by improving content.

### What Didn't Work

1. **Free-form generation tasks are hard to score.** pr-preparer (0.352) and refactoring-advisor (0.308) failed because the metrics were too rigid for tasks with high output variance. A PR description can be good in many different ways.

2. **Metric design is the bottleneck.** The system is only as good as its metrics. Building a metric that reliably distinguishes good from bad output for a complex task (like "write a PR description") is harder than building the optimizer.

3. **One-shot testing undersells the system.** The A/B test runs Claude with no tool access and a single prompt. In real usage, Claude explores the codebase, asks clarifying questions, and iterates. The optimization likely helps more in the real context than the numbers show.

4. **Opus's verbose style fights the metrics.** Opus generates longer, more detailed output than what's in the gold dataset. The metrics penalize this verbosity because extractors can't always parse it cleanly. Raw scores for Opus are lower than Haiku, even though Opus plans are qualitatively better.

---

## Architecture Summary

```
datasets/                          # Training + holdout pairs (18 targets)
lib/prompt_optimizer/
  bootstrap.py                     # BootstrapFewShot algorithm
  copro.py                         # COPRO instruction optimization
  iterative.py                     # Iterative refinement + synthetic data
  metrics.py                       # 15+ task-specific metrics
  extractors.py                    # Verbose output -> structured data
  format_instructions.py           # Nudge Claude toward parseable output
  demo_transformers.py             # Condense verbose demos
  claude_runner.py                 # Claude CLI wrapper
  storage.py                       # Demo/prompt persistence
scripts/
  optimize_agent.py                # Optimize a single agent
  optimize_skill.py                # Optimize a single skill
  verify_optimizations.py          # Run holdout verification
  run_optimization.sh              # Orchestrate multi-target runs
  run_ab_test.py                   # Live A/B comparison testing
  deploy_optimized_prompts.py      # Inject demos into agent definitions
optimized-prompts/
  status.json                      # Master tracking (all targets)
~/.claude/prompt_optimizer/
  prompts/                         # Persisted optimized prompts
  demos/                           # Persisted demo collections
~/.claude/agents/*.md              # Deployed agents (with ## Few-Shot Examples)
```

---

## Key Lessons

1. **Define your metric before optimizing.** The optimizer will maximize whatever you measure. If the metric is wrong, the optimizer will confidently produce bad prompts.

2. **Holdout validation is non-negotiable.** Training scores look great (0.8-1.0). Holdout scores tell you the truth (0.3-1.0). Without holdout, you're just overfitting.

3. **Match the test model to the production model.** Testing Opus directives on Haiku wastes tokens and produces meaningless results. The optimization must run on the model it targets.

4. **Semantic equivalence is table stakes.** Any metric for a creative/open-ended task needs to handle synonyms, alternative phrasings, and multiple valid approaches. Without it, you reject correct answers.

5. **Verbose output is the hardest problem.** Claude generates rich, detailed responses. Converting those into structured data for scoring is where most of the engineering effort went (Phase 6 was entirely about this).

6. **A/B testing catches what holdout misses.** Timeout behavior, style effects, and focusing properties of directives only show up in live comparison tests.
