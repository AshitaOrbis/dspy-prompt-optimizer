#!/usr/bin/env bash
#
# Background optimization runner with status tracking
#
# Usage:
#   ./run_optimization.sh --targets "mgrep-guide,mcp-search-framework" [OPTIONS]
#
# Options:
#   --targets       Comma-separated list of targets (required)
#   --algorithm     bootstrap|copro|iterative (default: bootstrap)
#   --model         haiku|sonnet|opus (default: sonnet)
#   --cv-folds      Cross-validation folds (default: 3)
#   --dropout       Example dropout rate (default: 0.2)
#   --output-dir    Output directory (default: optimized-prompts)
#   --datasets-dir  Datasets directory (default: datasets)
#   --foreground    Run in foreground instead of background
#
# Status tracking:
#   - Writes status to {output_dir}/status.json
#   - Logs to {output_dir}/optimization.log
#   - PID stored in {output_dir}/pid
#

set -euo pipefail

# Defaults
TARGETS=""
ALGORITHM="bootstrap"
MODEL="sonnet"
CV_FOLDS=3
DROPOUT=0.2
OUTPUT_DIR="optimized-prompts"
DATASETS_DIR="datasets"
FOREGROUND=false

# Target -> dataset mapping
declare -A DATASET_MAP=(
    ["mcp-search-framework"]="search-routing"
    ["mgrep-guide"]="search-decisions"
    ["advanced-tool-use"]="tool-selection"
    ["code-reviewer"]="code-reviews"
    ["test-writer"]="test-suites"
    ["security-auditor"]="security-audits"
    ["performance-analyzer"]="performance-analysis"
    ["capability-evaluator"]="evaluations"
)

# Target -> metric mapping
declare -A METRIC_MAP=(
    ["mcp-search-framework"]="routing"
    ["mgrep-guide"]="binary_decision"
    ["advanced-tool-use"]="tool_tier"
    ["code-reviewer"]="issue_severity"
    ["test-writer"]="test_coverage"
    ["security-auditor"]="security_cwe"
    ["performance-analyzer"]="complexity"
    ["capability-evaluator"]="evaluation_score"
)

# Target type (agent or skill)
declare -A TARGET_TYPE=(
    ["mcp-search-framework"]="skill"
    ["mgrep-guide"]="skill"
    ["advanced-tool-use"]="skill"
    ["code-reviewer"]="agent"
    ["test-writer"]="agent"
    ["security-auditor"]="agent"
    ["performance-analyzer"]="agent"
    ["capability-evaluator"]="agent"
)

usage() {
    head -30 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --targets)
            TARGETS="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --cv-folds)
            CV_FOLDS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --datasets-dir)
            DATASETS_DIR="$2"
            shift 2
            ;;
        --foreground)
            FOREGROUND=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Validate required args
if [[ -z "$TARGETS" ]]; then
    echo "Error: --targets is required" >&2
    usage
fi

# Script directory for relative imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Write initial status
write_status() {
    local current="$1"
    local progress="$2"
    local completed=("${!3}")
    local failed=("${!4}")
    local scores="${5:-{}}"

    cat > "$OUTPUT_DIR/status.json" << EOF
{
  "started": "$(date -Iseconds)",
  "targets": [$(echo "$TARGETS" | sed 's/,/", "/g' | sed 's/^/"/' | sed 's/$/"/')],
  "completed": [$(printf '"%s", ' "${completed[@]}" 2>/dev/null | sed 's/, $//')],
  "failed": [$(printf '"%s", ' "${failed[@]}" 2>/dev/null | sed 's/, $//')],
  "current": "$current",
  "progress": "$progress",
  "scores": $scores
}
EOF
}

# Main optimization function
run_optimization() {
    local targets_arr
    IFS=',' read -ra targets_arr <<< "$TARGETS"
    local total=${#targets_arr[@]}
    local completed=()
    local failed=()
    local scores="{}"
    local idx=0

    echo "[$(date -Iseconds)] Starting optimization for $total targets"
    echo "Targets: ${targets_arr[*]}"
    echo "Algorithm: $ALGORITHM"
    echo "Model: $MODEL"
    echo "CV Folds: $CV_FOLDS"
    echo "Dropout: $DROPOUT"
    echo "---"

    for target in "${targets_arr[@]}"; do
        idx=$((idx + 1))
        write_status "$target" "$idx/$total" completed[@] failed[@] "$scores"

        echo ""
        echo "======================================"
        echo "[$(date -Iseconds)] Processing: $target ($idx/$total)"
        echo "======================================"

        # Get dataset name
        local dataset="${DATASET_MAP[$target]:-$target}"
        local dataset_path="$DATASETS_DIR/${dataset}.jsonl"
        local holdout_path="$DATASETS_DIR/${dataset}-holdout.jsonl"

        # Get metric
        local metric="${METRIC_MAP[$target]:-score_similarity}"

        # Get target type
        local target_type="${TARGET_TYPE[$target]:-agent}"

        if [[ ! -f "$dataset_path" ]]; then
            echo "[ERROR] Dataset not found: $dataset_path"
            failed+=("$target")
            continue
        fi

        # Build command based on target type using batch_optimize.py
        local cmd=(python3 "$SCRIPT_DIR/batch_optimize.py")

        if [[ "$target_type" == "skill" ]]; then
            cmd+=(--skills "$target")
        else
            cmd+=(--agents "$target")
        fi

        cmd+=(--data-dir "$DATASETS_DIR")
        cmd+=(-o "$OUTPUT_DIR")
        cmd+=(--model "$MODEL")
        cmd+=(--algorithm "$ALGORITHM")
        cmd+=(--threshold 0.6)

        echo "Command: ${cmd[*]}"
        echo ""

        # Run optimization
        if "${cmd[@]}"; then
            echo ""
            echo "[SUCCESS] Optimized: $target"
            completed+=("$target")

            # Run holdout validation if available
            if [[ -f "$holdout_path" ]]; then
                echo "[$(date -Iseconds)] Running holdout validation..."

                local holdout_cmd=(python3 "$SCRIPT_DIR/verify_optimizations.py")

                # Use --skill or --agent based on target type
                if [[ "$target_type" == "skill" ]]; then
                    holdout_cmd+=(--skill "$target")
                else
                    holdout_cmd+=(--agent "$target")
                fi

                holdout_cmd+=(--holdout "$holdout_path")
                holdout_cmd+=(--model "haiku")  # Use cheaper model for validation
                holdout_cmd+=(--pass-threshold 0.5)

                if "${holdout_cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/optimization.log"; then
                    echo "[SUCCESS] Holdout validation passed"
                else
                    echo "[WARNING] Holdout validation had issues"
                fi
            fi
        else
            echo ""
            echo "[FAILED] Could not optimize: $target"
            failed+=("$target")
        fi
    done

    # Final status
    write_status "complete" "$total/$total" completed[@] failed[@] "$scores"

    echo ""
    echo "======================================"
    echo "[$(date -Iseconds)] OPTIMIZATION COMPLETE"
    echo "======================================"
    echo "Completed: ${#completed[@]}/${total}"
    echo "Failed: ${#failed[@]}/${total}"
    echo ""

    # Terminal signal
    if [[ ${#failed[@]} -eq 0 ]]; then
        echo "TERMINAL_SIGNAL: OPTIMIZATION_COMPLETE"
    elif [[ ${#completed[@]} -eq 0 ]]; then
        echo "TERMINAL_SIGNAL: OPTIMIZATION_FAILED"
    else
        echo "TERMINAL_SIGNAL: PARTIAL_SUCCESS"
    fi

    # Generate summary
    cat > "$OUTPUT_DIR/summary.md" << EOF
# Optimization Results

Generated: $(date -Iseconds)

## Configuration
| Parameter | Value |
|-----------|-------|
| Algorithm | $ALGORITHM |
| Model | $MODEL |
| CV Folds | $CV_FOLDS |
| Dropout | $DROPOUT |

## Results

| Target | Status | Notes |
|--------|--------|-------|
$(for t in "${completed[@]}"; do echo "| $t | ✅ Passed | Optimized |"; done)
$(for t in "${failed[@]}"; do echo "| $t | ❌ Failed | See logs |"; done)

## Summary
- **Completed**: ${#completed[@]}/$total
- **Failed**: ${#failed[@]}/$total

## Next Steps
$(if [[ ${#failed[@]} -gt 0 ]]; then echo "- Review failed targets in optimization.log"; fi)
- Run \`verify_optimizations.py\` to check holdout scores
- Deploy successful optimizations with \`deploy_optimized_prompts.py\`
EOF

    echo ""
    echo "Summary written to: $OUTPUT_DIR/summary.md"
}

# Run in background or foreground
cd "$PROJECT_DIR"

if [[ "$FOREGROUND" == "true" ]]; then
    run_optimization 2>&1 | tee "$OUTPUT_DIR/optimization.log"
else
    echo "Starting background optimization..."
    echo "Logs: $OUTPUT_DIR/optimization.log"
    echo "Status: $OUTPUT_DIR/status.json"

    nohup bash -c "
        cd '$PROJECT_DIR'
        $(declare -f run_optimization)
        $(declare -f write_status)
        $(declare -p DATASET_MAP)
        $(declare -p METRIC_MAP)
        $(declare -p TARGET_TYPE)
        TARGETS='$TARGETS'
        ALGORITHM='$ALGORITHM'
        MODEL='$MODEL'
        CV_FOLDS='$CV_FOLDS'
        DROPOUT='$DROPOUT'
        OUTPUT_DIR='$OUTPUT_DIR'
        DATASETS_DIR='$DATASETS_DIR'
        SCRIPT_DIR='$SCRIPT_DIR'
        run_optimization
    " > "$OUTPUT_DIR/optimization.log" 2>&1 &

    echo $! > "$OUTPUT_DIR/pid"
    echo "PID: $(cat "$OUTPUT_DIR/pid")"
    echo ""
    echo "Monitor with:"
    echo "  tail -f $OUTPUT_DIR/optimization.log"
    echo "  cat $OUTPUT_DIR/status.json | jq ."
fi
