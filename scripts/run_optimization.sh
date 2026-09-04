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
#   --allow-unverified  Deprecated compatibility flag; never marks ungated work complete
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
ALLOW_UNVERIFIED=false

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
    # Tier 1
    ["pr-preparer"]="pr-preparations"
    ["refactoring-advisor"]="refactoring-decisions"
    # Tier 2
    ["capability-discoverer"]="discoveries"
    ["fact-checker"]="fact-checks"
    ["web-researcher"]="web-research"
    # Tier 3
    ["writing-review"]="writing-reviews"
    ["session-handoff"]="session-handoffs"
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
    # Tier 1
    ["pr-preparer"]="agent"
    ["refactoring-advisor"]="agent"
    # Tier 2
    ["capability-discoverer"]="agent"
    ["fact-checker"]="agent"
    ["web-researcher"]="agent"
    # Tier 3
    ["writing-review"]="skill"
    ["session-handoff"]="skill"
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
        --allow-unverified)
            ALLOW_UNVERIFIED=true
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

# --- Input validation (defense against injection / path traversal / cost abuse) ---
# Enumerated options are checked against an explicit allowlist; free-form names
# are restricted to a safe character class. This runs BEFORE any value is used
# to build a command, a path, or a status file.
MAX_TARGETS="${MAX_TARGETS:-25}"

if [[ ! "$ALGORITHM" =~ ^(bootstrap|copro|iterative)$ ]]; then
    echo "Error: invalid --algorithm '$ALGORITHM' (allowed: bootstrap|copro|iterative)" >&2
    exit 2
fi
if [[ ! "$MODEL" =~ ^(haiku|sonnet|opus)$ ]]; then
    echo "Error: invalid --model '$MODEL' (allowed: haiku|sonnet|opus)" >&2
    exit 2
fi
if [[ ! "$CV_FOLDS" =~ ^[0-9]+$ ]]; then
    echo "Error: --cv-folds must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$DROPOUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: --dropout must be a number" >&2
    exit 2
fi
# Targets: comma-separated list of validated identifiers. Reject anything that
# could break out of a shell word, a path, or JSON.
if [[ ! "$TARGETS" =~ ^[A-Za-z0-9_.,-]+$ ]]; then
    echo "Error: --targets contains illegal characters (allowed: letters, digits, _ . - and ',')" >&2
    exit 2
fi
# Per-target checks: no leading dash/dot, no '..', enforce count cap.
IFS=',' read -ra _validate_targets <<< "$TARGETS"
if (( ${#_validate_targets[@]} > MAX_TARGETS )); then
    echo "Error: too many targets (${#_validate_targets[@]} > MAX_TARGETS=$MAX_TARGETS)" >&2
    exit 2
fi
for _t in "${_validate_targets[@]}"; do
    if [[ -z "$_t" || "$_t" == -* || "$_t" == .* || "$_t" == *..* || ! "$_t" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "Error: invalid target name: '$_t'" >&2
        exit 2
    fi
done
unset _validate_targets _t

# Pin the python interpreter at startup to avoid PATH-hijack of a relative
# 'python3' picked up from a world-writable or project-local directory.
PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python3 not found on PATH" >&2
    exit 2
fi

# Script directory for relative imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Encode a bash array as a JSON array string using jq (safe for any content).
to_json_array() {
    if (( $# == 0 )); then
        echo '[]'
        return
    fi
    printf '%s\n' "$@" | jq -R . | jq -s -c .
}

# Write status as JSON. Uses jq to encode every value so that target names
# containing quotes, backslashes, or control characters cannot corrupt or inject
# into status.json (the old sed-based string surgery was injection-prone).
write_status() {
    local current="$1"
    local progress="$2"
    local completed=("${!3}")
    local failed=("${!4}")
    local unverified=("${!5}")
    local scores="${6:-{}}"

    # Validate scores is well-formed JSON; fall back to {} if not.
    if ! echo "$scores" | jq -e . >/dev/null 2>&1; then
        scores="{}"
    fi

    local targets_arr
    IFS=',' read -ra targets_arr <<< "$TARGETS"

    local targets_json completed_json failed_json unverified_json
    targets_json="$(to_json_array "${targets_arr[@]}")"
    completed_json="$(to_json_array ${completed[@]+"${completed[@]}"})"
    failed_json="$(to_json_array ${failed[@]+"${failed[@]}"})"
    unverified_json="$(to_json_array ${unverified[@]+"${unverified[@]}"})"

    jq -n \
        --arg started "$(date -Iseconds)" \
        --arg current "$current" \
        --arg progress "$progress" \
        --argjson scores "$scores" \
        --argjson targets "$targets_json" \
        --argjson completed "$completed_json" \
        --argjson failed "$failed_json" \
        --argjson unverified "$unverified_json" \
        '{
            started: $started,
            targets: $targets,
            completed: $completed,
            failed: $failed,
            unverified: $unverified,
            current: $current,
            progress: $progress,
            scores: $scores
        }' > "$OUTPUT_DIR/status.json.tmp" \
        && mv "$OUTPUT_DIR/status.json.tmp" "$OUTPUT_DIR/status.json"
}

# Main optimization function
run_optimization() {
    local targets_arr
    IFS=',' read -ra targets_arr <<< "$TARGETS"
    local total=${#targets_arr[@]}
    local completed=()
    local failed=()
    local unverified=()
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
        write_status "$target" "$idx/$total" completed[@] failed[@] unverified[@] "$scores"

        echo ""
        echo "======================================"
        echo "[$(date -Iseconds)] Processing: $target ($idx/$total)"
        echo "======================================"

        # Get dataset name
        local dataset="${DATASET_MAP[$target]:-$target}"
        local dataset_path="$DATASETS_DIR/${dataset}.jsonl"
        local holdout_path="$DATASETS_DIR/${dataset}-holdout.jsonl"

        # Get target type
        local target_type="${TARGET_TYPE[$target]:-agent}"

        if [[ ! -f "$dataset_path" ]]; then
            echo "[ERROR] Dataset not found: $dataset_path"
            failed+=("$target")
            continue
        fi

        if [[ ! -f "$holdout_path" ]]; then
            echo "[UNVERIFIED] Holdout not found: $holdout_path"
            if [[ "$ALLOW_UNVERIFIED" == "true" ]]; then
                echo "[UNVERIFIED] --allow-unverified cannot bypass the promotion gate"
            fi
            unverified+=("$target")
            continue
        fi

        # Build command based on target type using batch_optimize.py
        local cmd=("$PYTHON_BIN" "$SCRIPT_DIR/batch_optimize.py")

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
        if [[ -f "$holdout_path" ]]; then
            # The batch CLI stages the candidate and performs the binding gate.
            # Its exit status therefore covers optimization and validation.
            cmd+=(--holdout-gate)
            cmd+=(--holdout-dir "$DATASETS_DIR")
        fi

        echo "Command: ${cmd[*]}"
        echo ""

        # Run optimization
        if "${cmd[@]}"; then
            echo ""
            echo "[SUCCESS] Optimized and holdout-gated: $target"
            completed+=("$target")
        else
            echo ""
            echo "[FAILED] Could not optimize: $target"
            failed+=("$target")
        fi
    done

    # Final status: reserve "complete" and OPTIMIZATION_COMPLETE for a fully
    # gated run. Failed and unverified targets get explicit terminal states.
    local final_state final_label
    if [[ ${#failed[@]} -eq 0 && ${#unverified[@]} -eq 0 ]]; then
        final_state="complete"
        final_label="OPTIMIZATION COMPLETE"
    elif [[ ${#completed[@]} -eq 0 && ${#failed[@]} -gt 0 ]]; then
        final_state="failed"
        final_label="OPTIMIZATION FAILED"
    elif [[ ${#completed[@]} -eq 0 && ${#unverified[@]} -gt 0 ]]; then
        final_state="unverified"
        final_label="OPTIMIZATION UNVERIFIED"
    else
        final_state="partial"
        final_label="OPTIMIZATION PARTIAL"
    fi
    write_status "$final_state" "$total/$total" completed[@] failed[@] unverified[@] "$scores"

    echo ""
    echo "======================================"
    echo "[$(date -Iseconds)] $final_label"
    echo "======================================"
    echo "Completed: ${#completed[@]}/${total}"
    echo "Failed: ${#failed[@]}/${total}"
    echo "Unverified: ${#unverified[@]}/${total}"
    echo ""

    # Terminal signal
    if [[ ${#failed[@]} -eq 0 && ${#unverified[@]} -eq 0 ]]; then
        echo "TERMINAL_SIGNAL: OPTIMIZATION_COMPLETE"
    elif [[ ${#completed[@]} -eq 0 && ${#unverified[@]} -gt 0 && ${#failed[@]} -eq 0 ]]; then
        echo "TERMINAL_SIGNAL: OPTIMIZATION_UNVERIFIED"
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
$(for t in "${unverified[@]}"; do echo "| $t | ⚠️ Unverified | Missing holdout; not optimized |"; done)

## Summary
- **Completed**: ${#completed[@]}/$total
- **Failed**: ${#failed[@]}/$total
- **Unverified**: ${#unverified[@]}/$total

## Next Steps
$(if [[ ${#failed[@]} -gt 0 ]]; then echo "- Review failed targets in optimization.log"; fi)
- Run \`verify_optimizations.py\` to check holdout scores
- Deploy successful optimizations with \`deploy_optimized_prompts.py\`
EOF

    echo ""
    echo "Summary written to: $OUTPUT_DIR/summary.md"

    if [[ ${#failed[@]} -gt 0 ]]; then
        return 1
    fi
    if [[ ${#unverified[@]} -gt 0 ]]; then
        return 2
    fi
    return 0
}

# Run in background or foreground
cd "$PROJECT_DIR"

if [[ "$FOREGROUND" == "true" ]]; then
    run_optimization 2>&1 | tee "$OUTPUT_DIR/optimization.log"
else
    echo "Starting background optimization..."
    echo "Logs: $OUTPUT_DIR/optimization.log"
    echo "Status: $OUTPUT_DIR/status.json"

    # SECURITY: Do NOT serialize this script's source with interpolated values
    # into `bash -c "..."`. The previous implementation embedded $TARGETS et al.
    # directly into a child shell program string, allowing a single quote in any
    # argument to break out and execute arbitrary commands. Instead, re-exec this
    # same script in --foreground mode with a proper argv array: every value is a
    # distinct, non-evaluated argument, so no shell metacharacter can escape.
    bg_cmd=(
        "$BASH" "${BASH_SOURCE[0]}"
        --foreground
        --targets "$TARGETS"
        --algorithm "$ALGORITHM"
        --model "$MODEL"
        --cv-folds "$CV_FOLDS"
        --dropout "$DROPOUT"
        --output-dir "$OUTPUT_DIR"
        --datasets-dir "$DATASETS_DIR"
    )
    if [[ "$ALLOW_UNVERIFIED" == "true" ]]; then
        bg_cmd+=(--allow-unverified)
    fi
    nohup "${bg_cmd[@]}" > "$OUTPUT_DIR/optimization.log" 2>&1 &

    echo $! > "$OUTPUT_DIR/pid"
    echo "PID: $(cat "$OUTPUT_DIR/pid")"
    echo ""
    echo "Monitor with:"
    echo "  tail -f $OUTPUT_DIR/optimization.log"
    echo "  cat $OUTPUT_DIR/status.json | jq ."
fi
