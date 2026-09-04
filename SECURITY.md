# Security Model

This document describes the security posture of `dspy-prompt-optimizer`, what
threats it defends against, and — importantly — the residual risks that are
**inherent** to what the tool does and cannot be "fixed" without changing it
into a different tool.

A security review (2026-06-11, archived under `reviews/`) drove the hardening
described here. Read this before running the optimizer against any dataset,
prompt output, or project directory you do not fully trust.

## Threat model in one sentence

The optimizer runs model CLIs **non-interactively** over training data and
treats model output as a signal to attach few-shot examples to your agent/skill
prompts. Both the training data and the model output are therefore *untrusted
input that flows into files you later run as agent instructions.*

## What is now defended (hardened)

| Risk | Mitigation |
|------|------------|
| **Shell injection in background mode** | `run_optimization.sh` no longer serializes its own source into `bash -c "...$VAR..."`. Background runs re-exec the script in `--foreground` mode with a proper argv array; no user value is ever evaluated as shell code. All enum/name/number arguments are validated against allowlists before use. |
| **Path traversal / arbitrary file overwrite** | All agent/skill/target names are validated (`lib/prompt_optimizer/validation.py`: `validate_name`) and every derived path is containment-checked (`contained_path`) so it cannot escape its intended directory. Applies to deploy, storage, utils, batch, and the `optimize_agent.py` / `optimize_skill.py` CLI entry points. |
| **Destructive deploy / unrecoverable backup** | `deploy_optimized_prompts.py` backs up the *original* file bytes (not the post-mutation content) before writing, refuses to follow symlinks, and writes atomically via `os.replace`. |
| **Secret leakage via process arguments** | Claude and Codex user prompts are passed on **stdin**; agy/Gemini is configured for the provisional NDJSON stdin channel documented below. `ClaudeRunner.run_with_system` puts the system prompt in a mode-0600 temporary file and passes only its path. Prompt text therefore never appears in runner argv (world-readable via `/proc` on many systems). Backend errors redact prompt/context/system text, child-environment values, and common secret formats. Model **stdout** is intentionally not redacted because demos are model output by design — review it with `--dry-run` before deploying. |
| **Markdown fence breakout (persistent prompt injection)** | Demo text is wrapped in dynamically-sized backtick fences so embedded ``` cannot break out and become live instruction text in a deployed agent. Demo fields are type-validated. |
| **PATH hijack of model CLIs / `python3`** | Claude, Codex, and agy executables are resolved before each runner starts; shell helpers resolve `python3` with `command -v` rather than relying on later ambient `PATH` resolution. |
| **Unbounded cost / memory** | Target count is capped (`MAX_TARGETS`, default 25). Every model backend uses the same timeout validator: non-positive or greater-than-1800-second values are rejected and positive values below five seconds are clamped to five. Stdout and stderr are drained concurrently into independent byte-bounded buffers. When either stream crosses `PROMPT_OPTIMIZER_MAX_OUTPUT_BYTES` (default 5 MB), the child is terminated, then killed if it does not exit; the returned error names every truncated stream. Timeout enforcement remains active during streaming capture. |
| **Backend parser drift / banner contamination** | The agy runner requests `--input-format=stream-json --output-format=stream-json` and the parser accepts only the provisionally expected terminal `result` event. Malformed streams and streams without that event fail closed. The exact live agy envelope is not established offline and is explicitly deferred below. |
| **Status-file injection** | `status.json` is generated with `jq` (every value JSON-encoded), not `sed` string surgery. |
| **Crash on malformed model output** | Extractors and the deploy/storage loaders type-guard untrusted JSON instead of assuming strings/numbers. |

## Inherent residual risk — read this

### Deferred: live agy stream-json contract

The current runner and mocked tests assert this provisional request schema:
one UTF-8 NDJSON line shaped as
`{"type":"user","message":"<context>\n\n<prompt>"}`, with no prompt in argv,
sent under `--input-format=stream-json --output-format=stream-json`. The parser
expects NDJSON objects and returns only a terminal
`{"type":"result","result":"..."}` event whose `is_error` value is not true.
Those mocks verify repository behavior; they do **not** verify agy's real wire
contract.

Offline inspection of the installed agy 1.1.19 verified that `--help` documents
NDJSON stdin and requires stream-json output, while its bundled changelog names
typed `init`, `step_update`, and terminal `result` output events. No vendored
request/response fixture was found. Every live probe in the no-network sandbox
failed while opening agy's localhost language-server socket, before stdin could
be parsed.

A network-enabled run that permits agy's localhost socket must execute the
following contract probe. Pipeline success plus the `jq` assertion establishes
that the mocked request envelope was accepted, every stdout line was JSON (no
banner/preamble contamination), and the terminal result schema matched:

```bash
set -euo pipefail
agy_contract_output="$(mktemp)"
trap 'rm -f "$agy_contract_output"' EXIT
printf '%s\n' '{"type":"user","message":"Reply with exactly AGY_CONTRACT_20260823."}' \
  | agy --model gemini-3.1-pro --print-timeout 30s \
      --input-format=stream-json --output-format=stream-json --print= \
      > "$agy_contract_output"
jq -se '
  all(.[]; type == "object") and
  any(.[];
    .type == "result" and
    .result == "AGY_CONTRACT_20260823" and
    ((.is_error // false) == false)
  )
' "$agy_contract_output"
```

Until that succeeds, the risk is that the mocked input keys are wrong: agy may
reject the request, ignore or misroute prompt text, or change output framing.
The parser should fail closed rather than accept banner text, but the Gemini
backend cannot be claimed operationally compatible and `bq-1283` remains
**DEFERRED**.

### Opt-in permission automation and prompt-injection risk

Permission bypass/automation is **disabled by default for every backend**. Set
`PROMPT_OPTIMIZER_SKIP_PERMISSIONS=1` only when unattended tool execution is
required. That opt-in adds `--dangerously-skip-permissions` for Claude and agy,
or Codex's sandbox-preserving `--approve-for-me` automation. Without the opt-in,
a backend can deny, fail, or wait when a model requests a gated tool.

The opt-in consequence is real: if training data, prompt context, or prior
optimized demos contain adversarial instructions (for example, instructions to
read local credentials), the model may be able to execute tools without a human
permission decision. In a directory containing secrets, credentials, tokens,
or `.env` files, that is a capability-boundary break.

**When opting into permission automation, mitigate it operationally:**

1. **Run only trusted datasets.** Treat every training example and every demo as
   code that could run on your machine.
2. **Run in a sandbox** — a throwaway user/container/VM with **no secrets**, no
   cloud credentials, and ideally no network.
3. **Scrub the environment**: set `PROMPT_OPTIMIZER_SCRUB_ENV=1` so every model
   subprocess receives only the shared minimal allowlist instead of inheriting
   exported secrets. Add specific pass-through variables with
   `PROMPT_OPTIMIZER_ENV_PASSTHROUGH=VAR1,VAR2` only when authentication needs
   them.
4. **Keep permission automation off** (the default). Set
   `PROMPT_OPTIMIZER_SKIP_PERMISSIONS=1` only inside the isolated environment
   described above and only when the run genuinely needs unattended tools.
5. **Review deployed demos.** Run `deploy_optimized_prompts.py --dry-run` and
   read the diff before injecting model-generated examples into agents you run
   with elevated permissions.

### Heuristic extraction is a quality signal, not a security gate

The extractors in `lib/prompt_optimizer/extractors.py` use layered heuristics
(explicit patterns first, keyword-density last resort) to pull a tool name,
severity count, or score out of verbose model output. These are inherently
fuzzy: a verbose or adversarial answer can be mis-parsed. They are appropriate
for *ranking demo quality* but should **not** be relied upon as a security or
correctness gate. If you need hard guarantees, have targets emit structured
JSON and validate it. (This is a documented design limitation, not a bug.)

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `PROMPT_OPTIMIZER_TIMEOUT` | `180` | Claude per-call timeout. Non-positive and >1800 values are rejected; positive values below 5 are clamped to 5. Explicit Codex/agy constructor timeouts use the same policy. |
| `PROMPT_OPTIMIZER_MAX_OUTPUT_BYTES` | `5000000` | Independent streaming cap for stdout and stderr. Crossing either cap terminates the child and returns a visible truncation error. |
| `PROMPT_OPTIMIZER_SKIP_PERMISSIONS` | unset (off) | Set `1` to opt into backend permission automation. Claude/agy use `--dangerously-skip-permissions`; Codex uses `--approve-for-me` with its workspace-write sandbox. |
| `PROMPT_OPTIMIZER_SCRUB_ENV` | unset | Set `1` to pass the shared minimal allowlist to every model backend. |
| `PROMPT_OPTIMIZER_ENV_PASSTHROUGH` | unset | Comma-separated extra env vars to allow when scrubbing. |
| `MAX_TARGETS` | `25` | Max targets per `run_optimization.sh` invocation. |

## Reporting

Found something? Open an issue (omit any secret/PoC payload from public reports).
