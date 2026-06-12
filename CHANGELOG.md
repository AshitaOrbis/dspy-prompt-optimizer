# Changelog

## 0.5.0 — 2026-06-11 — Security hardening

Driven by an external security review (archived in `reviews/`). See
[`SECURITY.md`](SECURITY.md) for the full threat model.

### Fixed (P0)
- **Background-mode shell injection** in `scripts/run_optimization.sh`: the
  background path no longer serializes the script's own source into
  `bash -c "...$VAR..."`. It re-execs in `--foreground` mode with an argv array,
  so no argument can break out and execute commands. Added allowlist/character
  validation for all CLI arguments.
- **Path traversal / arbitrary `.md` overwrite** in
  `scripts/deploy_optimized_prompts.py`: agent names are validated and paths are
  containment-checked; symlinked targets are refused.
- **Unrecoverable backup + non-atomic write** in deploy: the *original* bytes are
  backed up before mutation; the production file is written atomically via
  `os.replace`.
- **Secret leakage via argv**: `lib/prompt_optimizer/claude_runner.py` now passes
  prompts to the `claude` CLI on stdin instead of as command-line arguments, and
  redacts secrets from returned error strings.

### Fixed (P1)
- **Persistent prompt injection via unescaped demos**: dynamic backtick fences in
  `deploy_optimized_prompts.py` and `storage.py` prevent fence breakout; demo
  fields are type-validated.
- **Storage/utils path traversal**: all artifact read/write paths now validate
  names and enforce directory containment (`lib/prompt_optimizer/validation.py`).
- **Unbounded cost**: `MAX_TARGETS` cap in the runner; timeout clamped to
  5–1800s.
- **Unbounded output capture**: stdout/stderr capped at
  `PROMPT_OPTIMIZER_MAX_OUTPUT_BYTES` (default 5 MB).
- **Executable PATH hijack**: `claude` and `python3` are resolved via
  `shutil.which` / `command -v` at startup.

### Fixed (P2)
- `status.json` is generated with `jq` (every value JSON-encoded) instead of
  `sed` string surgery.
- Tool extractor uses word boundaries + longest-alias-first matching, fixing
  `mgrep`→`grep` and `codex-reply`→`codex` mis-extraction.
- Extractors and deploy/storage loaders type-guard malformed untrusted JSON
  instead of crashing.

### Documented (inherent / won't-fix)
- **Prompt-injection → tool execution via `--dangerously-skip-permissions`** is
  inherent to unattended batch operation. Not fake-fixed. Mitigations added:
  prompt-via-stdin, optional env scrubbing (`PROMPT_OPTIMIZER_SCRUB_ENV`),
  opt-out of the bypass (`PROMPT_OPTIMIZER_SKIP_PERMISSIONS=0`), and operational
  guidance in `SECURITY.md`.
- **Heuristic extraction** is a demo-quality signal, not a security/correctness
  gate (documented limitation).

### Added
- `lib/prompt_optimizer/validation.py` — shared `validate_name` / `contained_path`
  helpers used across deploy, storage, utils, and batch.
- `SECURITY.md`, `CHANGELOG.md`.
