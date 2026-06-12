# Adversarial Security/Correctness/Privacy Review: dspy-prompt-optimizer

_Blind follow-up review by a fresh Claude Fable 5 agent (2026-06-11), with no
knowledge of the prior GPT Pro review or the fixes applied. Archived verbatim._

---

## Summary

This codebase has clearly already been through one hardening pass (the `reviews/gptpro-2026-06-11.md` review and the SECURITY.md document drove it). The *centralized* primitives — `validation.py` (`validate_name`, `contained_path`), `claude_runner.py`, `storage.py`, `deploy_optimized_prompts.py`, `run_optimization.sh` — are genuinely solid. I tried to bypass the name validator with unicode, RTL overrides, ligatures, trailing/embedded newlines, length edges, and `..` variants, and the validator + `contained_path` containment held in every case (empirically tested). The shell script's background re-exec via argv array, jq-based status writing, and stdin prompt delivery are correctly implemented.

The real problems are at the **edges the hardening pass didn't reach**: two CLI entry scripts (`optimize_skill.py`, `run_ab_test.py`) duplicate logic and bypass the very controls the rest of the codebase (and SECURITY.md) relies on.

---

## P1 — Validation/containment bypass in `optimize_skill.py` (reachable arbitrary `.md` read; latent arbitrary write)

`scripts/optimize_skill.py` does **not** use the shared `validate_name` / `contained_path` helpers. It has its own local copies of the path logic that take the raw `--skill` argument straight into filesystem paths:

- `find_skill_path` — `scripts/optimize_skill.py:85-102` (read)
- `save_optimized_skill` — `scripts/optimize_skill.py:121-131` (write, line 126)
- integrated write — `scripts/optimize_skill.py:401`

I demonstrated both at the helper level (isolated, not theory):
```
find_skill_path("../../../../tmp/skilltest/secret")
  -> reads /home/ashita/.claude/skills/../../../../tmp/skilltest/secret/SKILL.md  ("SECRET PROMPT")
save_optimized_skill("../pwned", ..., "/tmp/skilltest/out")
  -> writes /tmp/skilltest/out/../pwned-optimized.md   (escaped the output dir)
```

**Reachability:**
- **Arbitrary read is reachable end-to-end.** `load_skill_prompt(args.skill)` runs unconditionally at `optimize_skill.py:259`, before any mode dispatch and before any `validate_name` gate. `--skill ../../../../etc/foo` will read `/etc/foo/SKILL.md` (or, via the flat fallback, any `<traversed-path>.md`) and feed it into a `claude --dangerously-skip-permissions` run. Practical impact is bounded by the required filename suffix (`SKILL.md` or `*.md`), so it isn't a fully arbitrary read — but it is a clear, unvalidated traversal on the tool's primary skill entry point.
- **Arbitrary write is currently gated only by accident.** Before the local `save_optimized_skill` runs, `optimizer.optimize(agent_name=f"skill-{args.skill}")` calls `DemoStorage.save_demos/save_optimized_prompt`, which *do* call `validate_name`. Because the name is `skill-<arg>`, any `..` or `/` is rejected there and the run crashes before the unvalidated write. This is defense-by-coincidence: removing the `skill-` prefix, catching the `ValidationError`, or reordering the save would immediately expose arbitrary file write through `optimize_skill.py:126`/`:401`.

**Doc-claim check:** SECURITY.md (lines 24) claims "All agent/skill/target names are validated … every derived path is containment-checked … Applies to deploy, storage, utils, and batch." Note it conspicuously does **not** list `optimize_skill.py`/`optimize_agent.py` — the two main CLI entry points a user would actually run. The claim is technically scoped-true but materially misleading.

**Contrast — `optimize_agent.py` is safe (transitively):** its `save_optimized_agent` (`optimize_agent.py:113`) is *also* unvalidated, but `load_agent_prompt` → `find_agent_path` (imported from `utils.py`, which validates) runs first and rejects `..`, gating the write. `optimize_skill.py` lacks this because its `find_skill_path` is a local unvalidated copy rather than the `utils.py` version.

**Fix:** delete the local `find_skill_path`/`save_optimized_skill` in `optimize_skill.py` and route through `utils.find_skill_path` + `contained_path(Path(output_dir), f"{name}-optimized.md")`. Do the same for `save_optimized_agent`. The duplicate path logic is the root cause.

---

## P2 — Secret-via-argv + un-hardened claude invocation in `run_ab_test.py`

`scripts/run_ab_test.py:83-87`:
```python
cmd = ["claude", "--print", "--model", model,
       "--dangerously-skip-permissions",
       "--", prompt]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
```

This passes the full prompt (plan directives + training-data input) as an **argv element**, which directly contradicts SECURITY.md (line 26): *"Prompts are passed to the `claude` CLI on **stdin**, not as argv elements (argv is world-readable via `/proc` on many systems)."* The hardening in `claude_runner._exec` (stdin delivery) was simply never applied to this script. On a multi-user host, any local user can read the in-flight prompt via `/proc/<pid>/cmdline`.

Additional gaps in the same call, all of which `claude_runner.py` handles but this script ignores:
- Bare `"claude"` instead of `shutil.which("claude")` → PATH-hijackable (SECURITY.md line 28 claims both interpreters are resolved at startup).
- `--dangerously-skip-permissions` hardcoded with no `PROMPT_OPTIMIZER_SKIP_PERMISSIONS` opt-out.
- No env scrubbing, no timeout clamping.

It's a dev/A-B helper rather than the core loop, which is why I rate it P2, but it's a live contradiction of three separate SECURITY.md mitigations.

---

## P3 / Informational

**1. Dead symlink guard in `deploy_optimized_prompts.py:128`.** `inject_demos_to_agent` does `if agent_path.is_symlink(): refuse`, but `agent_path` is the return value of `contained_path(...)`, which already called `.resolve()`. I confirmed empirically that a symlink pointing *inside* the agents dir resolves to a path whose `is_symlink()` is `False`, so this branch never fires in the normal case; a symlink pointing *outside* is already rejected by `contained_path` before reaching this line. The real protection is containment, not this check. The code comment ("refusing to follow symlinks: a symlinked agent file could redirect the write") overstates what the check does. Not exploitable (containment holds), but the guard is effectively no-op code and the comment should be corrected. (There's also a benign TOCTOU between `contained_path`'s resolve and the later `open`/`os.replace`, but it requires pre-existing write access inside `~/.claude`, so it's not meaningfully exploitable.)

**2. Redaction asymmetry (privacy).** `claude_runner.redact_secrets` is applied to error strings/stderr (`_exec` lines 184/199) but **not** to the model's `stdout`/`output`, which is saved as demos and later injected into agent `.md` files (`storage.save_optimized_prompt`, `deploy_optimized_prompts.py`). If a run induces the model to echo a secret it read from the environment or filesystem (very possible given `--dangerously-skip-permissions`), that secret is persisted unredacted into `~/.claude/prompt_optimizer/` and deployed into agent prompts. Partly by-design (demos *are* model output), but the asymmetry is worth a note and a `--dry-run` review remains the only backstop.

**3. "Unbounded memory" cap is partial.** `_exec` uses `subprocess.run(capture_output=True)`, which buffers the **entire** child stdout/stderr in memory, and only *afterward* checks `len(stdout) > MAX_OUTPUT_BYTES` (`claude_runner.py:188`). A runaway child can therefore peak-allocate well beyond the cap before the check rejects it. The SECURITY.md "Unbounded cost / memory" mitigation (line 29) prevents *storing/processing* oversized output, not the peak buffer. Minor accuracy caveat.

---

## Things I checked that are NOT vulnerable (verified, not assumed)

- **Name validator**: `_NAME_RE` with `fullmatch` correctly rejects trailing/embedded newlines (the classic `$`-matches-before-`\n` bypass does not apply here — fullmatch leaves the `\n` unconsumed), unicode/RTL/ligatures (ASCII-only char class), `..`, leading `-`/`.`, `/`, whitespace, `;`, and >80 chars. Tested all empirically.
- **No shell injection sinks**: grepped the whole tree — no `shell=True`, no `os.system`, no `eval`/`exec`, no `bash -c`/`sh -c`. All `subprocess.run` calls use argv lists. `create_training_data.extract_from_git` builds git argv from an int (`max_commits`) and operator `repo_path` — no injection.
- **run_optimization.sh**: the background path re-execs via a proper argv array (not `bash -c "...$VAR..."`), validates `ALGORITHM`/`MODEL`/`CV_FOLDS`/`DROPOUT`/`TARGETS` against allowlists/char-classes with a per-target `..`/leading-dash check and a `MAX_TARGETS` cap, and writes `status.json` purely through `jq` argjson encoding. I could not get injected target text to escape into shell execution or corrupt the JSON.
- **The core RCE-by-design** (dataset `input` → `claude --dangerously-skip-permissions` with full tool access) is real and severe, but it is honestly and prominently disclosed in SECURITY.md §"Inherent residual risk" with concrete operational mitigations (`PROMPT_OPTIMIZER_SCRUB_ENV`, `PROMPT_OPTIMIZER_SKIP_PERMISSIONS=0`, sandbox guidance). Not counted as a finding since it's inherent and documented — though note the P2 `run_ab_test.py` path silently undercuts two of those documented knobs.

**Net:** the documentation is mostly honest, but it describes the *library's* security posture while two shipped CLI scripts (`optimize_skill.py`, `run_ab_test.py`) sit outside it. The P1 is the one I'd fix before trusting the SECURITY.md containment claims at face value.

---

## Disposition of this review (added by remediation lead, 2026-06-11)

- **P1 (`optimize_skill.py`)** — FIXED. Local `find_skill_path` deleted; now routes through `utils.find_skill_path` (validated + contained). `save_optimized_skill` and the integrated write use `validate_name` + `contained_path`. `args.skill` is validated at the top of `main()` before any path/claude use. `optimize_agent.py` `save_optimized_agent` hardened identically and `args.agent` validated up front.
- **P2 (`run_ab_test.py`)** — FIXED. `run_claude` now routes through the hardened `ClaudeRunner` (stdin delivery, `shutil.which`, timeout clamp, redaction, `PROMPT_OPTIMIZER_SKIP_PERMISSIONS` honored).
- **P3-1 (dead symlink guard)** — FIXED. Symlink check now inspects the raw, unresolved path; comment corrected.
- **P3-2 (redaction asymmetry)** — DOCUMENTED. SECURITY.md now states stdout is intentionally unredacted (demos are model output) with `--dry-run` as the backstop.
- **P3-3 (partial memory cap)** — DOCUMENTED. SECURITY.md caveat clarifies the cap bounds retained/processed output, not the transient peak buffer.
