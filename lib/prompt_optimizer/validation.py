"""
Shared input-validation helpers for the prompt optimizer.

Centralizes the rules used to sanitize attacker-influenceable identifiers
(agent names, skill names, optimization targets) and to enforce that derived
filesystem paths stay inside an intended base directory.

Security rationale: agent/skill/target names flow into filesystem paths
(``~/.claude/agents/<name>.md``, ``<prompts_dir>/<name>_latest.json``, etc.)
and into subprocess argument lists. Without validation, values such as
``../../etc/passwd`` or ``x'; rm -rf ~ ; #`` enable path traversal and,
historically, shell injection. Every entry point that accepts a name MUST
route it through :func:`validate_name` before using it to build a path.
"""

from __future__ import annotations

import re
from pathlib import Path

# Allow a leading alphanumeric, then alphanumerics plus _ . - (no slashes,
# no whitespace, no quotes, no control chars, no leading dash/dot). Max 80 chars.
# A single trailing/internal dot is allowed but ".." cannot appear because the
# pattern forbids two consecutive dots via the negative lookahead.
_NAME_RE = re.compile(r"^[A-Za-z0-9](?!.*\.\.)[A-Za-z0-9_.-]{0,79}$")


class ValidationError(ValueError):
    """Raised when an identifier or path fails a security check."""


def validate_name(name: str, *, kind: str = "name") -> str:
    """
    Validate an agent/skill/target identifier.

    Args:
        name: The raw identifier.
        kind: Human-readable category for error messages.

    Returns:
        The validated name (unchanged).

    Raises:
        ValidationError: If the name is empty, too long, or contains any
            character outside ``[A-Za-z0-9_.-]``, a leading dash/dot, or a
            ``..`` sequence.
    """
    if not isinstance(name, str):
        raise ValidationError(f"Invalid {kind}: expected string, got {type(name).__name__}")
    if not _NAME_RE.fullmatch(name):
        raise ValidationError(
            f"Invalid {kind}: {name!r}. Must start with an alphanumeric and contain "
            f"only letters, digits, '_', '.', '-' (no '/', no '..', no whitespace, max 80 chars)."
        )
    return name


def contained_path(base: Path, *parts: str) -> Path:
    """
    Join ``parts`` onto ``base`` and guarantee the result stays under ``base``.

    Resolves symlinks/relative segments on both sides and rejects any path that
    escapes the base directory. This is the second line of defense after
    :func:`validate_name` (defense in depth: even a validator regression cannot
    produce an out-of-tree write).

    Args:
        base: The directory the result must remain within.
        parts: Path components to append.

    Returns:
        The resolved, contained path.

    Raises:
        ValidationError: If the joined path escapes ``base``.
    """
    base_resolved = base.expanduser().resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise ValidationError(f"Path escapes base directory: {candidate} not under {base_resolved}")
    return candidate
