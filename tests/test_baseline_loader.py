"""Regression tests for the baseline-scores loader (verification.VerificationSuite).

Guards the fail-open bug found 2026-07-06: a non-numeric metadata entry (e.g.
"<target>_note") in datasets/baseline-scores.json used to make the loader reject
the ENTIRE file, leaving _baseline_scores = {} so every regression check ran
against a 0.000 baseline and silently passed.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

from lib.prompt_optimizer.verification import VerificationSuite


def _suite(tmp_path, data):
    p = tmp_path / "baseline-scores.json"
    p.write_text(json.dumps(data))
    return VerificationSuite(runner=MagicMock(), baseline_scores_path=p)


def test_note_field_does_not_zero_out_baselines(tmp_path):
    vs = _suite(tmp_path, {
        "code-reviewer": 0.466,
        "code-reviewer_note": "expanded to 8 examples 2026-04-19",
        "publication-review-opus": 0.622,
    })
    b = vs._load_baseline_scores()
    assert b.get("code-reviewer") == 0.466
    assert b.get("publication-review-opus") == 0.622
    assert "code-reviewer_note" not in b


def test_all_numeric_file_unchanged(tmp_path):
    vs = _suite(tmp_path, {"a": 0.1, "b": 0.9})
    assert vs._load_baseline_scores() == {"a": 0.1, "b": 0.9}


def test_bool_is_not_treated_as_score(tmp_path):
    # bool is an int subclass; a stray True must not be read as score 1.0.
    vs = _suite(tmp_path, {"a": 0.5, "flag": True})
    b = vs._load_baseline_scores()
    assert b == {"a": 0.5}


def test_non_dict_file_fails_safe(tmp_path):
    p = tmp_path / "baseline-scores.json"
    p.write_text(json.dumps(["not", "a", "dict"]))
    vs = VerificationSuite(runner=MagicMock(), baseline_scores_path=p)
    assert vs._load_baseline_scores() == {}
