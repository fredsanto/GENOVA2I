"""
pipeline/core/acmg_points.py — shared "**ACMG points:** N → Classification"
line parsing/rewriting, used by every mechanical ACMG-criterion validator
(acmg_pp3.py, acmg_pp2_bp1.py, ...) that strips a criterion the SLM applied
without adequate grounding and needs to keep the stated total consistent.
"""

import re

_POINTS_LINE_RE = re.compile(
    r"(\*\*ACMG points:\*\*\s*)([-+]?\d+(?:\.\d+)?)(\s*→\s*)([A-Za-z /()]+)"
)

# (inclusive lower bound, label) — highest first; matches the thresholds
# block at the bottom of prompts/conclusion.txt.
_THRESHOLDS = [
    (10.0, "Pathogenic"),
    (6.0,  "Likely Pathogenic"),
    (0.0,  "Uncertain Significance (VUS)"),
    (-6.0, "Likely Benign"),
    (float("-inf"), "Benign"),
]


def classify(points: float) -> str:
    for lo, label in _THRESHOLDS:
        if points >= lo:
            return label
    return "Benign"


def adjust_points_line(text: str, delta: float) -> str:
    """
    Adds `delta` to the stated "**ACMG points:** N → Label" total and
    rewrites the classification label to match. No-op if the line isn't
    found. Safe to call repeatedly (e.g. once per stripped criterion) since
    it re-reads the current value from `text` each time.
    """
    def _adjust(m: re.Match) -> str:
        prefix, points_str, arrow, _old_label = m.groups()
        try:
            new_points = float(points_str) + delta
        except ValueError:
            return m.group(0)
        new_points_str = str(int(new_points)) if new_points == int(new_points) else str(new_points)
        return f"{prefix}{new_points_str}{arrow}{classify(new_points)}"

    return _POINTS_LINE_RE.sub(_adjust, text, count=1)
