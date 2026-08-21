"""
pipeline/core/acmg_points.py — shared "**ACMG points:** N → Classification"
line parsing/rewriting, used by every mechanical ACMG-criterion validator
(acmg_pp3.py, acmg_pp2_bp1.py, ...) that strips a criterion the SLM applied
without adequate grounding and needs to keep the stated total consistent.
"""

import re

_POINTS_LINE_RE = re.compile(
    r"(\*\*(?:Total )?ACMG points:\*\*\s*)([-+]?\d+(?:\.\d+)?)(\s*→\s*)([A-Za-z /()]+)"
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


def relabel_all_points_lines(text: str) -> str:
    """
    Re-derives the classification label on EVERY "**[Total ]ACMG points:** N →
    Label" line in `text` from N via classify(), leaving N itself unchanged.
    Fixes the SLM occasionally writing an internally-inconsistent label for
    its own stated total (e.g. "4.5 → Likely Pathogenic" when 4.5 is in the
    0-5 VUS band, not the 6-9 Likely Pathogenic band) — a real observed
    failure where a homozygous ClinVar Likely-Pathogenic variant scored 4.5
    points, got mislabeled "Likely Pathogenic" instead of "Uncertain
    Significance (VUS)", and as a result fell through every section of the
    Clinical Conclusion (not causative since 4.5 < 6, not Notable VUS since
    its label wasn't "VUS", so the SCOPE RULE omitted it from the report
    entirely). Unlike adjust_points_line(), fixes ALL matches in the text
    (a compound-het pair block has two such lines), not just the first.
    """

    def _relabel(m: re.Match) -> str:
        prefix, points_str, arrow, _old_label = m.groups()
        try:
            points = float(points_str)
        except ValueError:
            return m.group(0)
        return f"{prefix}{points_str}{arrow}{classify(points)}"

    return _POINTS_LINE_RE.sub(_relabel, text)
