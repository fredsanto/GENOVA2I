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

# moi_*.py's "**Base ACMG points:** N -> Label" line (the copied-verbatim
# base total, before that layer's own delta) — a third total-line format
# alongside _POINTS_LINE_RE ("ACMG points:"/"Total ACMG points:") and
# _CLASSIFICATION_LINE_RE below ("ACMG classification:"). Kept separate from
# _POINTS_LINE_RE (rather than making "Total" one of several optional
# prefixes) so adjust_points_line()/relabel_all_points_lines() — used
# elsewhere against the "Total" total specifically — don't start matching
# the "Base" line too.
_BASE_POINTS_LINE_RE = re.compile(
    r"(\*\*Base ACMG points:\*\*\s*)([-+]?\d+(?:\.\d+)?)(\s*→\s*)([A-Za-z /()]+)"
)

# final_conclusion.py's per-variant format: "**ACMG classification:** Label
# (N pts total)" — distinct from the "**ACMG points:** N -> Label" format
# above (conclusion.py / moi_*.py), so recompute_and_fix_totals() needs to
# recognize and rewrite both.
_CLASSIFICATION_LINE_RE = re.compile(
    r"(\*\*ACMG classification:\*\*\s*)([A-Za-z /()]+?)(\s*\(\s*)([-+]?\d+(?:\.\d+)?)(\s*pts total\s*\))"
)

# A criterion bullet's own point tag, e.g. "[VeryStrong, +8]" or
# "[Moderate, +2 pts]" — every criterion line carries exactly one of these
# per the fixed format both conclusion.txt and clinical_conclusion.txt
# require.
_CRITERION_TAG_RE = re.compile(r"\[[A-Za-z\s]+,\s*([+-]?\d+(?:\.\d+)?)\s*(?:pts?)?\]")

# moi_*.py's own MOI-layer delta line, e.g. "**Recessive delta:** +2" /
# "**De novo delta:** +4" / "**X-linked delta:** +1" — sits between a
# "**Base ACMG points:**" line and the "**Total ACMG points:**" line that
# should equal their sum.
_DELTA_LINE_RE = re.compile(r"\*\*[A-Za-z-]+ delta:\*\*\s*([+-]?\d+(?:\.\d+)?)")

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


def _sum_str(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def recompute_and_fix_totals(text: str) -> str:
    """
    Deterministically re-sums each criterion's own stated point tag (e.g.
    the "+4" in "PS3 [Strong, +4]:") over the contiguous run of criterion
    bullet lines immediately above a stated total line, and overwrites that
    total — and its classification label — whenever it disagrees, instead
    of trusting the SLM's own arithmetic on it.

    A real, RECURRING observed failure — the exact scenario conclusion.txt's
    own "real past failure" warning already describes, which happened again
    anyway: a criteria list totalling PS3(+4)+PM2(+2)+PP4(+1) = 7 stated as
    "ACMG points: 11". The stale 11 then carried forward unchanged through a
    MOI layer's "+2" delta into a final reported total of 13, with every
    downstream stage trusting the upstream total rather than the actual
    listed criteria. Prompt instructions to self-check this arithmetic
    exist and are followed only sometimes — this makes the check
    unconditional.

    Handles all three rendered total-line formats used across the pipeline:
    "**ACMG points:**"/"**Base ACMG points:** N -> Label" (a criteria-bullet
    list precedes these directly — conclusion.py's own total, and each
    moi_*.py layer's copied-verbatim base total), "**Total ACMG points:** N
    -> Label" (a "**<Layer> delta:** +/-D" line precedes this instead of
    bullets — reconciled as base + delta in a second pass below, using the
    (possibly just-corrected) Base line), and "**ACMG classification:**
    Label (N pts total)" (final_conclusion.py's per-variant blocks, bullets
    precede directly like the first case).
    """
    lines = text.split("\n")

    def _sum_preceding_criteria(idx: int) -> float | None:
        total = 0.0
        found_any = False
        j = idx - 1
        while j >= 0:
            line = lines[j]
            if not line.strip():
                break
            m = _CRITERION_TAG_RE.search(line)
            if not m:
                break
            total += float(m.group(1))
            found_any = True
            j -= 1
        return total if found_any else None

    for i, line in enumerate(lines):
        m = _POINTS_LINE_RE.search(line) or _BASE_POINTS_LINE_RE.search(line)
        if m:
            actual = _sum_preceding_criteria(i)
            if actual is not None and actual != float(m.group(2)):
                lines[i] = (
                    line[: m.start()]
                    + f"{m.group(1)}{_sum_str(actual)}{m.group(3)}{classify(actual)}"
                    + line[m.end() :]
                )
            continue
        m2 = _CLASSIFICATION_LINE_RE.search(line)
        if m2:
            actual = _sum_preceding_criteria(i)
            if actual is not None and actual != float(m2.group(4)):
                lines[i] = (
                    line[: m2.start()]
                    + f"{m2.group(1)}{classify(actual)}{m2.group(3)}{_sum_str(actual)}{m2.group(5)}"
                    + line[m2.end() :]
                )

    # Second pass: reconcile "**Total ACMG points:**" against
    # base-line-just-above-it + delta-line-in-between, now that the base
    # line has already been corrected above if it needed it. Only fires
    # when both neighbours are actually present in that exact shape.
    for i, line in enumerate(lines):
        if "Total ACMG points:" not in line:
            continue
        mt = _POINTS_LINE_RE.search(line)
        if not mt or i < 2:
            continue
        md = _DELTA_LINE_RE.search(lines[i - 1])
        mb = _BASE_POINTS_LINE_RE.search(lines[i - 2])
        if not md or not mb:
            continue
        expected = float(mb.group(2)) + float(md.group(1))
        if expected != float(mt.group(2)):
            lines[i] = (
                line[: mt.start()]
                + f"{mt.group(1)}{_sum_str(expected)}{mt.group(3)}{classify(expected)}"
                + line[mt.end() :]
            )

    return "\n".join(lines)
