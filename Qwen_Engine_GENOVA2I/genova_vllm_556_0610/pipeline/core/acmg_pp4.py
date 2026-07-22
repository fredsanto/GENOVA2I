"""
pipeline/core/acmg_pp4.py — mechanically enforces PP4's single-gene-etiology
gate for well-known genetically heterogeneous conditions.

Real observed failure, TWICE, even after prompts/conclusion.txt was given an
explicit worked negative example naming this exact condition: PP4 applied
for an LDLR variant in "familial hypercholesterolemia" ("a condition with a
known genetic etiology in LDLR") without ever addressing that FH is also
independently caused by APOB and PCSK9 — a textbook heterogeneous condition.
Since prompt-only guidance repeatedly failed on this exact case, this module
mechanically strips PP4 when its own justification names a condition on a
small maintained list of well-known heterogeneous Mendelian conditions,
regardless of how phenotype-specific the match looks.
"""

import re

from pipeline.core.acmg_points import adjust_points_line

_PP4_LINE_RE = re.compile(r"^-\s*PP4\b.*$", re.MULTILINE)

# Well-known genetically heterogeneous conditions (disease name substring,
# case-insensitive) that keep tripping up PP4's single-gene-etiology gate.
# Extend as new recurring cases are observed.
_HETEROGENEOUS_CONDITIONS = (
    "familial hypercholesterolemia",
)


def validate_pp4(conclusion_text: str) -> str:
    """
    Strips a PP4 line whose own justification names a condition known to be
    genetically heterogeneous (multiple independent causal genes) — PP4's
    single-gene-etiology condition cannot be satisfied for these regardless
    of how phenotype-specific the match looks. Adjusts the stated ACMG
    points/classification to match. No-op if no PP4 line, or its named
    condition isn't on the known-heterogeneous list.
    """
    m = _PP4_LINE_RE.search(conclusion_text)
    if not m:
        return conclusion_text

    line_lower = m.group(0).lower()
    if not any(cond in line_lower for cond in _HETEROGENEOUS_CONDITIONS):
        return conclusion_text

    text = _PP4_LINE_RE.sub("", conclusion_text, count=1)
    return adjust_points_line(text, delta=-1)
