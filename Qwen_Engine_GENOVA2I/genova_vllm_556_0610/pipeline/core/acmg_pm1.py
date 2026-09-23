"""
pipeline/core/acmg_pm1.py — grounds PM1 in the deterministic CLINVAR HOTSPOT
WINDOW evidence (pipeline/tools/clinvar_hotspot.py, VarSome hotspot rule).

PM1 is kept or added only when that block states "PM1 HOTSPOT: MET"; every
model-written PM1 line is stripped otherwise (no block, NOT MET, or NOT
EVALUABLE) — a gene-level statement that a domain exists says nothing about
whether this residue sits in a pathogenic-variant cluster free of benign
variation.
"""

import re

from pipeline.core.acmg_points import adjust_points_line

_VERDICT_RE = re.compile(r"PM1 HOTSPOT:\s*(MET|NOT MET|NOT EVALUABLE)")
_WINDOW_LINE_RE = re.compile(r"^[ \t]*Residue p\.\S+, window .*— MET$", re.MULTILINE)
_PM1_LINE_RE = re.compile(r"^-\s*PM1(?:_\w+)?\b.*\n?", re.MULTILINE)
_CRITERIA_INSERT_RE = re.compile(r"(?=\*\*ACMG points:\*\*)")


def _hotspot_met(variant_context: str) -> bool:
    m = _VERDICT_RE.search(variant_context)
    return bool(m) and m.group(1) == "MET"


def validate_pm1(conclusion_text: str, variant_context: str) -> str:
    """
    Strips every PM1 line unless the CLINVAR HOTSPOT WINDOW verdict is MET;
    when it is MET and the model's list has no PM1, adds PM1 [Moderate, +2]
    citing the window counts and adjusts the stated points.

    Stripping does NOT adjust stated points — callers must follow with
    acmg_points.recompute_and_fix_totals(), same as strip_ungrounded_ps1_pm5.
    """
    if not _hotspot_met(variant_context):
        return _PM1_LINE_RE.sub("", conclusion_text)
    if _PM1_LINE_RE.search(conclusion_text):
        return conclusion_text

    wm = _WINDOW_LINE_RE.search(variant_context)
    detail = wm.group(0).strip().removesuffix(" — MET") if wm else "ClinVar hotspot window"
    line = (
        f"- PM1 [Moderate, +2]: Located in a mutational hotspot — {detail} "
        "(CLINVAR HOTSPOT WINDOW, VarSome hotspot rule) [auto-added — evidence "
        "supported this but it was missing from the model's own criteria list].\n"
    )
    text = _CRITERIA_INSERT_RE.sub(line, conclusion_text, count=1)
    if text == conclusion_text:
        return text
    return adjust_points_line(text, 2)
