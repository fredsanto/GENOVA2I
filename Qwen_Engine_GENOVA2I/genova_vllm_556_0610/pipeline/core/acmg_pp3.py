"""
pipeline/core/acmg_pp3.py — mechanically enforces the PP3 "multiple lines of
computational evidence" rule.

Prompt-only guidance for this repeatedly failed in practice: the SLM applied
PP3 from a single crossing tool (or misread its own documented threshold
table — e.g. citing an AlphaMissense score BELOW the benign cutoff as
"inconclusive" and still counting it toward PP3). This module recomputes PP3
eligibility directly from the variant's own raw in-silico scores — the same
six tools and thresholds documented in prompts/conclusion.txt — and strips a
PP3 line the model added without at least two tools actually crossing their
deleterious threshold, adjusting the stated point total and classification
label to match.
"""

import re

from pipeline.core.acmg_points import adjust_points_line

_SCORE_FIELD_RE = re.compile(
    r"(REVEL_score|CADD_score|SIFT_score|PolyPhen2_score|AlphaMissense_score|SpliceAI_score)"
    r"=([^,\n]*)"
)

_PP3_LINE_RE = re.compile(r"^-\s*PP3\b.*$\n?", re.MULTILINE)


def _to_float(raw: str) -> float | None:
    raw = raw.strip()
    if not raw or raw.upper() in ("NA", "N/A", "NONE", "."):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _count_deleterious(scores: dict[str, str]) -> int:
    """Count of this variant's own in-silico tools crossing their documented
    deleterious threshold (same six tools/thresholds as prompts/conclusion.txt)."""
    n = 0

    revel = _to_float(scores.get("REVEL_score", ""))
    if revel is not None and revel >= 0.7:
        n += 1

    cadd = _to_float(scores.get("CADD_score", ""))
    if cadd is not None and cadd >= 20:
        n += 1

    sift = _to_float(scores.get("SIFT_score", ""))
    if sift is not None and sift < 0.05:
        n += 1

    # PolyPhen2 comes as either a numeric probability or an ANNOVAR-style
    # categorical call (D=damaging, P=possibly damaging, B=benign) depending
    # on the input CSV's annotation source — handle both.
    poly_raw = scores.get("PolyPhen2_score", "").strip()
    poly_num = _to_float(poly_raw)
    if poly_num is not None:
        if poly_num > 0.85:
            n += 1
    elif poly_raw.upper() in ("D", "DAMAGING", "PROBABLY_DAMAGING", "PROBABLYDAMAGING"):
        n += 1

    am = _to_float(scores.get("AlphaMissense_score", ""))
    if am is not None and am > 0.564:
        n += 1

    sai = _to_float(scores.get("SpliceAI_score", ""))
    if sai is not None and sai >= 0.5:
        n += 1

    return n


def validate_pp3(conclusion_text: str, variant_context: str) -> str:
    """
    If the model's ACMG criteria list includes a PP3 line but fewer than two
    of this variant's own in-silico scores actually cross their deleterious
    threshold, strip that line and reduce the stated ACMG points/classification
    by 1 to match. No-op if PP3 wasn't applied, or if it's properly grounded
    (>= 2 tools crossing threshold).
    """
    if not _PP3_LINE_RE.search(conclusion_text):
        return conclusion_text

    scores = dict(_SCORE_FIELD_RE.findall(variant_context))
    if _count_deleterious(scores) >= 2:
        return conclusion_text

    text = _PP3_LINE_RE.sub("", conclusion_text, count=1)
    return adjust_points_line(text, delta=-1)
