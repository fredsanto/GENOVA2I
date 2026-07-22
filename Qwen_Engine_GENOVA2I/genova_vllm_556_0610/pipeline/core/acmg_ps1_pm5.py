"""
pipeline/core/acmg_ps1_pm5.py — mechanically ADDS a PS1/PM5 criterion the SLM
omitted despite the deterministic CLINVAR_RESIDUE_SEARCH evidence supporting it.

Real observed failure: the SAME evidence block (containing both a "SAME amino
acid change ... Likely pathogenic" hit and a "DIFFERENT amino acid change ...
Likely pathogenic" hit at the same residue) led the model to write only ONE
of PS1/PM5 into its criteria list — and inconsistently which one across
otherwise-identical runs — despite prompts/conclusion.txt explicitly stating
both may apply from separate hits. This module re-derives PS1/PM5
eligibility directly from the tool's own structured tags (SAME/DIFFERENT
amino acid change, [PATHOGENIC/LIKELY PATHOGENIC] flag, Variation ID) and
ADDS whichever criterion the evidence supports but the model's own list is
missing — the mirror image of acmg_pp3.py / acmg_pp2_bp1.py, which STRIP
wrongly-applied criteria; this one fills in under-applied ones.
"""

import re

from pipeline.core.acmg_points import adjust_points_line
from pipeline.core.acmg_sf import is_pathogenic_clinvar

# Matches the CLINVAR RESIDUE-LEVEL SEARCH block up to the next blank-line +
# capitalized section header, or end of string.
_RESIDUE_BLOCK_RE = re.compile(r"CLINVAR RESIDUE-LEVEL SEARCH.*?(?=\n\n[A-Z]|\Z)", re.DOTALL)

# One hit line, e.g.:
#   - c.3961T>A (p.Cys1321Ser): Likely pathogenic (...) — SAME amino acid
#     change as this variant [PATHOGENIC/LIKELY PATHOGENIC] — ClinVar
#     Variation ID 99904 (https://.../99904/)
_HIT_RE = re.compile(
    r"-\s*(?P<nt>c\.\S+)\s*\(p\.(?P<protein>[A-Za-z*]+\d+[A-Za-z*]+)\):\s*"
    r"(?P<classification>[^—\n]+?)\s*—\s*(?P<tag>SAME|DIFFERENT) amino acid change"
    r"[^\n]*?ClinVar Variation ID (?P<vid>\d+)"
)

# This variant's own cDNA token(s) — scoped to the HGVS= field specifically,
# NOT run over the whole variant_context blob (which also contains the
# CLINVAR RESIDUE-LEVEL SEARCH block's OTHER variants' own cDNA notations —
# extracting over the full text would misclassify every hit as "this
# variant's own record" and silently find nothing). Same multi-transcript-
# aware lesson as build_detection.py: a pipe-separated HGVS carries a
# different cDNA number per transcript for the same physical variant.
_HGVS_FIELD_RE = re.compile(r"HGVS=(.*?)(?:,\s*Zygosity=|\n)")
_CDNA_TOKEN_RE = re.compile(r"c\.[^\s:;|]+")


def _own_cdnas(variant_context: str) -> set[str]:
    m = _HGVS_FIELD_RE.search(variant_context)
    if not m:
        return set()
    return set(_CDNA_TOKEN_RE.findall(m.group(1)))

_PS1_LINE_RE = re.compile(r"^-\s*PS1\b.*$", re.MULTILINE)
_PM5_LINE_RE = re.compile(r"^-\s*PM5\b.*$", re.MULTILINE)
_CRITERIA_INSERT_RE = re.compile(r"(?=\*\*ACMG points:\*\*)")


def _find_hits(variant_context: str) -> list[dict]:
    m = _RESIDUE_BLOCK_RE.search(variant_context)
    if not m:
        return []
    block = m.group(0)
    hits = []
    for hm in _HIT_RE.finditer(block):
        hits.append({
            "nt": hm.group("nt"),
            "protein": hm.group("protein"),
            "classification": hm.group("classification").strip(),
            "tag": hm.group("tag"),
            "vid": hm.group("vid"),
            "pathogenic": is_pathogenic_clinvar(hm.group("classification")),
        })
    return hits


def validate_ps1_pm5(conclusion_text: str, variant_context: str) -> str:
    """
    Scans the CLINVAR RESIDUE-LEVEL SEARCH evidence (if present) for a
    PS1-eligible hit (SAME amino acid change, pathogenic/likely pathogenic,
    a different nucleotide than this variant's own) and a PM5-eligible hit
    (DIFFERENT amino acid change, pathogenic/likely pathogenic). Adds
    whichever the model's own criteria list is missing, with the ClinVar
    Variation ID/URL inline, and adjusts the stated points/classification.
    No-op if no CLINVAR RESIDUE-LEVEL SEARCH block, or both are already
    present, or neither hit type exists in the evidence.
    """
    hits = _find_hits(variant_context)
    if not hits:
        return conclusion_text

    own_cdnas = _own_cdnas(variant_context)

    has_ps1 = bool(_PS1_LINE_RE.search(conclusion_text))
    has_pm5 = bool(_PM5_LINE_RE.search(conclusion_text))
    if has_ps1 and has_pm5:
        return conclusion_text

    ps1_hit = None
    pm5_hit = None
    for h in hits:
        if h["nt"] in own_cdnas or not h["pathogenic"]:
            continue  # this variant's own record, or not an established P/LP precedent
        if h["tag"] == "SAME" and ps1_hit is None:
            ps1_hit = h
        elif h["tag"] == "DIFFERENT" and pm5_hit is None:
            pm5_hit = h

    text = conclusion_text
    delta = 0.0

    if ps1_hit and not has_ps1:
        line = (
            f"- PS1 [Strong, +4]: Same amino acid change (p.{ps1_hit['protein']}) as a "
            f"previously established {ps1_hit['classification']} allele ({ps1_hit['nt']}) at "
            f"the same residue, via a different nucleotide change — ClinVar Variation ID "
            f"{ps1_hit['vid']} (https://www.ncbi.nlm.nih.gov/clinvar/variation/{ps1_hit['vid']}/) "
            "[auto-added from CLINVAR RESIDUE-LEVEL SEARCH — evidence supported this but it "
            "was missing from the model's own criteria list].\n"
        )
        text = _CRITERIA_INSERT_RE.sub(line, text, count=1)
        delta += 4

    if pm5_hit and not has_pm5:
        line = (
            f"- PM5 [Moderate, +2]: Different missense change (p.{pm5_hit['protein']}) at the "
            f"same residue as this variant, previously established as {pm5_hit['classification']} "
            f"— ClinVar Variation ID {pm5_hit['vid']} "
            f"(https://www.ncbi.nlm.nih.gov/clinvar/variation/{pm5_hit['vid']}/) "
            "[auto-added from CLINVAR RESIDUE-LEVEL SEARCH — evidence supported this but it "
            "was missing from the model's own criteria list].\n"
        )
        text = _CRITERIA_INSERT_RE.sub(line, text, count=1)
        delta += 2

    if delta:
        text = adjust_points_line(text, delta)

    return text
