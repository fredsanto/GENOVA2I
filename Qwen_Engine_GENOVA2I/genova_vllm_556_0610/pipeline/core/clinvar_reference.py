"""
pipeline/core/clinvar_reference.py — mechanically appends this variant's own
ClinVar reference (variation ID + URL) to a MOI-layer/final-report block.

The base conclusion stage's "**ClinVar:**" field is prose (classification +
condition), not a clickable reference, and MOI-layer stages only copy the
"ACMG criteria:" list verbatim from it (see prompts/moi_*.txt) — the ClinVar
ID itself is never guaranteed to reach the final printed report. Rather than
ask yet another SLM call to remember to carry it forward (the exact failure
mode this session repeatedly ran into for other fields), this reads the
variation ID directly out of the retrieval-stage evidence text — ClinGenAlleleTool's
"ClinVar variation ID: NNNN" line (this variant's own cross-referenced allele)
or, failing that, ClinVarGeneStatsTool's submission-tally header — and appends
a plain reference line deterministically.
"""

import re

_CLINGEN_ID_RE = re.compile(r"ClinVar variation ID:\s*(\d+)")
_TALLY_ID_RE   = re.compile(r"CLINVAR VARIANT-LEVEL SUBMISSION TALLY \(variation ID (\d+)")

_REFERENCE_MARKER = "**ClinVar reference:**"


def _extract_variation_id(evidence_context: str) -> str | None:
    m = _CLINGEN_ID_RE.search(evidence_context)
    if m:
        return m.group(1)
    m = _TALLY_ID_RE.search(evidence_context)
    if m:
        return m.group(1)
    return None


def append_clinvar_reference(block: str, evidence_context: str) -> str:
    """
    Appends "**ClinVar reference:** <url>" (or an explicit "not found" note)
    to `block`, unless it already contains a ClinVar reference line. Always
    states one or the other — silence here reads as "not checked", which is
    worse than an explicit negative.
    """
    if _REFERENCE_MARKER in block:
        return block  # already present (MOI prompt happened to carry it forward)

    return block.rstrip() + "\n\n" + _reference_line(evidence_context)


def append_clinvar_references(block: str, labeled_contexts: list[tuple[str, str]]) -> str:
    """
    Same as append_clinvar_reference, for MOI blocks covering MULTIPLE
    variants (e.g. a compound-het pair) — one reference line per
    (label, evidence_context) entry, each resolved independently so a
    variant with no match doesn't shadow one that does.
    """
    if _REFERENCE_MARKER in block:
        return block

    lines = [f"{_REFERENCE_MARKER} {label}: {_reference_line(ctx, bare=True)}" for label, ctx in labeled_contexts]
    return block.rstrip() + "\n\n" + "\n".join(lines)


def _reference_line(evidence_context: str, bare: bool = False) -> str:
    """`bare=True` omits the leading marker (caller supplies its own prefix)."""
    variation_id = _extract_variation_id(evidence_context)
    prefix = "" if bare else f"{_REFERENCE_MARKER} "
    if variation_id:
        # Not prefixed "VCV{id}" — the zero-padded VCV accession format isn't
        # reconstructible from the bare numeric ID alone; the URL form is
        # unambiguous and always correct regardless of accession formatting.
        return (
            f"{prefix}ClinVar Variation ID {variation_id} — "
            f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/"
        )
    return f"{prefix}Not found in ClinVar (novel/unresolvable allele)."
