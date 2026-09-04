"""
pipeline/core/acmg_pvs1.py — mechanically enforces AutoPVS1's own verdict
against two genuine observed failures.

First failure: the model applying PVS1 anyway when AutoPVS1 already examined
this exact variant and returned "PVS1 applicable: False".

Real case: ARID2 c.706-7A>G (intron, position -7 — not a canonical ±1/2
splice site). AutoPVS1's own block for this variant read "PVS1 applicable:
False" with every other field (path/steps/strength) blank — AutoPVS1
determined no qualifying LoF pathway exists. The model applied PVS1
[VeryStrong, +8] anyway, justified only as "supported by gene constraint and
variant location at a canonical splice site" (false — position -7 is not
canonical), turning a would-be VUS into a false-positive Pathogenic/Likely
Pathogenic call. A near-identical case (POGZ c.1186-10T>G, intron position
-10) produced the same failure. Telling the model in prompts/conclusion.txt
that AutoPVS1's verdict is authoritative did not reliably stop this — this
module enforces it deterministically instead, the same way acmg_pp2_bp1.py
enforces the CLINVAR_GENE_STATS verdict against a plausibility-based PP2/BP1
substitution.

Second failure: AutoPVS1 doesn't always resolve to a clean True/False — it
can be gated ON for a variant (Type=indel/deletion, no usable HGVS/frame
info) and still come back with no output at all (unresolvable coordinates,
gene mismatch, fetch/parse failure). A missing verdict line was originally
treated identically to "AutoPVS1 was never gated on for this variant type at
all" and left the model's own Type/HGVS judgment untouched — but the model
then credited full VeryStrong PVS1 (+8) from gene-level constraint alone
(pLI/LOEUF), never having confirmed the variant is actually a null allele.

Real case: MN1 chr22:28195637 — Ref_seq/Var_seq show a 21 bp deletion,
exactly divisible by 3 (an in-frame 7-codon deletion, not a frameshift), with
HGVS=NA/Transcript=NA. AutoPVS1 was gated on (Type=indel) but returned no
resolvable output. The model wrote "PVS1 [VeryStrong, +8]: Loss-of-function
variant in a LoF-intolerant gene (*MN1* pLI=1.000) where haploinsufficiency
is the likely disease mechanism" — a gene-constraint argument standing in for
a variant-level null-allele finding that was never actually made; an in-frame
deletion does not itself satisfy PVS1's null-variant requirement.

This module now caps PVS1 at Strong/+4 (one tier below AutoPVS1-confirmed
VeryStrong/+8, on this codebase's own Strength->points scale: VeryStrong=8,
Strong=4, Moderate=2, Supporting=1) when AutoPVS1 gave no verdict at all, and
only when the variant's own Type/HGVS is unambiguous null — frameshift,
nonsense/stop-gain, or canonical splice ±1/2. Anything else in that
situation — a bare "indel"/"deletion" Type with no frame evidence, missense,
deep intronic (ARID2/POGZ-style), UTR, etc. — gets PVS1 stripped entirely,
same as an explicit AutoPVS1 "False".
"""

import re

from pipeline.core.acmg_points import adjust_points_line
from pipeline.core.protein_change import PROTEIN_CHANGE_RE

_PVS1_LINE_RE = re.compile(r"^-\s*PVS1\b.*$\n?", re.MULTILINE)

# The criterion bullet's own strength/points tag, e.g. "[VeryStrong, +8]" —
# captured separately from _PVS1_LINE_RE so the capped-credit branch can
# rewrite just the tag and leave the justification text after it untouched.
_PVS1_TAG_RE = re.compile(
    r"^-\s*PVS1\s*\[([A-Za-z\s]+),\s*\+?(\d+(?:\.\d+)?)\s*(?:pts?)?\]",
    re.MULTILINE,
)

_AUTOPVS1_APPLICABLE_RE = re.compile(r"PVS1 applicable:\s*(True|False)")

_TYPE_RE = re.compile(r"Type=([^,\n]*)")
_HGVS_RE = re.compile(r"HGVS=(.*?),\s*Zygosity=")

# Canonical splice site: exactly ±1 or ±2 from the exon boundary. Deep
# intronic (±3 and beyond, e.g. the ARID2 c.706-7A>G case above) must NOT
# match this.
_CANONICAL_SPLICE_RE = re.compile(r"c\.\d+[+-][12][A-Za-z]")

_CAPPED_LABEL  = "Strong"
_CAPPED_POINTS = 4.0


def _is_unambiguous_null_variant(variant_context: str) -> bool:
    """
    True only for a variant whose own Type/HGVS unambiguously predicts a
    null allele: frameshift, nonsense/stop-gain, or canonical ±1/2 splice.
    Deliberately narrower than a bare Type=="indel"/"deletion" match (that
    covers in-frame indels too, which do not on their own satisfy PVS1 — see
    the MN1 case above) and narrower than autopvs1.py's own gate list, which
    exists only to decide whether an AutoPVS1 lookup is worth attempting,
    not to award points when that lookup comes back empty.
    """
    type_m = _TYPE_RE.search(variant_context)
    hgvs_m = _HGVS_RE.search(variant_context)
    variant_type = (type_m.group(1) if type_m else "").strip().lower()
    hgvs         = (hgvs_m.group(1) if hgvs_m else "").strip()

    if "frameshift" in variant_type or "fs" in hgvs.lower():
        return True
    if any(m in variant_type for m in ("nonsense", "stopgain", "stop-gain", "stop_gain")):
        return True
    m = PROTEIN_CHANGE_RE.search(hgvs)
    if m and m.group(3).upper() in ("TER", "*", "X"):
        return True
    if _CANONICAL_SPLICE_RE.search(hgvs):
        return True
    return False


def validate_pvs1(conclusion_text: str, variant_context: str) -> str:
    """
    Enforces a three-way rule on any applied PVS1 criterion:

      - AutoPVS1 ran and returned "PVS1 applicable: True"  -> leave as-is,
        trust AutoPVS1's own strength.
      - AutoPVS1 ran and returned "PVS1 applicable: False" -> strip PVS1
        entirely.
      - AutoPVS1 gave no verdict at all (didn't run, or ran and returned no
        resolvable result) -> PVS1 may only be credited, capped at
        Strong/+4, when the variant's own Type/HGVS is unambiguous null
        (frameshift, nonsense/stop-gain, canonical splice ±1/2). Any other
        variant in this situation gets PVS1 stripped.

    No-op if PVS1 isn't in the criteria list, or (case 1) AutoPVS1 already
    agrees.
    """
    m_tag = _PVS1_TAG_RE.search(conclusion_text)
    if m_tag is None:
        return conclusion_text

    original_points = float(m_tag.group(2))

    m_verdict = _AUTOPVS1_APPLICABLE_RE.search(variant_context)

    if m_verdict is not None and m_verdict.group(1) == "True":
        return conclusion_text  # AutoPVS1 confirmed — trust its strength

    if m_verdict is not None and m_verdict.group(1) == "False":
        text = _PVS1_LINE_RE.sub("", conclusion_text, count=1)
        return adjust_points_line(text, delta=-original_points)

    # AutoPVS1 gave no verdict at all.
    if _is_unambiguous_null_variant(variant_context):
        if original_points <= _CAPPED_POINTS:
            return conclusion_text  # already at or below the cap
        capped_text = _PVS1_TAG_RE.sub(
            f"- PVS1 [{_CAPPED_LABEL}, +{int(_CAPPED_POINTS)}]",
            conclusion_text,
            count=1,
        )
        return adjust_points_line(capped_text, delta=_CAPPED_POINTS - original_points)

    text = _PVS1_LINE_RE.sub("", conclusion_text, count=1)
    return adjust_points_line(text, delta=-original_points)
