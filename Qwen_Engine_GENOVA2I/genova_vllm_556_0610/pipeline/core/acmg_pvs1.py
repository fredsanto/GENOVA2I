"""
pipeline/core/acmg_pvs1.py — mechanically enforces AutoPVS1's own verdict
against a genuine observed failure: the model applying PVS1 anyway when
AutoPVS1 already examined this exact variant and returned
"PVS1 applicable: False".

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

Only strips PVS1 when AutoPVS1 explicitly ran for this variant and returned
"PVS1 applicable: False". When AutoPVS1 is absent/gated-off entirely (no
"PVS1 applicable:" line in this variant's context at all — it only runs on
LoF-typed variants to begin with), this is a no-op: the model's own
Type/HGVS-based PVS1 judgment stands, per conclusion.txt's fallback rule.
"""

import re

from pipeline.core.acmg_points import adjust_points_line

_PVS1_LINE_RE = re.compile(r"^-\s*PVS1\b.*$\n?", re.MULTILINE)
_AUTOPVS1_APPLICABLE_RE = re.compile(r"PVS1 applicable:\s*(True|False)")


def validate_pvs1(conclusion_text: str, variant_context: str) -> str:
    """
    Strips a PVS1 line when this variant's own AUTOPVS1 evidence block
    explicitly states "PVS1 applicable: False". No-op if PVS1 isn't in the
    criteria list, or AutoPVS1 didn't run for this variant, or AutoPVS1
    itself says applicable: True.
    """
    if not _PVS1_LINE_RE.search(conclusion_text):
        return conclusion_text

    m = _AUTOPVS1_APPLICABLE_RE.search(variant_context)
    if m is None or m.group(1) == "True":
        return conclusion_text  # AutoPVS1 didn't run, or agrees — leave as-is

    text = _PVS1_LINE_RE.sub("", conclusion_text, count=1)
    return adjust_points_line(text, delta=-8)  # removing a +8 item subtracts 8
