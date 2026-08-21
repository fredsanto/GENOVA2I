"""
pipeline/stages/moi_recessive.py — MOI Layer 5: recessive / compound-het analysis.

For gene groups with >=2 kept variants and a recessive-relevant inheritance
mode (AR, XLR, AD_AR, XLD_XLR), where the TRANS phase has already been
confirmed (or left unknown-but-permitted) by the deterministic Python gate in
pipeline.py BEFORE this stage ever runs — a CIS pair never reaches here. Adds
PM3 (in-trans with a pathogenic/likely-pathogenic variant) on top of each
variant's Layer 2 base ACMG score — never re-scores base criteria.

Prompt loaded from prompts/moi_recessive.txt.

Public API:
    run_pair(gene, variant_a_label, variant_a_context, variant_a_base_conclusion,
              variant_b_label, variant_b_context, variant_b_base_conclusion,
              phase, cross_analysis, llm) -> str
    run_solo(gene, variant_label, variant_context, variant_base_conclusion,
              cross_analysis, llm) -> str
        For a CONFIRMED HOMOZYGOUS variant in a recessive-relevant gene with no
        compound-het partner — homozygosity alone satisfies the biallelic
        requirement, so this variant belongs in the Recessive layer even
        though it never goes through run_pair()'s >=2-variant TRANS gate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core.citations import validate_citations
from pipeline.core.clinvar_reference import append_clinvar_reference, append_clinvar_references
from pipeline.core.acmg_points import relabel_all_points_lines

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_recessive.txt"
_SOLO_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_recessive_homozygous.txt"

MAX_NEW_TOKENS_RECESSIVE = 900
MAX_NEW_TOKENS_RECESSIVE_SOLO = 600

# A compound-het pair is only a confirmed P/LP biallelic diagnosis if BOTH
# variants independently clear the Likely Pathogenic threshold — one strong
# variant does not rescue a weak/VUS partner (the partner's own base score
# already reflects that it does not, on its own, explain a recessive disease).
# Computed deterministically here (not left to the LLM) because the SLM
# repeatedly let a single >=6 partner carry the whole pair as "causative"
# during final synthesis, ignoring the weak partner entirely.
_CAUSATIVE_THRESHOLD = 6.0

_TOTAL_POINTS_RE = re.compile(
    r"\*\*Total ACMG points:\*\*\s*([-+]?\d+(?:\.\d+)?)\s*→"
)


def _joint_compound_het_status(pair_block: str) -> str | None:
    """
    Parse both variants' own "**Total ACMG points:** N →" lines (in Variant A,
    Variant B order, per the fixed template below) and deterministically decide
    whether this pair is a confirmed compound-het P/LP diagnosis or a compound
    VUS. Returns None (no-op) if the expected two totals can't be found, so a
    malformed LLM response degrades gracefully instead of raising.
    """
    matches = _TOTAL_POINTS_RE.findall(pair_block)
    if len(matches) < 2:
        logger.warning(
            "[MOIRecessive] Could not find two 'Total ACMG points' lines in pair "
            "block — skipping joint compound-het classification."
        )
        return None
    try:
        points_a, points_b = float(matches[0]), float(matches[1])
    except ValueError:
        return None

    if points_a >= _CAUSATIVE_THRESHOLD and points_b >= _CAUSATIVE_THRESHOLD:
        return (
            f"CAUSATIVE — compound heterozygous Pathogenic/Likely Pathogenic. "
            f"Both alleles independently reach the Likely Pathogenic threshold "
            f"(Variant A: {points_a:g} pts, Variant B: {points_b:g} pts; both >= "
            f"{_CAUSATIVE_THRESHOLD:g})."
        )
    return (
        f"COMPOUND VUS — NOT a confirmed causative biallelic diagnosis. "
        f"At least one allele is below the Likely Pathogenic threshold "
        f"(Variant A: {points_a:g} pts, Variant B: {points_b:g} pts; both must "
        f"independently reach >= {_CAUSATIVE_THRESHOLD:g}). Report this gene as "
        f"compound VUS, not as Pathogenic/Likely Pathogenic, even though one "
        f"partner individually clears the threshold — a single strong allele "
        f"does not establish a biallelic recessive diagnosis on its own."
    )


def _inject_joint_status(pair_block: str, joint_status: str | None) -> str:
    if joint_status is None:
        return pair_block
    marker = "\n## Variant A"
    if marker not in pair_block:
        logger.warning(
            "[MOIRecessive] '## Variant A' marker not found — could not inject "
            "joint compound-het classification line."
        )
        return pair_block
    return pair_block.replace(
        marker,
        f"\n**Joint compound-het classification:** {joint_status}\n{marker}",
        1,
    )

_PHASE_LABELS = {
    "trans":   "TRANS confirmed (different parents) — compound heterozygous model applies.",
    "unknown": "UNKNOWN (phase not determinable from available allelic-balance data) — "
               "compound heterozygous model may still apply per biallelic evidence, "
               "but phase uncertainty must be noted explicitly.",
}

# classify_segregation() outputs (pipeline.core.segregation), relabelled for the
# homozygous-solo PM3 decision. "homozygous_parent" is the hard negative gate —
# every other outcome is treated as "not contradicted" (PM3 may still apply at
# reduced weight; see prompts/moi_recessive_homozygous.txt for the point logic).
_SEGREGATION_LABELS = {
    "both_carriers": "Both parents confirmed heterozygous carriers by allelic balance "
                      "(~0.5 each), proband homozygous (~1.0) — classic autosomal "
                      "recessive segregation pattern. Not contradicted; PM3 may apply "
                      "per the rules below.",
    "homozygous_parent": "At least one parent's allelic balance is itself ~1.0 "
                      "(homozygous) for this variant — an \"unaffected\" carrier "
                      "parent should be heterozygous, not homozygous. Segregation is "
                      "DISCORDANT with simple biallelic transmission from two carrier "
                      "parents — do NOT apply PM3 (see rule 2a).",
    "insufficient_data": "No parental allelic-balance data available — segregation not "
                      "verified. Not contradicted; PM3 may still apply at reduced weight "
                      "per the rules below (parental carrier status assumed but unconfirmed).",
    "uncertain": "Parental allelic-balance data present but does not match a clean "
                      "carrier pattern — segregation ambiguous. Not contradicted; PM3 "
                      "may still apply at reduced weight per the rules below.",
}
_SEGREGATION_DEFAULT_LABEL = (
    "Segregation pattern does not match the expected two-carrier-parents pattern for "
    "a homozygous proband — treat as unconfirmed. Not contradicted; PM3 may still "
    "apply at reduced weight per the rules below."
)


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_recessive prompt not found at {_PROMPT_PATH}.")


def _load_solo_prompt() -> str:
    if _SOLO_PROMPT_PATH.exists():
        return _SOLO_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_recessive_homozygous prompt not found at {_SOLO_PROMPT_PATH}.")


def run_pair(
    gene: str,
    variant_a_label: str,
    variant_a_context: str,
    variant_a_base_conclusion: str,
    variant_b_label: str,
    variant_b_context: str,
    variant_b_base_conclusion: str,
    phase: str,
    cross_analysis: str | None,
    llm: "LLMClient",
) -> str:
    """
    MOI Layer 5 — recessive/compound-het analysis for one gene's variant pair.

    Args:
        gene:                          Gene symbol.
        variant_a_label, variant_b_label: Display labels (e.g. "Variant 3 — LRTOMT (p.Leu117Val)").
        variant_a_context, variant_b_context: Per-variant context strings from retrieval.
        variant_a_base_conclusion, variant_b_base_conclusion: Layer 2's full structured
                          output for each variant — ground truth this stage adds to.
        phase:            classify_phase() result — must be "trans" or "unknown"; callers
                           must never call this with "cis" (that pair should have been
                           excluded before reaching this stage).
        cross_analysis:    Gene-level cross_analysis.run() text, or None if unavailable.
        llm:               Shared LLMClient instance.

    Returns:
        Structured recessive layer block covering both variants, each with its own
        "Total ACMG points: [base+delta] → [classification]".
    """
    if phase not in _PHASE_LABELS:
        raise ValueError(
            f"moi_recessive.run_pair called with phase={phase!r} for gene={gene} — "
            "only 'trans' or 'unknown' may reach this stage; 'cis' must be excluded upstream."
        )

    logger.info("[MOIRecessive] Generating recessive analysis for gene %s (phase=%s)...", gene, phase)

    template = _load_prompt()
    cross_analysis_block = (
        f"GENE-LEVEL CROSS-ANALYSIS:\n{cross_analysis}\n" if cross_analysis is not None else ""
    )

    user_prompt = (template
        .replace("{gene}", gene)
        .replace("{variant_a_label}", variant_a_label)
        .replace("{variant_a_base_conclusion}", variant_a_base_conclusion)
        .replace("{variant_b_label}", variant_b_label)
        .replace("{variant_b_base_conclusion}", variant_b_base_conclusion)
        .replace("{phase_block}", _PHASE_LABELS[phase])
        .replace("{cross_analysis_block}", cross_analysis_block))

    result = llm.generate(
        system="You are an expert clinical geneticist evaluating compound heterozygous/biallelic recessive inheritance. Limit your response to 500 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_RECESSIVE,
    )
    result = relabel_all_points_lines(result)
    full_context = variant_a_context + "\n" + variant_b_context + "\n" + variant_a_base_conclusion + "\n" + variant_b_base_conclusion
    result = validate_citations(result, full_context)
    joint_status = _joint_compound_het_status(result)
    logger.info("[MOIRecessive] Joint compound-het status for %s: %s", gene, joint_status)
    result = _inject_joint_status(result, joint_status)
    return append_clinvar_references(result, [
        (variant_a_label, variant_a_context + "\n" + variant_a_base_conclusion),
        (variant_b_label, variant_b_context + "\n" + variant_b_base_conclusion),
    ])


def run_solo(
    gene: str,
    variant_label: str,
    variant_context: str,
    variant_base_conclusion: str,
    segregation: str,
    cross_analysis: str | None,
    llm: "LLMClient",
) -> str:
    """
    MOI Layer 5 (solo path) — confirmed-homozygous recessive analysis for one
    variant with no compound-het partner.

    Args:
        gene:                     Gene symbol.
        variant_label:            Display label (e.g. "Variant 3 — PMM2 (p.Thr237Met)").
        variant_context:          Per-variant context string from retrieval.
        variant_base_conclusion:  Layer 2's full structured output — ground truth this
                                   stage documents against (does not re-score it).
        segregation:              pipeline.core.segregation.classify_segregation() result
                                   for this variant's trio AB data — a hard Python gate on
                                   PM3 eligibility (see _SEGREGATION_LABELS): a parent who
                                   is themselves homozygous blocks PM3 outright.
        cross_analysis:           Gene-level cross_analysis.run() text, or None if unavailable.
        llm:                      Shared LLMClient instance.

    Returns:
        Structured recessive layer block for a single confirmed-homozygous variant,
        with "Total ACMG points: [base + PM3 delta] → [classification]" where the PM3
        delta is +0, +0.5 (consanguinity not excluded), or +1 (consanguinity explicitly
        excluded in evidence) per ClinGen SVI's homozygous-PM3 guidance.
    """
    logger.info("[MOIRecessive] Generating homozygous-solo recessive analysis for gene %s...", gene)

    template = _load_solo_prompt()
    cross_analysis_block = (
        f"GENE-LEVEL CROSS-ANALYSIS:\n{cross_analysis}\n" if cross_analysis is not None else ""
    )
    segregation_block = _SEGREGATION_LABELS.get(segregation, _SEGREGATION_DEFAULT_LABEL)

    user_prompt = (template
        .replace("{gene}", gene)
        .replace("{variant_label}", variant_label)
        .replace("{variant_base_conclusion}", variant_base_conclusion)
        .replace("{segregation_block}", segregation_block)
        .replace("{cross_analysis_block}", cross_analysis_block))

    result = llm.generate(
        system="You are an expert clinical geneticist evaluating a confirmed homozygous recessive variant. Limit your response to 400 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_RECESSIVE_SOLO,
    )
    result = relabel_all_points_lines(result)
    full_context = variant_context + "\n" + variant_base_conclusion
    result = validate_citations(result, full_context)
    return append_clinvar_reference(result, full_context)
