"""
pipeline/stages/conclusion.py — SLM structured report stage.

Ported from run_conclusion() in server_main.py.

Single SLM call per variant. Takes one variant's context string, its reasoning
output, and an optional gene-level cross-analysis text, then produces the
structured clinical block for that variant.

The Clinical Conclusion paragraph (overall synthesis across all variants) is
generated separately by stages/final_conclusion.py.

Prompt loaded from prompts/conclusion.txt.

Public API:
    run_one(variant_context, reasoning, cross_analysis, llm) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core.errors import SLMError
from pipeline.core.citations import validate_citations
from pipeline.core.acmg_pp3 import validate_pp3
from pipeline.core.acmg_pp2_bp1 import validate_pp2_bp1
from pipeline.core.acmg_ps1_pm5 import validate_ps1_pm5
from pipeline.core.acmg_bp6 import validate_bp6
from pipeline.core.acmg_pp4 import validate_pp4, validate_pp4_full_coverage
from pipeline.core.acmg_pvs1 import validate_pvs1
from pipeline.core.acmg_points import relabel_all_points_lines, recompute_and_fix_totals

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "conclusion.txt"

MAX_NEW_TOKENS_REPORT = 1500


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Conclusion prompt not found at {_PROMPT_PATH}. "
        "Run step 19 to extract prompts from server_main.py."
    )


def _has_acmg_criteria_section(text: str) -> bool:
    """False when the response is missing its "ACMG criteria:" heading —
    the prompt (see prompts/conclusion.txt) always requires this heading,
    even for a variant with zero applicable criteria ("None apply" is still
    written under it). Its absence means the SLM response was truncated or
    otherwise malformed before reaching that section, not that the model
    legitimately found nothing — a real observed failure: a large,
    uncompressed GeneReviews chapter for the variant's gene bloated the
    prompt enough that MAX_NEW_TOKENS_REPORT ran out mid-response, silently
    dropping the ACMG criteria (and total-points) lines. Downstream MOI
    layers and acmg_points.recompute_and_fix_totals() have no criteria
    bullets to sum in that case and silently leave the SLM's own broken
    "[Base score + N] -> [...]" placeholder text untouched — the variant
    then vanishes from the final Clinical Conclusion with no error anywhere.
    Failing loudly here, before that silent chain starts, is the fix."""
    return "ACMG criteria" in text


def run_one(
    variant_context: str,
    reasoning: str,
    cross_analysis: str | None,
    llm: "LLMClient",
) -> str:
    """
    Stage 4 — Structured clinical report for one variant.

    Args:
        variant_context: Per-variant context string from Stage 1 (retrieval).
                         Contains PATIENT DATA header + one VARIANT block with tool outputs.
        reasoning:       Output from reasoning.run_one() for this variant.
        cross_analysis:  Gene-level cross-analysis text from cross_analysis.run(), or None
                         if this variant's gene has fewer than two variants in this run.
        llm:             Shared LLMClient instance.

    Returns:
        Structured report block for this variant:
            # Variant [N] — [GENE] ([HGVS])
            **Molecular mechanism:** ...
            **Phenotype fit:** ...
            **Inheritance check:** ...
            **Evidence strength:** ...
            **ACMG criteria**: ...
            **Comment:** ...
    """
    logger.info("[Conclusion] Generating structured report for variant...")

    template = _load_prompt()

    cross_analysis_block = (
        f"GENE-LEVEL CROSS-ANALYSIS:\n{cross_analysis}\n"
        if cross_analysis is not None else ""
    )

    user_prompt = (template
        .replace("{augmented_context}", variant_context)
        .replace("{reasoning_output}", reasoning)
        .replace("{cross_analysis_block}", cross_analysis_block))

    result = llm.generate(
        system="You are an expert clinical geneticist. Limit your response to 1000 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_REPORT,
    )
    if not _has_acmg_criteria_section(result):
        raise SLMError(
            "Conclusion response missing required 'ACMG criteria' section "
            f"(likely truncated — response ended with: ...{result[-200:]!r})"
        )
    result = validate_citations(result, variant_context)
    result = validate_pp3(result, variant_context)
    result = validate_pp2_bp1(result, variant_context)
    result = validate_ps1_pm5(result, variant_context)
    result = validate_bp6(result)
    result = validate_pp4(result)
    # reasoning (not reasoning_output/variant_context) carries the
    # backend-injected "PHENOTYPE CLUSTER MATCH" verdict block pipeline.py's
    # _cluster_match_block() appended — the authoritative source for PP4's
    # FULL COVERAGE condition, checked independently of whatever the model's
    # own prose in `result` claims about phenotype fit.
    result = validate_pp4_full_coverage(result, reasoning)
    result = validate_pvs1(result, variant_context)
    # Unconditional final pass: the validators above only adjust the stated
    # total when THEY strip a criterion. They never check whether the SLM's
    # own original total already matched its own criteria list — a real,
    # recurring observed failure: a list totalling PS3(+4)+PM2(+2)+PP4(+1) =
    # 7 stated as "ACMG points: 11", with nothing here to catch it since no
    # criterion needed stripping. Re-sum every points/classification line
    # from its own preceding criteria bullets and correct the stated total
    # (and label) whenever it disagrees.
    result = recompute_and_fix_totals(result)
    # Final safety net: re-derive any remaining line's label from its own
    # stated N (e.g. "4.5 -> Likely Pathogenic" when 4.5 is in the 0-5 VUS
    # band, not 6-9) — covers the case recompute_and_fix_totals() left
    # untouched (no criteria bullets found to check against).
    return relabel_all_points_lines(result)
