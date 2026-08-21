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

from pipeline.core.citations import validate_citations
from pipeline.core.acmg_pp3 import validate_pp3
from pipeline.core.acmg_pp2_bp1 import validate_pp2_bp1
from pipeline.core.acmg_ps1_pm5 import validate_ps1_pm5
from pipeline.core.acmg_bp6 import validate_bp6
from pipeline.core.acmg_pp4 import validate_pp4
from pipeline.core.acmg_points import relabel_all_points_lines

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
    result = validate_citations(result, variant_context)
    result = validate_pp3(result, variant_context)
    result = validate_pp2_bp1(result, variant_context)
    result = validate_ps1_pm5(result, variant_context)
    result = validate_bp6(result)
    result = validate_pp4(result)
    # Unconditional final pass: the validators above only re-derive the
    # classification label (via adjust_points_line -> classify()) when they
    # actively strip a criterion. If none fire, the SLM's own raw "ACMG
    # points: N -> Label" line is never independently checked — a real
    # observed failure had the SLM write "4.5 -> Likely Pathogenic" for a
    # variant whose points value is in the 0-5 VUS band, not 6-9. Re-derive
    # every points line's label from its own stated N unconditionally so this
    # can't slip through undetected.
    return relabel_all_points_lines(result)
